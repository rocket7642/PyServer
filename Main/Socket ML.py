import socket
import sys
import threading
import FreeSimpleGUI as sg
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
# TensorBoard for visualization - helps us understand what the agent is learning
from torch.utils.tensorboard import SummaryWriter
import datetime
import traceback  # For detailed error reporting
import time

HOST = "127.0.0.1"
PORT = 25000

units = []
eUnits = []
eKUnits = []

# Global variables for map data (assigned after connection)
global map_heights, mass_spots, map_width, map_height
map_heights = None
normalized_map_heights = None
mass_spots = []
map_spots_norm = []
map_width = 0
map_height = 0
map_height_min = 0.0
map_height_max = 1.0

# Cached map embedding (since map is static)
cached_map_embedding = None
cached_map_embedding_device = None

# === STANDARDIZED MAP SETTINGS ===
STANDARD_MAP_WIDTH = 1024
STANDARD_MAP_HEIGHT = 1024
STANDARD_MAP_Y = 256

# === ACTION SETTINGS (NOW CONTINUOUS TARGETS) ===
# We now allow the agent to move to ANY point on the map by selecting a target
# (x, z) directly. We still keep a NOOP option.
NOOP_ACTION = "NOOP"

# Sampling settings for continuous target selection
NUM_RANDOM_TARGETS = 30   # Number of random candidate targets per decision
TARGET_PADDING = 50       # Avoid sampling too close to map edges

# Command persistence settings
MIN_STEPS_BETWEEN_COMMANDS = 3  # Prefer to let commands finish
COMMAND_DISTANCE_EPS = 50       # Consider command "complete" if within 50 units
OVERRIDE_SCORE_THRESHOLD = 10.0 # Override existing command if new score is much better
CANCEL_COMMAND_PENALTY = 2.0    # Small penalty for cancelling an in-flight command

# === EPISODIC TRAINING SETTINGS ===
EPISODE_TIMEOUT_SECONDS = 60
MASS_REACH_RADIUS = 200  # World units (normalized by normalize_distance)
MAX_SEGMENT_STEPS = 300  # Hard cap to prevent buffer bloat per unit

# Potential-based move rewards
POTENTIAL_DISTANCE_SCALE = 0.1
POTENTIAL_DIRECTION_SCALE = 0.05
HEIGHT_JUMP_TOLERANCE = 15.0  # In normalized Y units
HEIGHT_JUMP_PENALTY_SCALE = 0.2

# Segment (mass-spot) rewards
SEGMENT_BASE_REWARD = 200.0
FAILURE_BASE_PENALTY = 150.0
SEGMENT_TIME_PENALTY = 0.5
SEGMENT_DISTANCE_PENALTY = 0.1
SEGMENT_DAMAGE_PENALTY = 1.0
HEIGHT_DISTANCE_FACTOR = 0.2

# Number of action features for the potential field
NUM_ACTION_FEATURES = 7

# === ENCODER SETTINGS ===
# Fixed-size embeddings per branch keep the LSTM input stable even if raw features change.
SELF_FEATURES_SIZE = 6   # x, z, y, health, friendly_count, enemy_count
SELF_EMBED_SIZE = 16
MASS_FEATURES_SIZE = 3   # dx, dz, distance to nearest mass
MASS_EMBED_SIZE = 8
MAP_FEATURES_SIZE = 6    # summary stats of surrounding heights
MAP_EMBED_SIZE = 16
UNIT_FEATURES_SIZE = 3   # dx, dz, health
ENEMY_FEATURES_SIZE = 4  # dx, dz, health, range
FRIENDLY_EMBED_SIZE = 16
ENEMY_EMBED_SIZE = 16

# === LSTM SETTINGS ===
# LSTM lets the agent keep memory across time steps to learn long-term effects
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 1

# Feature-Based RL Agent
class RTSAgent(nn.Module):
    def __init__(self, input_size, num_features):
        """
        Feature-based agent that learns weights for action features.
        Instead of outputting Q-values for each action, it outputs feature weights.
        """
        super(RTSAgent, self).__init__()
        # Branch encoders (fixed-size outputs)
        self.self_encoder = nn.Sequential(
            nn.Linear(SELF_FEATURES_SIZE, 32),
            nn.ReLU(),
            nn.Linear(32, SELF_EMBED_SIZE)
        )
        self.mass_encoder = nn.Sequential(
            nn.Linear(MASS_FEATURES_SIZE, 16),
            nn.ReLU(),
            nn.Linear(16, MASS_EMBED_SIZE)
        )
        self.map_encoder = nn.Sequential(
            nn.Linear(MAP_FEATURES_SIZE, 32),
            nn.ReLU(),
            nn.Linear(32, MAP_EMBED_SIZE)
        )
        self.map_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.map_fc = nn.Sequential(
            nn.LayerNorm(16),
            nn.Linear(16, MAP_EMBED_SIZE)
        )
        
        # Initialize CNN weights properly to prevent NaN
        self._initialize_cnn_weights()
        
        self.friendly_unit_encoder = nn.Sequential(
            nn.Linear(UNIT_FEATURES_SIZE, 32),
            nn.ReLU(),
            nn.Linear(32, FRIENDLY_EMBED_SIZE)
        )
        self.enemy_unit_encoder = nn.Sequential(
            nn.Linear(ENEMY_FEATURES_SIZE, 32),
            nn.ReLU(),
            nn.Linear(32, ENEMY_EMBED_SIZE)
        )
        # LSTM processes sequences to capture long-term dependencies
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            batch_first=True
        )
        # Fully-connected head to convert LSTM output to feature weights
        self.fc1 = nn.Linear(LSTM_HIDDEN_SIZE, 64)
        self.fc2 = nn.Linear(64, num_features)  # Output feature weights

    def _initialize_cnn_weights(self):
        """Initialize CNN weights using He/Kaiming initialization for ReLU."""
        for module in self.map_cnn.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        
        # Initialize the FC layer after CNN
        for module in self.map_fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x, hidden=None):
        """
        Forward pass through LSTM + FC head.
        x shape: (batch, seq_len, input_size)
        hidden: (h0, c0) for LSTM (optional)
        Returns:
            feature_weights: (batch, num_features)
            new_hidden: updated LSTM hidden state
        """
        lstm_out, new_hidden = self.lstm(x, hidden)
        # Use the last timestep output for decision making
        last_out = lstm_out[:, -1, :]
        x = torch.relu(self.fc1(last_out))
        feature_weights = self.fc2(x)
        return feature_weights, new_hidden

    def encode_state_parts(self, agent_unit, friendly_units, enemy_units):
        """Encode raw game state into branch embeddings."""
        device = next(self.parameters()).device

        # Self branch (normalized positions)
        unit_nx = normalize_x(agent_unit['x'])
        unit_nz = normalize_z(agent_unit['z'])
        unit_ny = normalize_y(agent_unit['y'])
        self_features = torch.tensor([
            unit_nx,
            unit_nz,
            unit_ny,
            agent_unit['health'],
            float(len(friendly_units)),
            float(len(enemy_units))
        ], dtype=torch.float32, device=device)
        self_emb = self.self_encoder(self_features)

        # Mass branch (nearest mass spot)
        if map_spots_norm:
            nearest_mass = min(map_spots_norm, key=lambda p: (p[0] - unit_nx)**2 + (p[1] - unit_nz)**2)
            dx = nearest_mass[0] - unit_nx
            dz = nearest_mass[1] - unit_nz
            dist = (dx ** 2 + dz ** 2) ** 0.5
        else:
            dx, dz, dist = 0.0, 0.0, 0.0
        mass_features = torch.tensor([dx, dz, dist], dtype=torch.float32, device=device)
        mass_emb = self.mass_encoder(mass_features)

        # Map branch (cached)
        map_emb = get_cached_map_embedding(device)

        # Friendly branch (mean pooled unit embeddings)
        friendly_vectors = []
        for u in friendly_units:
            if u['id'] == agent_unit['id']:
                continue
            friendly_vectors.append([
                normalize_x(u['x']) - unit_nx,
                normalize_z(u['z']) - unit_nz,
                u['health']
            ])
        if friendly_vectors:
            friendly_tensor = torch.tensor(friendly_vectors, dtype=torch.float32, device=device)
            friendly_embeds = self.friendly_unit_encoder(friendly_tensor)
            friendly_emb = torch.mean(friendly_embeds, dim=0)
        else:
            friendly_emb = torch.zeros(FRIENDLY_EMBED_SIZE, dtype=torch.float32, device=device)

        # Enemy branch (mean pooled unit embeddings)
        enemy_vectors = []
        for u in enemy_units:
            enemy_vectors.append([
                normalize_x(u['x']) - unit_nx,
                normalize_z(u['z']) - unit_nz,
                u['health'],
                normalize_range(u['range']) if 'range' in u else 0.0
            ])
        if enemy_vectors:
            enemy_tensor = torch.tensor(enemy_vectors, dtype=torch.float32, device=device)
            enemy_embeds = self.enemy_unit_encoder(enemy_tensor)
            enemy_emb = torch.mean(enemy_embeds, dim=0)
        else:
            enemy_emb = torch.zeros(ENEMY_EMBED_SIZE, dtype=torch.float32, device=device)

        return self_emb, mass_emb, map_emb, friendly_emb, enemy_emb

    def encode_state(self, agent_unit, friendly_units, enemy_units):
        """Encode raw game state into a fixed-size embedding vector."""
        self_emb, mass_emb, map_emb, friendly_emb, enemy_emb = self.encode_state_parts(
            agent_unit, friendly_units, enemy_units
        )
        state = torch.cat([self_emb, mass_emb, map_emb, friendly_emb, enemy_emb], dim=0)
        state = torch.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        return state

    def encode_state_no_map(self, agent_unit, friendly_units, enemy_units):
        """Encode state without the (static) map embedding."""
        self_emb, mass_emb, _, friendly_emb, enemy_emb = self.encode_state_parts(
            agent_unit, friendly_units, enemy_units
        )
        state = torch.cat([self_emb, mass_emb, friendly_emb, enemy_emb], dim=0)
        state = torch.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        return state

ENCODER_OUTPUT_SIZE = SELF_EMBED_SIZE + MASS_EMBED_SIZE + MAP_EMBED_SIZE + FRIENDLY_EMBED_SIZE + ENEMY_EMBED_SIZE
agent = RTSAgent(input_size=ENCODER_OUTPUT_SIZE, num_features=NUM_ACTION_FEATURES)
optimizer = optim.Adam(agent.parameters(), lr=0.001)
criterion = nn.MSELoss()

# === TENSORBOARD VISUALIZATION SETUP ===
# Create a unique run name with timestamp so we can compare different training sessions
run_name = f"feature_based_agent_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(f'runs/{run_name}')
step_counter = 0  # Global step counter for logging to TensorBoard
print(f"TensorBoard logging to: runs/{run_name}")
print("View with: tensorboard --logdir=runs")

previous_healths = {}
previous_states = {}
previous_states_no_map = {}
previous_actions = {}
previous_positions = {}  # Track previous positions for movement penalty
previous_y_positions = {}  # Track previous y position for training
previous_distances = {}  # Track distance to nearest mass spot
previous_targets = {}    # Track the last target sent to each unit
previous_action_scores = {}  # Track last action score to decide overrides
previous_command_steps = {}  # Steps since last command
cancel_command_penalties = {}  # Pending penalties for cancelling in-flight commands

# LSTM hidden state per unit (for long-term memory)
lstm_hidden_states = {}
previous_lstm_hidden_states = {}

visited_mass_spots = set()
visited_mass_spots_norm = set()

consecutive_inactive = {}

last_mass_visit = {}

# Episodic tracking per unit
segment_buffers = {}
segment_stats = {}

def init_lstm_hidden(batch_size=1):
    """Initialize LSTM hidden state (h0, c0) with zeros."""
    h0 = torch.zeros(LSTM_NUM_LAYERS, batch_size, LSTM_HIDDEN_SIZE)
    c0 = torch.zeros(LSTM_NUM_LAYERS, batch_size, LSTM_HIDDEN_SIZE)
    return (h0, c0)

def normalize_x(x):
    return (x / map_width) * STANDARD_MAP_WIDTH if map_width > 0 else x

def normalize_z(z):
    return (z / map_height) * STANDARD_MAP_HEIGHT if map_height > 0 else z

def denormalize_x(nx):
    return (nx / STANDARD_MAP_WIDTH) * map_width if map_width > 0 else nx

def denormalize_z(nz):
    return (nz / STANDARD_MAP_HEIGHT) * map_height if map_height > 0 else nz

def normalize_range(rng):
    if map_width > 0 and map_height > 0:
        scale = (STANDARD_MAP_WIDTH / map_width + STANDARD_MAP_HEIGHT / map_height) / 2.0
        return rng * scale
    return rng

def normalize_distance(dist):
    if map_width > 0 and map_height > 0:
        scale = (STANDARD_MAP_WIDTH / map_width + STANDARD_MAP_HEIGHT / map_height) / 2.0
        return dist * scale
    return dist

def normalize_y(y):
    if np.isfinite(map_height_min) and np.isfinite(map_height_max) and map_height_max > map_height_min:
        return (y - map_height_min) / (map_height_max - map_height_min) * STANDARD_MAP_Y
    return y

def denormalize_y(ny):
    if np.isfinite(map_height_min) and np.isfinite(map_height_max) and map_height_max > map_height_min:
        return map_height_min + (ny / STANDARD_MAP_Y) * (map_height_max - map_height_min)
    return ny

def build_normalized_height_map():
    global normalized_map_heights, map_height_min, map_height_max, cached_map_embedding, cached_map_embedding_device

    if map_heights is None or map_heights.size == 0:
        normalized_map_heights = None
        return

    # Sanitize input heights to avoid NaN/Inf propagation
    heights = np.array(map_heights, dtype=np.float32)
    heights = np.nan_to_num(heights, nan=0.0, posinf=0.0, neginf=0.0)

    map_height_min = float(np.nanmin(heights))
    map_height_max = float(np.nanmax(heights))
    if not np.isfinite(map_height_min) or not np.isfinite(map_height_max):
        map_height_min = 0.0
        map_height_max = 1.0

    src_h, src_w = heights.shape
    x_idx = np.linspace(0, src_w - 1, STANDARD_MAP_WIDTH).astype(int)
    z_idx = np.linspace(0, src_h - 1, STANDARD_MAP_HEIGHT).astype(int)
    normalized = heights[np.ix_(z_idx, x_idx)]

    if map_height_max > map_height_min:
        normalized = (normalized - map_height_min) / (map_height_max - map_height_min) * STANDARD_MAP_Y

    normalized_map_heights = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    cached_map_embedding = None
    cached_map_embedding_device = None

def height_at_normalized(nx, nz):
    if normalized_map_heights is None:
        return -1000.0

    map_x = int(max(0, min(STANDARD_MAP_WIDTH - 1, nx)))
    map_z = int(max(0, min(STANDARD_MAP_HEIGHT - 1, nz)))
    return float(normalized_map_heights[map_z, map_x])

def get_cached_map_embedding(device):
    global cached_map_embedding, cached_map_embedding_device
    if normalized_map_heights is None:
        return torch.zeros(MAP_EMBED_SIZE, dtype=torch.float32, device=device)

    if cached_map_embedding is not None and cached_map_embedding_device == device:
        return cached_map_embedding

    map_array = np.array(normalized_map_heights, dtype=np.float32, copy=True)
    map_array = np.nan_to_num(map_array, nan=0.0, posinf=0.0, neginf=0.0)
    map_mean = np.mean(map_array)
    map_std = np.std(map_array)
    if np.isfinite(map_std) and map_std > 0:
        map_array = (map_array - map_mean) / map_std
    else:
        map_array = map_array - map_mean
    map_array = np.nan_to_num(map_array, nan=0.0, posinf=0.0, neginf=0.0)

    map_tensor = torch.tensor(map_array, dtype=torch.float32, device=device)
    map_tensor = map_tensor.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        map_cnn_out = agent.map_cnn(map_tensor)
        map_emb = agent.map_fc(map_cnn_out.squeeze(0))
        map_emb = torch.nan_to_num(map_emb, nan=0.0, posinf=0.0, neginf=0.0)

    cached_map_embedding = map_emb.detach()
    cached_map_embedding_device = device
    return cached_map_embedding

def reconstruct_state_with_map(state_no_map):
    """Reinsert cached map embedding between mass and friendly embeddings."""
    split_idx = SELF_EMBED_SIZE + MASS_EMBED_SIZE
    device = state_no_map.device
    map_emb = get_cached_map_embedding(device)
    prefix = state_no_map[:split_idx]
    suffix = state_no_map[split_idx:]
    return torch.cat([prefix, map_emb, suffix], dim=0)

def parse_units(message, header):
    # Parse the units string into list of dicts
    # Assuming format: header\nid name x y z range hp\n...\nEND\n (possibly multiple sections)
    units_list = []
    lines = message.strip().split('\n')
    i = 0
    while i < len(lines):
        if lines[i].strip() == header:
            i += 1
            while i < len(lines) and lines[i].strip() != "END":
                line = lines[i]
                if line.strip():
                    parts = line.split(' ')
                    if len(parts) >= 7:
                        unit = {
                            'id': int(parts[0]),
                            'name': parts[1],
                            'x': float(parts[2]),
                            'y': float(parts[3]),
                            'z': float(parts[4]),
                            'health': float(parts[6])  # hp is the 7th field
                        }
                        units_list.append(unit)
                i += 1
        i += 1
    return units_list

def normalize_image(img_array):
    """Normalize an image array to [0,1] for TensorBoard visualization."""
    if img_array.size == 0:
        return img_array
    min_val = np.min(img_array)
    max_val = np.max(img_array)
    if max_val - min_val == 0:
        return np.zeros_like(img_array, dtype=np.float32)
    return ((img_array - min_val) / (max_val - min_val)).astype(np.float32)

def generate_enemy_range_image(enemy_units, map_w, map_h, map_heights_shape, enemy_range=300):
    """
    Create a 2D image where enemy ranges are marked as high values.
    This helps visualize danger zones the agent should avoid.
    """
    if map_heights_shape is None:
        return None
    h, w = map_heights_shape
    img = np.zeros((h, w), dtype=np.float32)
    if map_width > 0 and map_height > 0:
        range_scale = (STANDARD_MAP_WIDTH / map_width + STANDARD_MAP_HEIGHT / map_height) / 2.0
    else:
        range_scale = 1.0
    for enemy in enemy_units:
        ex = int(normalize_x(enemy['x']))
        ez = int(normalize_z(enemy['z']))
        r = int(enemy_range * range_scale)
        x_min = max(0, ex - r)
        x_max = min(w - 1, ex + r)
        z_min = max(0, ez - r)
        z_max = min(h - 1, ez + r)
        for z in range(z_min, z_max + 1):
            for x in range(x_min, x_max + 1):
                # Circular range mask
                if (x - ex) ** 2 + (z - ez) ** 2 <= r ** 2:
                    img[z, x] = 1.0
    return img

def get_local_view_image(unit_x, unit_z, map_heights_array, view_size=250): # WORK ON, PROVIDING INCORRECT HIGHLIGHTS
    """
    Extract a local view (square patch) around the unit as an image.
    This represents what the agent is "seeing" locally.
    """
    if map_heights_array is None:
        return None
    h, w = map_heights_array.shape
    cx = int(normalize_x(unit_x))
    cz = int(normalize_z(unit_z))
    half = view_size // 2
    x_min = max(0, cx - half)
    x_max = min(w, cx + half)
    z_min = max(0, cz - half)
    z_max = min(h, cz + half)

    patch = map_heights_array[z_min:z_max, x_min:x_max]
    # Pad if near boundaries to keep consistent size
    padded = np.zeros((view_size, view_size), dtype=np.float32)
    padded[0:patch.shape[0], 0:patch.shape[1]] = patch

    # highlight the unit's position in the local view for better visualization
    unit_local_x = min(half, max(0, int(normalize_x(unit_x) - x_min)))
    unit_local_z = min(half, max(0, int(normalize_z(unit_z) - z_min)))
    if 0 <= unit_local_z < view_size and 0 <= unit_local_x < view_size:
        padded[unit_local_z, unit_local_x] = np.max(padded) + 10.0  # Bright spot for unit position

    # highlight the units sense spots around it for better visualization
    # This covers the 6 close (100 units) and 12 far (200 units) sense spots
    for radius in [100, 200]:
        for angle in np.linspace(0, 2 * np.pi, num=6 if radius == 100 else 12, endpoint=False):
            sense_x = normalize_x(unit_x) + (radius / map_width * STANDARD_MAP_WIDTH if map_width > 0 else radius) * np.cos(angle)
            sense_z = normalize_z(unit_z) + (radius / map_height * STANDARD_MAP_HEIGHT if map_height > 0 else radius) * np.sin(angle)
            sense_local_x = min(half, max(0, int(sense_x - x_min)))
            sense_local_z = min(half, max(0, int(sense_z - z_min)))
            if 0 <= sense_local_z < view_size and 0 <= sense_local_x < view_size:
                padded[sense_local_z, sense_local_x] = np.max(padded) + 5.0  # Bright spots for sense positions

    return padded

def get_state(agent_unit, friendly_units, enemy_units):
    # Encode raw game state into a fixed-size embedding
    with torch.no_grad():
        return agent.encode_state(agent_unit, friendly_units, enemy_units).detach()

def get_state_no_map(agent_unit, friendly_units, enemy_units):
    # Encode state without map embedding (map is static and cached)
    with torch.no_grad():
        return agent.encode_state_no_map(agent_unit, friendly_units, enemy_units).detach()

def compute_action_features(action, unit_x, unit_z, unit_y, unvisited_mass, target_x=None, target_z=None, enemy_range_image=None):
    """
    Compute spatial features for a given action.
    Returns a feature vector that describes the quality of this action.
    
    Features:
    1. Distance reduction to nearest mass spot
    2. Boundary proximity penalty
    3. Terrain steepness penalty
    4. Move magnitude (prefer shorter moves for efficiency)
    5. Is NOOP (1 if NOOP, 0 otherwise)
    6. Height change from current position
    7. Within danger zone of enemy range (if we have enemy range data)
    """
    features = []
    
    # Handle NOOP action
    if action == NOOP_ACTION:
        features.append(0.0)  # No distance reduction
        features.append(0.0)  # No boundary change
        features.append(0.0)  # No move magnitude
        features.append(0.0)  # No terrain change
        features.append(1.0)  # Is NOOP
        features.append(0.0)  # No height change
        features.append(0.0)  # No danger zone penalty
        return features
    
    # If target not passed in, default to current position
    if target_x is None or target_z is None:
        target_x = unit_x
        target_z = unit_z
    
    # Feature 1: Distance reduction to nearest unvisited mass spot
    if unvisited_mass:
        nearest = min(unvisited_mass, key=lambda p: (p[0] - unit_x)**2 + (p[1] - unit_z)**2)
        current_dist = ((nearest[0] - unit_x)**2 + (nearest[1] - unit_z)**2)**0.5
        target_dist = ((nearest[0] - target_x)**2 + (nearest[1] - target_z)**2)**0.5
        dist_reduction = current_dist - target_dist  # Positive = getting closer
        features.append(dist_reduction)
    else:
        features.append(0.0)
    
    # Feature 2: Boundary proximity (negative = closer to edge)
    dist_to_boundary = min(
        target_x,
        target_z,
        STANDARD_MAP_WIDTH - target_x,
        STANDARD_MAP_HEIGHT - target_z
    )
    boundary_feature = -max(0, 100 - dist_to_boundary)  # Negative penalty within 100 units
    features.append(boundary_feature)
    
    # Feature 3: Move magnitude (prefer shorter for efficiency)
    move_magnitude = ((target_x - unit_x) ** 2 + (target_z - unit_z) ** 2) ** 0.5
    magnitude_feature = -move_magnitude / 100.0  # Normalize and make negative
    features.append(magnitude_feature)

    # Feature 4: Terrain steepness penalty
    try:
        # Get height at target position
        target_height = height_at_normalized(target_x, target_z)
        current_height = height_at_normalized(unit_x, unit_z)
        height_diff = abs(target_height - current_height)
        terrain_penalty = -height_diff * 0.05  # Negative = steep
        terrain_penalty = max(terrain_penalty, -5.0)  # Cap penalty to avoid extreme values
        terrain_penalty = min(terrain_penalty, 0.0)  # Only penalize, no positive reward
        terrain_penalty = -(height_diff / max(move_magnitude, 1.0)) * 0.1  # Combine with move magnitude to penalize steep long moves more
    except (IndexError, TypeError):
        terrain_penalty = -100.0
    features.append(terrain_penalty)
    
    # Feature 5: Is NOOP
    features.append(0.0)
    
    # Feature 6: Height change
    try:
        target_height = height_at_normalized(target_x, target_z)
        height_change = target_height - unit_y
        features.append(height_change)
    except (IndexError, TypeError):
        features.append(0.0)

    # Feature 7: Within danger zone of enemy range (if we have enemy range data)
    if enemy_range_image is not None and target_x is not None and target_z is not None:
        map_x = int(target_x)
        map_z = int(target_z)
        if 0 <= map_z < enemy_range_image.shape[0] and 0 <= map_x < enemy_range_image.shape[1]:
            danger_value = enemy_range_image[map_z, map_x]
            danger_feature = -danger_value  # Negative penalty for being in danger zone
            features.append(danger_feature)
        else:
            features.append(0.0)
    else:
        features.append(0.0)
    
    return features

def compute_reward(agent_unit, prev_health):
    """Compute reward with detailed component tracking for TensorBoard visualization."""
    reward = 0
    unit_id = agent_unit['id']
    
    # Dictionary to track individual reward components for visualization
    reward_components = {
        'damage_penalty': 0,
        'mass_reward': 0,
        'distance_improvement': 0,
        'base_distance_reward': 0,
        'inactivity_penalty': 0,
        'no_mass_penalty': 0,
        'boundary_penalty': 0,
        'danger_zone_penalty': 0,
        'cancel_command_penalty': 0
    }
    
    # Punish damage taken
    if agent_unit['health'] < prev_health:
        damage_penalty = prev_health - agent_unit['health']
        reward -= damage_penalty
        reward_components['damage_penalty'] = -damage_penalty
        print(f"Unit {unit_id} damage penalty: -{damage_penalty}")
    
    unit_nx = normalize_x(agent_unit['x'])
    unit_nz = normalize_z(agent_unit['z'])

    # Reward reaching unique mass spots (within range)
    mass_reward = 0
    for spot in mass_spots:
        if spot not in visited_mass_spots:
            dist_to_spot = ((spot[0] - agent_unit['x'])**2 + (spot[1] - agent_unit['z'])**2)**0.5
            if normalize_distance(dist_to_spot) < normalize_distance(200):  # Within 200 units (normalized)
                visited_mass_spots.add(spot)
                visited_mass_spots_norm.add((normalize_x(spot[0]), normalize_z(spot[1])))
                reward += 1000  # Big reward for new mass spot
                mass_reward = 100
                reward_components['mass_reward'] = 100
                print(f"Unit {unit_id} mass reward: +100")
                last_mass_visit[unit_id] = 0  # Reset on visit
                break  # Only reward one per step

    # Distance-based reward for approaching unvisited mass spots
    unvisited_mass = [p for p in map_spots_norm if p not in visited_mass_spots_norm]
    if unvisited_mass:
        nearest_mass = min(unvisited_mass, key=lambda p: (p[0]-unit_nx)**2 + (p[1]-unit_nz)**2)
        current_dist = ((nearest_mass[0]-unit_nx)**2 + (nearest_mass[1]-unit_nz)**2)**0.5
        prev_dist = previous_distances.get(unit_id, current_dist)
        print(f"Unit {unit_id} nearest mass: {nearest_mass}, dist: {current_dist:.2f}, prev_dist: {prev_dist:.2f}")
        
        # Reward for getting closer
        if current_dist < prev_dist:
            dist_improvement = (prev_dist - current_dist) * 0.1
            reward += dist_improvement
            reward_components['distance_improvement'] = dist_improvement
            print(f"Unit {unit_id} distance improvement: +{dist_improvement:.2f}")
        elif current_dist > prev_dist:
            away_penalty = (current_dist - prev_dist) * 0.05
            reward -= away_penalty
            reward_components['distance_improvement'] = -away_penalty
            print(f"Unit {unit_id} moving away penalty: -{away_penalty:.2f}")
        
        # Base reward inversely proportional to distance
        base_dist_reward = max(0, normalize_distance(750) - current_dist) * 0.1
        reward += base_dist_reward
        reward_components['base_distance_reward'] = base_dist_reward
        print(f"Unit {unit_id} base distance reward: +{base_dist_reward:.2f}")
        
        previous_distances[unit_id] = current_dist
    
    # Penalty for not moving (staying in similar position)
    prev_pos = previous_positions.get(unit_id, (agent_unit['x'], agent_unit['z']))
    dist_moved = ((prev_pos[0]-agent_unit['x'])**2 + (prev_pos[1]-agent_unit['z'])**2)**0.5
    if normalize_distance(dist_moved) < normalize_distance(5):  # If moved less than 5 units
        consecutive_inactive[unit_id] = consecutive_inactive.get(unit_id, 0) + 3
        inactivity_penalty = consecutive_inactive[unit_id]
        reward -= inactivity_penalty
        reward_components['inactivity_penalty'] = -inactivity_penalty
        print(f"Unit {unit_id} inactivity penalty: -{inactivity_penalty} (consecutive: {consecutive_inactive[unit_id]})")
    else:
        consecutive_inactive[unit_id] = 0
    
    # Penalty for not visiting mass spots recently
    last_mass_visit[unit_id] = last_mass_visit.get(unit_id, 0) + 1
    no_mass_penalty = last_mass_visit[unit_id] * 0.05  # Slower scaling than inactivity
    reward -= no_mass_penalty
    reward_components['no_mass_penalty'] = -no_mass_penalty
    print(f"Unit {unit_id} no mass visit penalty: -{no_mass_penalty:.2f} (steps: {last_mass_visit[unit_id]})")
    
    # Penalty for being near map boundaries
    boundary_threshold = normalize_distance(200)  # Units within 200 of edge get penalized (normalized)
    dist_to_boundary = min(
        unit_nx,  # Distance to x=0 edge
        unit_nz,  # Distance to z=0 edge
        STANDARD_MAP_WIDTH - unit_nx,  # Distance to x=max edge
        STANDARD_MAP_HEIGHT - unit_nz  # Distance to z=max edge
    )
    
    if dist_to_boundary < boundary_threshold:
        boundary_penalty = (boundary_threshold - dist_to_boundary) * 0.2
        reward -= boundary_penalty
        reward_components['boundary_penalty'] = -boundary_penalty
        print(f"Unit {unit_id} boundary penalty: -{boundary_penalty:.2f} (distance to edge: {dist_to_boundary:.1f})")
    
    # Penalty for being within enemy danger zone
    enemy_range_image = generate_enemy_range_image(eUnits, map_width, map_height, normalized_map_heights.shape if normalized_map_heights is not None else None)
    if enemy_range_image is not None:
        map_x = int(normalize_x(agent_unit['x']))
        map_z = int(normalize_z(agent_unit['z']))
        if 0 <= map_z < enemy_range_image.shape[0] and 0 <= map_x < enemy_range_image.shape[1]:
            if enemy_range_image[map_z, map_x] > 0:
                danger_penalty = -50.0  # Flat penalty for being in danger zone
                reward -= danger_penalty
                reward_components['danger_zone_penalty'] = -danger_penalty
                print(f"Unit {unit_id} danger zone penalty: -{danger_penalty}")
    else:
        enemy_range_image = None

    # Penalty for cancelling an in-flight command (applied on next reward tick)
    cancel_penalty = cancel_command_penalties.pop(unit_id, 0.0)
    if cancel_penalty:
        reward -= cancel_penalty
        reward_components['cancel_command_penalty'] = -cancel_penalty
        print(f"Unit {unit_id} cancel command penalty: -{cancel_penalty}")

    previous_positions[unit_id] = (agent_unit['x'], agent_unit['z'])
    
    # === LOG REWARD COMPONENTS TO TENSORBOARD ===
    # This helps us see which parts of the reward function are driving behavior
    global step_counter
    for component_name, component_value in reward_components.items():
        writer.add_scalar(f'Reward_Components/{component_name}', component_value, step_counter)
    writer.add_scalar('Reward/total_reward', reward, step_counter)
    
    print(f"Unit {unit_id} reward: {reward}")
    return reward

def _init_segment_tracking(unit):
    unit_id = unit['id']
    if unit_id not in segment_stats:
        now = time.time()
        segment_stats[unit_id] = {
            'start_time': now,
            'last_mass_time': now,
            'distance': 0.0,
            'height_change': 0.0,
            'damage_taken': 0.0,
            'steps': 0
        }
        segment_buffers[unit_id] = deque(maxlen=MAX_SEGMENT_STEPS)
        last_mass_visit[unit_id] = now

def _terrain_adjusted_distance(prev_pos, curr_pos, prev_y, curr_y):
    raw_dist = ((prev_pos[0] - curr_pos[0]) ** 2 + (prev_pos[1] - curr_pos[1]) ** 2) ** 0.5
    raw_dist = normalize_distance(raw_dist)
    height_delta = abs(normalize_y(curr_y) - normalize_y(prev_y))
    return raw_dist + height_delta * HEIGHT_DISTANCE_FACTOR

def compute_move_potential(prev_pos, curr_pos, prev_y, curr_y, unvisited_mass):
    """
    Potential-based reward for individual moves.
    Encourages moving toward nearest objective with reachable height changes.
    """
    unit_nx = normalize_x(prev_pos[0])
    unit_nz = normalize_z(prev_pos[1])
    curr_nx = normalize_x(curr_pos[0])
    curr_nz = normalize_z(curr_pos[1])

    move_dx = curr_nx - unit_nx
    move_dz = curr_nz - unit_nz
    move_dist = (move_dx ** 2 + move_dz ** 2) ** 0.5

    if not unvisited_mass:
        return 0.0, {'distance': 0.0, 'direction': 0.0, 'height_jump': 0.0}

    nearest = min(unvisited_mass, key=lambda p: (p[0] - unit_nx) ** 2 + (p[1] - unit_nz) ** 2)
    prev_dist = ((nearest[0] - unit_nx) ** 2 + (nearest[1] - unit_nz) ** 2) ** 0.5
    curr_dist = ((nearest[0] - curr_nx) ** 2 + (nearest[1] - curr_nz) ** 2) ** 0.5

    distance_reduction = prev_dist - curr_dist
    distance_reward = distance_reduction * POTENTIAL_DISTANCE_SCALE

    # Direction alignment reward
    dir_dx = nearest[0] - unit_nx
    dir_dz = nearest[1] - unit_nz
    dir_mag = (dir_dx ** 2 + dir_dz ** 2) ** 0.5
    if move_dist > 1e-6 and dir_mag > 1e-6:
        alignment = (move_dx * dir_dx + move_dz * dir_dz) / ((move_dist * dir_mag) + 1e-6)
    else:
        alignment = 0.0
    direction_reward = alignment * move_dist * POTENTIAL_DIRECTION_SCALE

    # Height jump penalty (reachability)
    height_delta = abs(normalize_y(curr_y) - normalize_y(prev_y))
    height_jump_penalty = -max(0.0, height_delta - HEIGHT_JUMP_TOLERANCE) * HEIGHT_JUMP_PENALTY_SCALE

    total = distance_reward + direction_reward + height_jump_penalty
    components = {
        'distance': distance_reward,
        'direction': direction_reward,
        'height_jump': height_jump_penalty
    }
    return total, components

def _check_mass_reached(unit):
    unit_id = unit['id']
    for spot in mass_spots:
        if spot not in visited_mass_spots:
            dist_to_spot = ((spot[0] - unit['x']) ** 2 + (spot[1] - unit['z']) ** 2) ** 0.5
            if normalize_distance(dist_to_spot) < normalize_distance(MASS_REACH_RADIUS):
                visited_mass_spots.add(spot)
                visited_mass_spots_norm.add((normalize_x(spot[0]), normalize_z(spot[1])))
                now = time.time()
                last_mass_visit[unit_id] = now
                if unit_id in segment_stats:
                    segment_stats[unit_id]['last_mass_time'] = now
                return True, spot
    return False, None

def _compute_segment_reward(unit_id, success, now):
    stats = segment_stats.get(unit_id, None)
    if not stats:
        return 0.0
    time_taken = max(0.01, now - stats['last_mass_time'])
    distance = stats['distance']
    damage_taken = stats['damage_taken']

    time_penalty = time_taken * SEGMENT_TIME_PENALTY
    distance_penalty = distance * SEGMENT_DISTANCE_PENALTY
    damage_penalty = damage_taken * SEGMENT_DAMAGE_PENALTY

    if success:
        return SEGMENT_BASE_REWARD - time_penalty - distance_penalty - damage_penalty
    return -FAILURE_BASE_PENALTY - time_penalty - distance_penalty - damage_penalty

def _finalize_segment_training(unit_id, success, reason):
    buffer = segment_buffers.get(unit_id, [])
    if not buffer:
        return

    now = time.time()
    segment_reward = _compute_segment_reward(unit_id, success, now)
    per_step_bonus = segment_reward / max(1, len(buffer))

    for i, transition in enumerate(list(buffer)):
        state_full = reconstruct_state_with_map(transition['state'])
        next_state_full = reconstruct_state_with_map(transition['next_state'])
        total_reward = transition['potential_reward'] + per_step_bonus
        done = (i == len(buffer) - 1)
        train_agent(
            state_full,
            transition['action'],
            total_reward,
            next_state_full,
            done,
            transition['unit_x'], transition['unit_z'], transition['unit_y'],
            transition['next_unit_x'], transition['next_unit_z'], transition['next_unit_y'],
            transition['target_x'], transition['target_z'],
            unit_id
        )

    # Log segment summary
    global step_counter
    writer.add_scalar('Segment/segment_reward', segment_reward, step_counter)
    writer.add_scalar('Segment/segment_steps', len(buffer), step_counter)
    writer.add_scalar('Segment/success', 1.0 if success else 0.0, step_counter)
    writer.add_text('Segment/reason', reason, step_counter)

    # Reset buffer and stats for next segment
    segment_buffers[unit_id] = deque(maxlen=MAX_SEGMENT_STEPS)
    if unit_id in segment_stats:
        segment_stats[unit_id]['distance'] = 0.0
        segment_stats[unit_id]['height_change'] = 0.0
        segment_stats[unit_id]['damage_taken'] = 0.0
        segment_stats[unit_id]['steps'] = 0
        segment_stats[unit_id]['last_mass_time'] = now

def _finalize_all_units(success, reason):
    for uid in list(segment_buffers.keys()):
        _finalize_segment_training(uid, success, reason)

def get_action(state, unit_x, unit_z, unit_y, unit_id):
    """
    Select action using feature-based potential field.
    1. Get feature weights from neural network
    2. Compute features for each possible action
    3. Score = dot product of features and weights
    4. Select action with highest score
    
    Now with TensorBoard logging to visualize:
    - What feature weights the network learned
    - What scores each action gets
    - Which features contribute most to the chosen action
    """
    with torch.no_grad():
        # === LSTM MEMORY ===
        # Retrieve the unit's hidden state so decisions can depend on the past
        hidden = lstm_hidden_states.get(unit_id, init_lstm_hidden())
        previous_lstm_hidden_states[unit_id] = (hidden[0].detach(), hidden[1].detach())

        # LSTM expects input shape: (batch, seq_len, input_size)
        input_seq = state.unsqueeze(0).unsqueeze(0)
        feature_weights, new_hidden = agent(input_seq, hidden)
        feature_weights = feature_weights.squeeze(0)  # (num_features,)

        # Store updated hidden state for next time step
        lstm_hidden_states[unit_id] = (new_hidden[0].detach(), new_hidden[1].detach())
        
        # === LOG LEARNED FEATURE WEIGHTS ===
        # These weights tell us what the agent has learned to value
        # Higher positive weights = agent thinks this feature is important for good outcomes
        global step_counter
        feature_names = ['distance_reduction', 'boundary_proximity', 'move_magnitude',
                'terrain_steepness', 'is_noop', 'height_change', 'danger_zone']
        
        # Print feature weights for debugging
        print(f"\nFeature weights for unit {unit_id}:")
        for i, (name, weight) in enumerate(zip(feature_names, feature_weights)):
            print(f"  {name}: {weight.item():.4f}")
            writer.add_scalar(f'Feature_Weights/{name}', weight.item(), step_counter)
        
        # Get unvisited mass spots for feature computation (normalized)
        unvisited_mass = [p for p in map_spots_norm if p not in visited_mass_spots_norm]

        unit_nx = normalize_x(unit_x)
        unit_nz = normalize_z(unit_z)
        unit_ny = normalize_y(unit_y)
        
        # === CONTINUOUS TARGET SELECTION ===
        # We sample candidate targets across the entire map to allow movement
        # to ANY point. This gives flexibility without exploding computation.
        candidates = []
        
        # Always include unvisited mass spots (important goals)
        for spot in unvisited_mass:
            candidates.append((spot[0], spot[1]))

        # Sample targets across the entire standardized map (Due to fixed size, 10x10 grid around unit)
        # for _ in range(NUM_RANDOM_TARGETS):
        #     rx = random.uniform(0, STANDARD_MAP_WIDTH)
        #     rz = random.uniform(0, STANDARD_MAP_HEIGHT)
        #     candidates.append((rx, rz))

        for dx in np.linspace(-200, 200, num=10):  # 10 steps in x direction
            for dz in np.linspace(-200, 200, num=10):  # 10 steps in z direction
                tx = unit_nx + dx
                tz = unit_nz + dz
                # Clamp to map boundaries
                tx = max(0, min(STANDARD_MAP_WIDTH, tx))
                tz = max(0, min(STANDARD_MAP_HEIGHT, tz))
                candidates.append((tx, tz))
            

        # Add current position as NOOP candidate
        candidates.append((unit_nx, unit_nz))
        
        # Compute score for each candidate target
        action_scores = []
        best_score = -float('inf')
        best_target = (unit_nx, unit_nz)
        chosen_action_features = [0.0] * 7  # Default to zero features
        
        # Debug: track NOOP vs MOVE scores
        noop_score = None
        move_scores = []
        
        for (tx, tz) in candidates:

            # print out all candidate targets locations to figure out how many are out of bounds and if our clamping is working
            # denorm_tx = denormalize_x(tx)
            # denorm_tz = denormalize_z(tz)
            # print(f"Candidate target: normalized ({tx:.2f}, {tz:.2f}), denormalized ({denorm_tx:.1f}, {denorm_tz:.1f})")

            try:
                # If target is current position, treat as NOOP for scoring
                if abs(tx - unit_nx) < 1e-3 and abs(tz - unit_nz) < 1e-3:
                    features = compute_action_features(NOOP_ACTION, unit_nx, unit_nz, unit_ny, unvisited_mass)
                else:
                    features = compute_action_features("MOVE", unit_nx, unit_nz, unit_ny, unvisited_mass, tx, tz)
                
                if features is None:
                    features = [0.0] * 7
                    
                features_tensor = torch.tensor(features, dtype=torch.float32)
                
                # Score = weighted sum of features
                score = torch.dot(feature_weights, features_tensor).item()
                action_scores.append(score)
                
                # Track NOOP vs MOVE scores for debugging
                if abs(tx - unit_nx) < 1e-3 and abs(tz - unit_nz) < 1e-3:
                    noop_score = score
                else:
                    move_scores.append(score)
                
                if score > best_score:
                    best_score = score
                    best_target = (tx, tz)
                    chosen_action_features = features
            except Exception as e:
                print(f"Error computing action features: {e}")
                continue
        
        # Debug logging
        if move_scores:
            print(f"Unit {unit_id}: NOOP score={noop_score:.3f}, MOVE scores: min={min(move_scores):.3f}, max={max(move_scores):.3f}, mean={np.mean(move_scores):.3f}")
        
        # Determine if we should NOOP
        is_noop = (abs(best_target[0] - unit_nx) < 1e-3 and abs(best_target[1] - unit_nz) < 1e-3)
        best_action = NOOP_ACTION if is_noop else "MOVE"
        if is_noop:
            noop_features = compute_action_features(NOOP_ACTION, unit_nx, unit_nz, unit_ny, unvisited_mass)
            if noop_features is not None:
                chosen_action_features = noop_features
        
        # === LOG ACTION SELECTION DETAILS ===
        # Log statistics about the action scores distribution
        writer.add_scalar('Action_Selection/best_score', best_score, step_counter)
        writer.add_scalar('Action_Selection/mean_score', np.mean(action_scores), step_counter)
        writer.add_scalar('Action_Selection/std_score', np.std(action_scores), step_counter)
        writer.add_scalar('Action_Selection/chosen_action', 1 if best_action == "MOVE" else 0, step_counter)
        writer.add_scalar('Action_Selection/is_noop', 1.0 if best_action == NOOP_ACTION else 0.0, step_counter)
        
        # Log the features of the chosen action to see what made it attractive
        for i, (name, feature_val) in enumerate(zip(feature_names, chosen_action_features)):
            writer.add_scalar(f'Chosen_Action_Features/{name}', feature_val, step_counter)
        
        # Log action score histogram every 100 steps to see the distribution
        if step_counter % 100 == 0:
            writer.add_histogram('Action_Scores/distribution', np.array(action_scores), step_counter)
        
        return best_action, best_target, best_score

def train_agent(state, action, reward, next_state, done, unit_x, unit_z, unit_y, next_unit_x, next_unit_z, next_unit_y, target_x=None, target_z=None, unit_id=None):
    """
    Train the feature weight network.
    The agent learns which features matter most for maximizing reward.
    
    Now with TensorBoard logging to visualize:
    - Training loss (how well predictions match targets)
    - Q-values (agent's estimate of action quality)
    - TD error (difference between prediction and target)
    - Gradient norms (to detect training instability)
    """
    # Get current feature weights using the LSTM hidden state from the previous step
    if unit_id is not None:
        hidden = previous_lstm_hidden_states.get(unit_id, init_lstm_hidden())
    else:
        hidden = init_lstm_hidden()
    input_seq = state.unsqueeze(0).unsqueeze(0)
    current_weights, _ = agent(input_seq, hidden)
    current_weights = current_weights.squeeze(0)
    
    # Compute features for the action taken
    unvisited_mass = [p for p in map_spots_norm if p not in visited_mass_spots_norm]
    unit_nx = normalize_x(unit_x)
    unit_nz = normalize_z(unit_z)
    unit_ny = normalize_y(unit_y)
    target_nx = normalize_x(target_x) if target_x is not None else unit_nx
    target_nz = normalize_z(target_z) if target_z is not None else unit_nz

    action_features = compute_action_features(action, unit_nx, unit_nz, unit_ny, unvisited_mass, target_nx, target_nz)
    features_tensor = torch.tensor(action_features, dtype=torch.float32)
    
    # Current Q-value = dot(weights, features)
    current_q = torch.dot(current_weights, features_tensor)
    
    # Compute next state's best Q-value
    if not done:
        with torch.no_grad():
            # Use the most recent hidden state for next-state evaluation
            next_hidden = lstm_hidden_states.get(unit_id, init_lstm_hidden())
            next_input_seq = next_state.unsqueeze(0).unsqueeze(0)
            next_weights, _ = agent(next_input_seq, next_hidden)
            next_weights = next_weights.squeeze(0)
            next_unvisited = [p for p in map_spots_norm if p not in visited_mass_spots_norm]
            
            # Find best Q-value in next state by sampling candidate targets
            max_next_q = -float('inf')
            candidates = []
            for spot in next_unvisited:
                candidates.append((spot[0], spot[1]))
            # for _ in range(NUM_RANDOM_TARGETS):
            #     tx = random.uniform(0, STANDARD_MAP_WIDTH)
            #     tz = random.uniform(0, STANDARD_MAP_HEIGHT)
            #     candidates.append((tx, tz))
            for dx in np.linspace(-200, 200, num=10):  # 10 steps in x direction
                for dz in np.linspace(-200, 200, num=10):  # 10 steps in z direction
                    tx = unit_nx + dx
                    tz = unit_nz + dz
                    # Clamp to map boundaries
                    tx = max(0, min(STANDARD_MAP_WIDTH, tx))
                    tz = max(0, min(STANDARD_MAP_HEIGHT, tz))
                    candidates.append((tx, tz))
            candidates.append((normalize_x(next_unit_x), normalize_z(next_unit_z)))

            for (tx, tz) in candidates:
                next_nx = normalize_x(next_unit_x)
                next_nz = normalize_z(next_unit_z)
                next_ny = normalize_y(next_unit_y)
                next_features = compute_action_features("MOVE", next_nx, next_nz, next_ny, next_unvisited, tx, tz)
                next_features_tensor = torch.tensor(next_features, dtype=torch.float32)
                next_q = torch.dot(next_weights, next_features_tensor)
                max_next_q = max(max_next_q, next_q.item())
    else:
        max_next_q = 0
    
    # TD target (Temporal Difference target)
    # This is what we think the Q-value SHOULD be based on the reward we got
    target = reward + 0.99 * max_next_q
    target_tensor = torch.tensor(target, dtype=torch.float32)
    
    # === LOG TRAINING METRICS ===
    global step_counter
    td_error = abs(target - current_q.item())  # How wrong was our prediction?
    
    writer.add_scalar('Training/current_q_value', current_q.item(), step_counter)
    writer.add_scalar('Training/target_q_value', target, step_counter)
    writer.add_scalar('Training/max_next_q', max_next_q, step_counter)
    writer.add_scalar('Training/td_error', td_error, step_counter)
    
    # Loss and backprop
    loss = criterion(current_q, target_tensor)
    writer.add_scalar('Training/loss', loss.item(), step_counter)
    
    optimizer.zero_grad()
    loss.backward()
    
    # Clip gradients to prevent exploding gradients that cause NaN
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
    
    # Log gradient norms to detect vanishing/exploding gradients
    total_grad_norm = 0.0
    for param in agent.parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.norm().item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    writer.add_scalar('Training/gradient_norm', total_grad_norm, step_counter)
    
    optimizer.step()

def save_agent():
    torch.save(agent.state_dict(), 'agent_weights_feature_based.pth')
    print("Feature-based agent weights saved.")

def _reset_agent_parameters(model: nn.Module):
    """Reset model parameters safely when NaNs are detected in loaded weights."""
    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    # Re-apply CNN-specific initialization
    if hasattr(model, "_initialize_cnn_weights"):
        model._initialize_cnn_weights()

def load_agent():
    try:
        agent.load_state_dict(torch.load('agent_weights_feature_based.pth'))
        print("Feature-based agent weights loaded.")
        # Detect NaNs in loaded weights
        has_nan = any(torch.isnan(p).any().item() for p in agent.parameters())
        if has_nan:
            print("Loaded weights contain NaN. Reinitializing model weights.")
            _reset_agent_parameters(agent)
    except FileNotFoundError:
        print("No saved feature-based weights found, starting fresh.")
    except RuntimeError as e:
        print("Saved weights are incompatible with the new LSTM architecture.")
        print(f"Details: {e}")
        print("Starting with fresh weights.")
        _reset_agent_parameters(agent)

# Load weights at start
load_agent()

def format_action(action, unit_id, unit_x, unit_z, unit_y, target_x=None, target_z=None):    # Return None for NOOP (no command to send)
    if action == NOOP_ACTION or target_x is None or target_z is None:
        return None
    distance = ((target_x - unit_x) ** 2 + (target_z - unit_z) ** 2) ** 0.5
    command = "C: MU Q" if distance > 50 else "C: MU I"  # Use Q for larger moves
    return f"{command} {unit_id} {target_x} {target_z} {unit_y}\n"

# All the stuff inside your window.
layout = [
    [sg.Multiline(size=(60, 10), key='-LOG-', autoscroll=True, disabled=True)],
    [sg.InputText()],
    [sg.Button('Ok'), sg.Button('Cancel')]
]

# Create the Window
window = sg.Window("Socket Reader", layout, finalize=True)

def receive_messages(conn):
    global units, eUnits, eKUnits, visited_mass_spots, step_counter
    
    while True:
        try:
            data = conn.recv(1024)
            if not data:
                break
            message = data.decode('utf-8')
            print(f"[{addr}] {message}")

            # Parse units directly from message
            if "FRIENDLY_UNITS" in message:
                units = parse_units(message, "FRIENDLY_UNITS")
            if "ENEMY_UNITS" in message:
                eUnits = parse_units(message, "ENEMY_UNITS")
            if "KNOWN_ENEMY_UNITS" in message:
                eKUnits = parse_units(message, "KNOWN_ENEMY_UNITS")
            
            # Use the parsed lists
            friendly_units = units
            enemy_units = eUnits
            print(f"Parsed {len(friendly_units)} friendly units, {len(enemy_units)} enemy units")
            if friendly_units:
                print(f"Sample unit: {friendly_units[0]}")
            
            for unit in friendly_units:
                state = get_state(unit, friendly_units, enemy_units)
                state_no_map = get_state_no_map(unit, friendly_units, enemy_units)
                
                if "TURN" in message:
                    try:
                        # Increment step counter for TensorBoard logging
                        step_counter += 1
                        now = time.time()
                        _init_segment_tracking(unit)

                        prev_health = previous_healths.get(unit['id'], unit['health'])
                        prev_state = previous_states_no_map.get(unit['id'], None)
                        prev_action = previous_actions.get(unit['id'], None)
                        prev_pos = previous_positions.get(unit['id'], (unit['x'], unit['z']))
                        prev_y = previous_y_positions.get(unit['id'], unit['y'])

                        # Compute and store potential reward for the previous move
                        if prev_state is not None and prev_action is not None:
                            unvisited_mass = [p for p in map_spots_norm if p not in visited_mass_spots_norm]
                            potential_reward, components = compute_move_potential(
                                prev_pos,
                                (unit['x'], unit['z']),
                                prev_y,
                                unit['y'],
                                unvisited_mass
                            )

                            cancel_penalty = cancel_command_penalties.pop(unit['id'], 0.0)
                            if cancel_penalty:
                                potential_reward -= cancel_penalty
                                writer.add_scalar('Move_Potential/cancel_command_penalty', -cancel_penalty, step_counter)

                            segment_stats[unit['id']]['distance'] += _terrain_adjusted_distance(
                                prev_pos, (unit['x'], unit['z']), prev_y, unit['y']
                            )
                            segment_stats[unit['id']]['height_change'] += abs(normalize_y(unit['y']) - normalize_y(prev_y))
                            if unit['health'] < prev_health:
                                segment_stats[unit['id']]['damage_taken'] += (prev_health - unit['health'])
                            segment_stats[unit['id']]['steps'] += 1

                            segment_buffers[unit['id']].append({
                                'state': prev_state,
                                'action': prev_action,
                                'next_state': state_no_map,
                                'unit_x': prev_pos[0],
                                'unit_z': prev_pos[1],
                                'unit_y': prev_y,
                                'next_unit_x': unit['x'],
                                'next_unit_z': unit['z'],
                                'next_unit_y': unit['y'],
                                'target_x': previous_targets.get(unit['id'], (unit['x'], unit['z']))[0],
                                'target_z': previous_targets.get(unit['id'], (unit['x'], unit['z']))[1],
                                'potential_reward': potential_reward
                            })

                            writer.add_scalar('Move_Potential/total', potential_reward, step_counter)
                            writer.add_scalar('Move_Potential/distance', components['distance'], step_counter)
                            writer.add_scalar('Move_Potential/direction', components['direction'], step_counter)
                            writer.add_scalar('Move_Potential/height_jump', components['height_jump'], step_counter)
                        elif unit['id'] not in previous_actions:
                            print(f"[INFO] Skipping move potential for unit {unit['id']} (no previous action).")

                        # Check for mass spot reached and train on segment
                        reached_mass, _ = _check_mass_reached(unit)
                        if reached_mass:
                            _finalize_segment_training(unit['id'], True, "mass_reached")
                            print(f"Unit {unit['id']} reached a mass spot. Segment trained.")

                        # Epoch end conditions
                        if mass_spots and len(visited_mass_spots) == len(mass_spots):
                            _finalize_all_units(True, "all_mass_reached")
                            visited_mass_spots.clear()
                            visited_mass_spots_norm.clear()
                            print("All mass spots reached. Epoch ended and reset.")
                        else:
                            last_time = segment_stats[unit['id']]['last_mass_time']
                            if now - last_time >= EPISODE_TIMEOUT_SECONDS:
                                _finalize_segment_training(unit['id'], False, "timeout")
                                print(f"Unit {unit['id']} timed out without reaching a mass spot. Segment punished.")

                        # Print out how many mass spots have been visited
                        print(f"Visited mass spots: {len(visited_mass_spots)}/{len(mass_spots)}")

                        # === LOG GAME STATE METRICS ===
                        writer.add_scalar('Game_State/visited_mass_spots', len(visited_mass_spots), step_counter)
                        writer.add_scalar('Game_State/total_mass_spots', len(mass_spots), step_counter)
                        writer.add_scalar('Game_State/mass_completion_ratio',
                                        len(visited_mass_spots) / max(1, len(mass_spots)), step_counter)
                        writer.add_scalar('Game_State/unit_health', unit['health'], step_counter)
                        writer.add_scalar('Game_State/friendly_unit_count', len(friendly_units), step_counter)
                        writer.add_scalar('Game_State/enemy_unit_count', len(enemy_units), step_counter)
                    except Exception as reward_err:
                        print(f"[ERROR during reward/training for unit {unit['id']}]")
                        print(f"  Exception: {reward_err}")
                        traceback.print_exc()
                
                try:
                    previous_healths[unit['id']] = unit['health']
                    previous_states[unit['id']] = state
                    previous_states_no_map[unit['id']] = state_no_map
                    previous_y_positions[unit['id']] = unit['y']
                    previous_positions[unit['id']] = (unit['x'], unit['z'])
                    
                    # Get action from agent (continuous target selection + LSTM memory)
                    action, best_target, best_score = get_action(state, unit['x'], unit['z'], unit['y'], unit['id'])
                    
                    # Print out the chosen action and target in both normalized and world coordinates for debugging
                    denorm_tx = denormalize_x(best_target[0])
                    denorm_tz = denormalize_z(best_target[1])
                    print(f"Unit {unit['id']} chose action {action} -> target normalized ({best_target[0]:.2f}, {best_target[1]:.2f}), denormalized ({denorm_tx:.1f}, {denorm_tz:.1f}) (score {best_score:.2f})")
                    
                    best_target_world = (denorm_tx, denorm_tz)
                    # print(f"Unit {unit['id']} chose action {action} -> target {best_target_world} (score {best_score:.2f})")
                    last_score = previous_action_scores.get(unit['id'], None)
                except Exception as action_err:
                    print(f"[ERROR during action selection for unit {unit['id']}]")
                    print(f"  Exception: {action_err}")
                    traceback.print_exc()
                    continue
                
                try:
                    # === COMMAND PERSISTENCE LOGIC ===
                    # Always allow command changes, but penalize cancelling in-flight commands.
                    last_target = previous_targets.get(unit['id'], None)
                    previous_command_steps[unit['id']] = previous_command_steps.get(unit['id'], 0) + 1

                    action_command = format_action(action, unit['id'], unit['x'], unit['z'], unit['y'], best_target_world[0], best_target_world[1])
                    if action_command is not None:
                        # If we cancel an in-flight command, apply a small reward penalty next tick
                        if last_target is not None:
                            dist_to_last = ((unit['x'] - last_target[0]) ** 2 + (unit['z'] - last_target[1]) ** 2) ** 0.5
                            if dist_to_last > COMMAND_DISTANCE_EPS and previous_command_steps[unit['id']] < MIN_STEPS_BETWEEN_COMMANDS:
                                if best_target != last_target:
                                    cancel_command_penalties[unit['id']] = CANCEL_COMMAND_PENALTY
                                    print(f"Unit {unit['id']} cancelling in-flight command (penalty {CANCEL_COMMAND_PENALTY})")

                        conn.sendall(action_command.encode('utf-8'))
                        print(f"{unit['id']} has been sent action: {action_command.strip()}")
                        previous_actions[unit['id']] = action
                        previous_targets[unit['id']] = best_target_world
                        previous_action_scores[unit['id']] = best_score
                        previous_command_steps[unit['id']] = 0
                    else:
                        print(f"{unit['id']} executing NOOP (no command sent)")
                except Exception as send_err:
                    print(f"[ERROR during command sending for unit {unit['id']}]")
                    print(f"  Exception: {send_err}")
                    traceback.print_exc()
                
                # === LOG WHAT THE AGENT "SEES" ===
                # Every couple seconds (approx by steps), log local view and enemy range image
                try:
                    if step_counter % 20 == 0:
                        local_view = get_local_view_image(unit['x'], unit['z'], normalized_map_heights, view_size=256)
                        if local_view is not None:
                            local_view_norm = normalize_image(local_view)
                            writer.add_image('Agent_View/local_heights', local_view_norm, step_counter, dataformats='HW')
                        enemy_img = generate_enemy_range_image(enemy_units, map_width, map_height, normalized_map_heights.shape if normalized_map_heights is not None else None)
                        if enemy_img is not None:
                            writer.add_image('Enemy_Ranges/map', enemy_img, step_counter, dataformats='HW')
                except Exception as img_err:
                    print(f"[ERROR during image logging for unit {unit['id']}]")
                    print(f"  Exception: {img_err}")
                    traceback.print_exc()
            
            window.write_event_value('-SOCKET-', message)
        except Exception as e:
            # Print full traceback to pinpoint the actual error location
            print(f"\n!!! ERROR in receive_messages !!!")
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception message: {e}")
            print("Full traceback:")
            traceback.print_exc()
            print("!!!\n")
            break
    print(f"[DISCONNECTED] {addr} disconnected.")
    conn.close()

# Create a socket object using the 'with' statement for automatic resource management
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    try:
        s.bind((HOST, PORT))  # Bind the socket to the address and port
        s.listen()  # Enable the server to accept connections
        print(f"Server listening on {HOST}:{PORT}...")

        # Accept an incoming connection. 'conn' is a new socket object for the client, 'addr' is the client's address
        conn, addr = s.accept()

        # "F:/BAR Beyond All Reason/Beyond-All-Reason/data"
        # Load map heights and mass points after connection (generated by game)
        
        map_heights = pd.read_csv('F:/BAR Beyond All Reason/Beyond-All-Reason/data/mapHeightInfo.txt', header=None).values
        map_size = map_heights.shape[0]

        # Load map dimensions from mapInfo
        # map_width = 0
        # map_height = 0
        try:
            with open('F:/BAR Beyond All Reason/Beyond-All-Reason/data/mapInfo.txt', 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('Map Width:'):
                        map_width = int(line.split(':')[1]) * 8
                    elif line.startswith('Map Height:'):
                        map_height = int(line.split(':')[1]) * 8
        except FileNotFoundError:
            print("mapInfo.txt not found, using height map dimensions")
            map_width = map_size
            map_height = map_size
        
        print(f"Map dimensions: {map_width} x {map_height}")

        # Build standardized height map
        build_normalized_height_map()

        # === LOG ENTIRE MAP AS IMAGE ===
        # This shows the full height map the agent is operating on
        if normalized_map_heights is not None:
            full_map_norm = normalize_image(normalized_map_heights)
            writer.add_image('Map/height_map', full_map_norm, 0, dataformats='HW')

        mass_spots = []
        try:
            with open('F:/BAR Beyond All Reason/Beyond-All-Reason/data/massInfo.txt', 'r') as f:
                for line in f:
                    if line.strip():
                        x, y, z = map(float, line.strip().split(','))
                        mass_spots.append((x, z))  # Use x,z for positions
        except FileNotFoundError:
            print("mass_points.csv not found, using empty mass spots.")
            mass_spots = []

        # Normalize mass spots to standardized map space
        map_spots_norm = [(normalize_x(x), normalize_z(z)) for x, z in mass_spots]
        visited_mass_spots_norm = set()

        # sock.sendall(message.encode('utf-8'))
        server_thread = threading.Thread(target=receive_messages, args=(conn,), daemon=True)
        server_thread.start()
        print(f"[NEW CONNECTION] {addr} connected.")
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


    while True:
        event, values = window.read()
        # if user closes window or clicks cancel
        if event == sg.WIN_CLOSED or event == 'Cancel':
            break

        if event == 'Ok':
            user_input = values[0] + "\n"
            if user_input.strip() != "":
                pass
                # # Run send in a separate thread to avoid blocking the GUI
                # send_thread = threading.Thread(target=lambda: conn.sendall(user_input.encode('utf-8')), daemon=True)
                # send_thread.start()
                # print(f"User Input: {user_input}")

        # Handle incoming socket data
        if event == '-SOCKET-':
            data = values[event]
            window['-LOG-'].update(str(units) + str(eUnits) + str(eKUnits), append=False)

    window.close()
    save_agent()
    
    # === CLOSE TENSORBOARD WRITER ===
    # Flush all pending logs and close the writer
    print("Closing TensorBoard writer...")
    writer.close()
    print(f"TensorBoard logs saved to: runs/{run_name}")
    print("View results with: tensorboard --logdir=runs")




