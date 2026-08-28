import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add parent directory to path to import main modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
import map_utils
import runtime_state as state
from Rewards import MoveJudger

# === ACTION SETTINGS ===
NOOP_ACTION = "NOOP"
NUM_ACTION_FEATURES = 4

# === STANDARDIZED MAP SETTINGS ===
STANDARD_MAP_WIDTH = 1024
STANDARD_MAP_HEIGHT = 1024
STANDARD_MAP_Y = 256

# === ENCODER SETTINGS (match Socket ML Feature-Based.py) ===
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

LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 1

# === TRAINING SETTINGS ===
LEARNING_RATE = 1e-3
NEGATIVE_SAMPLES = 8

map_heights = {}
normalized_map_heights = None
terrain_cost_map = None
mass_cost_fields = {}
map_spots = []
map_spots_norm = []
visited_mass_spots_norm = set()
map_width = 0
map_height = 0
map_height_min = 0.0
map_height_max = 1.0
eUnits = []


class RTSAgent(nn.Module):
    def __init__(self, input_size, num_features):
        super().__init__()
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

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            batch_first=True
        )
        self.fc1 = nn.Linear(LSTM_HIDDEN_SIZE, 64)
        self.fc2 = nn.Linear(64, num_features)
        
        # Initialize CNN weights properly to prevent NaN
        self._initialize_cnn_weights()
    
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
        
        for module in self.map_fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x, hidden=None):
        lstm_out, new_hidden = self.lstm(x, hidden)
        last_out = lstm_out[:, -1, :]
        x = torch.relu(self.fc1(last_out))
        feature_weights = self.fc2(x)
        return feature_weights, new_hidden

    def encode_state(self, agent_unit, friendly_units, enemy_units, seed=None):
        device = next(self.parameters()).device

        self_features = torch.tensor([
            agent_unit['x'],
            agent_unit['z'],
            agent_unit['y'],
            agent_unit['health'],
            float(len(friendly_units)),
            float(len(enemy_units))
        ], dtype=torch.float32, device=device)
        self_emb = self.self_encoder(self_features)

        if map_spots:
            nearest_mass = min(map_spots, key=lambda p: (p[0] - agent_unit['x'])**2 + (p[1] - agent_unit['z'])**2)
            dx = nearest_mass[0] - agent_unit['x']
            dz = nearest_mass[1] - agent_unit['z']
            dist = (dx ** 2 + dz ** 2) ** 0.5
        else:
            dx, dz, dist = 0.0, 0.0, 0.0
        mass_features = torch.tensor([dx, dz, dist], dtype=torch.float32, device=device)
        mass_emb = self.mass_encoder(mass_features)

        # Map branch (full map CNN) - normalized coordinates
        if normalized_map_heights is not None:
            # Normalize the height map to mean=0, std=1 for numerical stability
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
            map_tensor = map_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 1024, 1024)
            map_cnn_out = self.map_cnn(map_tensor)
            map_emb = self.map_fc(map_cnn_out.squeeze(0))
            map_emb = torch.nan_to_num(map_emb, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            map_emb = torch.zeros(MAP_EMBED_SIZE, dtype=torch.float32, device=device)

        friendly_vectors = []
        for u in friendly_units:
            if u['id'] == agent_unit['id']:
                continue
            friendly_vectors.append([
                u['x'] - agent_unit['x'],
                u['z'] - agent_unit['z'],
                u['health']
            ])
        if friendly_vectors:
            friendly_tensor = torch.tensor(friendly_vectors, dtype=torch.float32, device=device)
            friendly_embeds = self.friendly_unit_encoder(friendly_tensor)
            friendly_emb = torch.mean(friendly_embeds, dim=0)
        else:
            friendly_emb = torch.zeros(FRIENDLY_EMBED_SIZE, dtype=torch.float32, device=device)

        enemy_vectors = []
        for u in enemy_units:
            enemy_vectors.append([
                u['x'] - agent_unit['x'],
                u['z'] - agent_unit['z'],
                u['health'],
                u.get('range', 0.0)
            ])
        if enemy_vectors:
            enemy_tensor = torch.tensor(enemy_vectors, dtype=torch.float32, device=device)
            enemy_embeds = self.enemy_unit_encoder(enemy_tensor)
            enemy_emb = torch.mean(enemy_embeds, dim=0)
        else:
            enemy_emb = torch.zeros(ENEMY_EMBED_SIZE, dtype=torch.float32, device=device)

        state = torch.cat([self_emb, mass_emb, map_emb, friendly_emb, enemy_emb], dim=0)
        state = torch.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        return state


ENCODER_OUTPUT_SIZE = SELF_EMBED_SIZE + MASS_EMBED_SIZE + MAP_EMBED_SIZE + FRIENDLY_EMBED_SIZE + ENEMY_EMBED_SIZE
agent = RTSAgent(input_size=ENCODER_OUTPUT_SIZE, num_features=NUM_ACTION_FEATURES)
optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)


def load_map_heights(filepath: str):
    global map_heights, map_width, map_height, normalized_map_heights, map_height_min, map_height_max
    map_heights = {}
    map_width = 0
    map_height = 0

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("MapSizeX"):
                try:
                    map_width = int(line.split(',')[1])
                except ValueError:
                    pass
                continue
            if line.startswith("MapSizeZ"):
                try:
                    map_height = int(line.split(',')[1])
                except ValueError:
                    pass
                continue
            if line.startswith("WaterLevel") or line.startswith("x,z,height"):
                continue

            parts = line.split(',')
            if len(parts) >= 3:
                try:
                    x = int(float(parts[0]))
                    z = int(float(parts[1]))
                    h = float(parts[2])
                    map_heights[(x, z)] = h
                except ValueError:
                    continue
    
    # Build standardized height map after loading
    build_normalized_height_map()


def build_normalized_height_map():
    # \"\"\"Build 1024x1024 standardized height map from loaded map_heights.\"\"\"
    global normalized_map_heights, map_height_min, map_height_max
    
    if not map_heights or map_width == 0 or map_height == 0:
        normalized_map_heights = None
        return
    
    # Convert dict to 2D array
    heights_array = np.zeros((map_height, map_width), dtype=np.float32)
    for (x, z), h in map_heights.items():
        if 0 <= z < map_height and 0 <= x < map_width:
            heights_array[z, x] = h

    heights_array = np.nan_to_num(heights_array, nan=0.0, posinf=0.0, neginf=0.0)
    map_height_min = float(np.nanmin(heights_array))
    map_height_max = float(np.nanmax(heights_array))
    if not np.isfinite(map_height_min) or not np.isfinite(map_height_max):
        map_height_min = 0.0
        map_height_max = 1.0
    
    # Resample to standard 1024x1024
    src_h, src_w = heights_array.shape
    x_idx = np.linspace(0, src_w - 1, STANDARD_MAP_WIDTH).astype(int)
    z_idx = np.linspace(0, src_h - 1, STANDARD_MAP_HEIGHT).astype(int)
    normalized = heights_array[np.ix_(z_idx, x_idx)]
    
    # Normalize heights to [0, STANDARD_MAP_Y]
    if map_height_max > map_height_min:
        normalized = (normalized - map_height_min) / (map_height_max - map_height_min) * STANDARD_MAP_Y
    
    normalized_map_heights = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


def load_map_spots(filepath: str):
    global map_spots
    map_spots = []
    with open(filepath, 'r') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            parts = line.strip().split(',')
            if len(parts) >= 4:
                try:
                    x = float(parts[2])
                    z = float(parts[3])
                    map_spots.append((x, z))
                except ValueError:
                    continue


def height_at(x: float, z: float) -> float:
    map_x = int(x / 8) * 8
    map_z = int(z / 8) * 8
    return map_heights.get((map_x, map_z), -1000.0)


def sample_surrounding_heights(unit_x: float, unit_z: float, seed=None) -> List[float]:
    if not map_heights:
        return []

    rnd = random.Random(seed)
    heights = []

    for _ in range(6):
        angle = rnd.uniform(0, 2 * math.pi)
        dx = unit_x + 100 * math.cos(angle)
        dz = unit_z + 100 * math.sin(angle)
        heights.append(height_at(dx, dz))

    for _ in range(12):
        angle = rnd.uniform(0, 2 * math.pi)
        dx = unit_x + 200 * math.cos(angle)
        dz = unit_z + 200 * math.sin(angle)
        heights.append(height_at(dx, dz))

    return heights


def danger_zone_feature(target_x: float, target_z: float, enemy_units: List[Dict]) -> float:
    for enemy in enemy_units:
        rng = enemy.get('range', 0.0)
        if rng <= 0:
            continue
        dist = ((enemy['x'] - target_x) ** 2 + (enemy['z'] - target_z) ** 2) ** 0.5
        if dist <= rng:
            return -1.0
    return 0.0


def compute_action_features(action_type: str, unit_x: float, unit_z: float, unit_y: float,
                            target_x: float, target_z: float, enemy_units: List[Dict]) -> List[float]:
    if action_type == NOOP_ACTION:
        return [0.0, 0.0, 0.0, 0.0]
    
    # Normalize coordinates
    unit_nx = map_utils.normalize_x(unit_x) if map_width > 0 else unit_x
    unit_nz = map_utils.normalize_z(unit_z) if map_height > 0 else unit_z
    unit_ny = map_utils.normalize_y(unit_y)
    target_nx = map_utils.normalize_x(target_x) if map_width > 0 else target_x
    target_nz = map_utils.normalize_z(target_z) if map_height > 0 else target_z
    
    # Get unvisited mass spots
    unvisited_mass = [p for p in map_spots_norm if p not in visited_mass_spots_norm]
    
    # Use MoveJudger to compute features consistent with main training
    features = MoveJudger.compute_action_features(
        action_type,
        unit_nx,
        unit_nz,
        unit_ny,
        unvisited_mass,
        target_nx,
        target_nz,
        enemy_range_image=None
    )
    
    return features


def load_imitation_data(filepath: str) -> Dict:
    if not Path(filepath).exists():
        print(f"File {filepath} not found.")
        return {}

    with open(filepath, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data.get('samples', []))} samples from {filepath}")
    return data


def _safe_int(value: Any, default: int = 0) -> int:
    """Best-effort integer conversion for map metadata values."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _map_signature_from_metadata(meta: Dict[str, Any]) -> Tuple[str, str, int, int]:
    """Build a stable map signature tuple from dataset-level metadata."""
    return (
        str(meta.get('map_heights_file', '') or ''),
        str(meta.get('map_spots_file', '') or ''),
        _safe_int(meta.get('map_width', 0), 0),
        _safe_int(meta.get('map_height', 0), 0),
    )


def _map_signature_from_sample(sample: Dict[str, Any], fallback: Tuple[str, str, int, int]) -> Tuple[str, str, int, int]:
    """Read per-sample map signature when present, otherwise fallback to dataset metadata."""
    raw = sample.get('_map_signature')
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        return (
            str(raw[0] or ''),
            str(raw[1] or ''),
            _safe_int(raw[2], 0),
            _safe_int(raw[3], 0),
        )
    return fallback


def _apply_map_context(map_signature: Tuple[str, str, int, int]):
    """Load map assets and rebuild derived map state for the active training group."""
    global map_spots_norm, visited_mass_spots_norm

    map_heights_file, map_spots_file, _, _ = map_signature

    # Keep loader behavior deterministic: always rebuild from declared files when available.
    if map_heights_file:
        load_map_heights(map_heights_file)
    if map_spots_file:
        load_map_spots(map_spots_file)

    state.map_heights = map_heights
    state.map_width = map_width
    state.map_height = map_height
    state.map_height_min = map_height_min
    state.map_height_max = map_height_max
    state.mass_spots = map_spots

    build_normalized_height_map()
    state.normalized_map_heights = normalized_map_heights

    if state.normalized_map_heights is not None:
        map_utils.build_terrain_cost_map()
        map_utils.build_mass_cost_fields()

    map_spots_norm = [
        (map_utils.normalize_x(x) if map_width > 0 else x,
         map_utils.normalize_z(z) if map_height > 0 else z)
        for x, z in map_spots
    ]
    state.map_spots_norm = map_spots_norm
    # Reset visited state per map group to avoid cross-map contamination.
    visited_mass_spots_norm = set()


def train_imitation(dataset_file: str, epochs: int = 10, shuffle: bool = True):
    data = load_imitation_data(dataset_file)
    samples = data.get('samples', [])
    if not samples:
        print("No samples to train on.")
        return

    meta = data.get('metadata', {})

    default_signature = _map_signature_from_metadata(meta)

    # Group samples by map signature so each batch uses the correct terrain/mass context.
    grouped_samples: Dict[Tuple[str, str, int, int], List[Dict[str, Any]]] = {}
    for sample in samples:
        signature = _map_signature_from_sample(sample, default_signature)
        grouped_samples.setdefault(signature, []).append(sample)

    if not grouped_samples:
        print("No valid grouped samples to train on.")
        return

    print(f"Detected {len(grouped_samples)} map group(s) in dataset.")
    for idx, (signature, group) in enumerate(grouped_samples.items(), start=1):
        print(
            f"  Group {idx}: samples={len(group)}, "
            f"map_heights='{signature[0]}', map_spots='{signature[1]}'"
        )

    print(f"Training for {epochs} epochs...")

    for epoch in range(epochs):
        # Shuffle each map group independently to preserve map-aware batching.
        if shuffle:
            for group in grouped_samples.values():
                random.shuffle(group)

        epoch_loss = 0.0
        sample_count = 0

        for group_idx, (signature, group_samples) in enumerate(grouped_samples.items(), start=1):
            print(f"Epoch {epoch + 1}: loading map context for group {group_idx}/{len(grouped_samples)}")
            _apply_map_context(signature)
            print("Building terrain cost map...")
            print("Building mass point cost fields...")

            for sample in group_samples:
                unit_id = sample['unit_id']
                friendly_units = sample['friendly_units']
                enemy_units = sample['enemy_units']

                agent_unit = next((u for u in friendly_units if u['id'] == unit_id), None)
                if agent_unit is None:
                    continue

                seed = int(sample.get('timestamp', 0) * 1000) ^ unit_id
                state_vec = agent.encode_state(agent_unit, friendly_units, enemy_units, seed=seed)

                action = sample['action']
                action_type = action.get('type', NOOP_ACTION)
                target_x = action.get('x', agent_unit['x'])
                target_z = action.get('z', agent_unit['z'])

                input_seq = state_vec.unsqueeze(0).unsqueeze(0)
                feature_weights, _ = agent(input_seq)
                feature_weights = feature_weights.squeeze(0)

                pos_features = compute_action_features(
                    action_type,
                    agent_unit['x'],
                    agent_unit['z'],
                    agent_unit['y'],
                    target_x,
                    target_z,
                    enemy_units
                )
                pos_score = torch.dot(feature_weights, torch.tensor(pos_features, dtype=torch.float32))

                scores = [pos_score]
                for _ in range(NEGATIVE_SAMPLES):
                    rx = random.uniform(0, map_width) if map_width > 0 else agent_unit['x']
                    rz = random.uniform(0, map_height) if map_height > 0 else agent_unit['z']
                    neg_features = compute_action_features(
                        "MOVE",
                        agent_unit['x'],
                        agent_unit['z'],
                        agent_unit['y'],
                        rx,
                        rz,
                        enemy_units
                    )
                    neg_score = torch.dot(feature_weights, torch.tensor(neg_features, dtype=torch.float32))
                    scores.append(neg_score)

                scores_tensor = torch.stack(scores)
                loss = -torch.log_softmax(scores_tensor, dim=0)[0]

                optimizer.zero_grad()
                loss.backward()
                epoch_loss += loss.item()
                sample_count += 1
        # Clip gradients to prevent exploding gradients that cause NaN
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
        optimizer.step()

        avg_loss = epoch_loss / max(1, sample_count)
        print(f"Epoch {epoch + 1}/{epochs} - Avg Loss: {avg_loss:.4f}")

    print("Imitation training complete!")


def save_agent(filepath='agent_weights_feature_based.pth'):
    has_nan = any(torch.isnan(p).any().item() for p in agent.parameters())
    if has_nan:
        print("Warning: NaN detected in weights. Skipping save to avoid corrupting checkpoint.")
        return
    torch.save(agent.state_dict(), filepath)
    print(f"Agent weights saved to {filepath}")


def load_agent(filepath='agent_weights_feature_based.pth'):
    try:
        agent.load_state_dict(torch.load(filepath))
        print(f"Agent weights loaded from {filepath}")
        has_nan = any(torch.isnan(p).any().item() for p in agent.parameters())
        if has_nan:
            print("Warning: NaN detected in loaded weights. Reinitializing model.")
            for module in agent.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
            if hasattr(agent, "_initialize_cnn_weights"):
                agent._initialize_cnn_weights()
    except FileNotFoundError:
        print(f"No saved weights found at {filepath}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        dataset_file = sys.argv[1]
    else:
        dataset_file = 'replay.json'

    load_agent('agent_weights_feature_based.pth')
    train_imitation(dataset_file, epochs=10)
    save_agent('agent_weights_feature_based.pth')

    print("\nUsage: python offline_train.py [imitation_dataset.json]")
