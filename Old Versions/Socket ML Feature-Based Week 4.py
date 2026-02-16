import socket
import sys
import threading
import FreeSimpleGUI as sg
import numpy as np
import pandas as pd
from collections import defaultdict
import random
import torch
import torch.nn as nn
import torch.optim as optim
# TensorBoard for visualization - helps us understand what the agent is learning
from torch.utils.tensorboard import SummaryWriter
import datetime

HOST = "127.0.0.1"
PORT = 25000

units = []
eUnits = []
eKUnits = []

# Global variables for map data (assigned after connection)
global map_heights, mass_spots, map_width, map_height
map_heights = None
mass_spots = []
map_width = 0
map_height = 0

# Discretize moves: dx, dz from -200 to 200 in steps of 25 (17 values each)
move_steps = list(range(-200, 201, 25))  # -200, -175, ..., 200
num_moves = len(move_steps)
action_size = num_moves * num_moves + 1  # 17*17 = 289, +1 for NOOP (action 289)
NOOP_ACTION = action_size - 1  # Last action is "do nothing"

# Number of action features for the potential field
NUM_ACTION_FEATURES = 6

# Feature-Based RL Agent
class RTSAgent(nn.Module):
    def __init__(self, input_size, num_features):
        """
        Feature-based agent that learns weights for action features.
        Instead of outputting Q-values for each action, it outputs feature weights.
        """
        super(RTSAgent, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_features)  # Output feature weights

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Returns feature weights

agent = RTSAgent(input_size=17, num_features=NUM_ACTION_FEATURES)
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
previous_actions = {}
previous_positions = {}  # Track previous positions for movement penalty
previous_distances = {}  # Track distance to nearest mass spot

visited_mass_spots = set()

consecutive_inactive = {}

last_mass_visit = {}

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

def sample_surrounding_heights(unit_x, unit_z, offsets=[-50, 0, 50]):
    """
    Sample heights in a 3x3 grid around the unit.
    Heights are stored at 8-unit increments, so divide position by 8 to get map index.
    
    Returns a list of 9 height values in order:
    [-50,-50], [-50,0], [-50,50], [0,-50], [0,0], [0,50], [50,-50], [50,0], [50,50]
    """
    heights = []
    
    for dx in offsets:
        for dz in offsets:
            sample_x = unit_x + dx
            sample_z = unit_z + dz
            
            # Convert to map indices (8-unit increments)
            map_x = int(sample_x / 8)
            map_z = int(sample_z / 8)
            
            # Get height with boundary checking
            try:
                if 0 <= map_z < map_heights.shape[0] and 0 <= map_x < map_heights.shape[1]:
                    height = float(map_heights[map_z, map_x])
                else:
                    height = 0.0  # Default for out-of-bounds
            except (IndexError, TypeError):
                height = 0.0
            
            heights.append(height)
    
    return heights

def get_state(agent_unit, friendly_units, enemy_units):
    # Create state vector
    state = [
        agent_unit['x'], agent_unit['z'], agent_unit['y'], agent_unit['health'],
        len(friendly_units), len(enemy_units)
    ]
    # Add nearest mass spot
    nearest_mass = min(mass_spots, key=lambda p: (p[0]-agent_unit['x'])**2 + (p[1]-agent_unit['z'])**2) if mass_spots else (0, 0)
    state.extend([nearest_mass[0], nearest_mass[1]])
    
    # Add surrounding heights (3x3 grid)
    surrounding_heights = sample_surrounding_heights(agent_unit['x'], agent_unit['z'])
    state.extend(surrounding_heights)
    
    return torch.tensor(state, dtype=torch.float32)

def compute_action_features(action, unit_x, unit_z, unit_y, unvisited_mass):
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
    """
    features = []
    
    # Handle NOOP action
    if action == NOOP_ACTION:
        features.append(0.0)  # No distance reduction
        features.append(0.0)  # No boundary change
        features.append(0.0)  # No terrain change
        features.append(1.0)  # NOOP has value for waiting
        features.append(1.0)  # Is NOOP
        features.append(0.0)  # No height change
        return features
    
    # Decode action to movement
    dy_idx = action // num_moves
    dx_idx = action % num_moves
    dx = move_steps[dx_idx]
    dz = move_steps[dy_idx]
    
    target_x = unit_x + dx
    target_z = unit_z + dz
    
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
        map_width - target_x,
        map_height - target_z
    )
    boundary_feature = -max(0, 100 - dist_to_boundary)  # Negative penalty within 100 units
    features.append(boundary_feature)
    
    # Feature 3: Terrain steepness penalty
    try:
        # Get height at target position
        map_x = int(target_x / 8)
        map_z = int(target_z / 8)
        current_map_x = int(unit_x / 8)
        current_map_z = int(unit_z / 8)
        
        if (0 <= map_z < map_heights.shape[0] and 0 <= map_x < map_heights.shape[1] and
            0 <= current_map_z < map_heights.shape[0] and 0 <= current_map_x < map_heights.shape[1]):
            target_height = float(map_heights[map_z, map_x])
            current_height = float(map_heights[current_map_z, current_map_x])
            height_diff = abs(target_height - current_height)
            terrain_penalty = -height_diff  # Negative = steep
        else:
            terrain_penalty = -100.0  # Out of bounds
    except (IndexError, TypeError):
        terrain_penalty = -100.0
    features.append(terrain_penalty)
    
    # Feature 4: Move magnitude (prefer shorter for efficiency)
    move_magnitude = (dx**2 + dz**2)**0.5
    magnitude_feature = -move_magnitude / 100.0  # Normalize and make negative
    features.append(magnitude_feature)
    
    # Feature 5: Is NOOP
    features.append(0.0)
    
    # Feature 6: Height change
    try:
        map_x = int(target_x / 8)
        map_z = int(target_z / 8)
        if 0 <= map_z < map_heights.shape[0] and 0 <= map_x < map_heights.shape[1]:
            target_height = float(map_heights[map_z, map_x])
            height_change = target_height - unit_y
            features.append(height_change)
        else:
            features.append(0.0)
    except (IndexError, TypeError):
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
        'boundary_penalty': 0
    }
    
    # Punish damage taken
    if agent_unit['health'] < prev_health:
        damage_penalty = prev_health - agent_unit['health']
        reward -= damage_penalty
        reward_components['damage_penalty'] = -damage_penalty
        print(f"Unit {unit_id} damage penalty: -{damage_penalty}")
    
    # Reward reaching unique mass spots (within range)
    mass_reward = 0
    for spot in mass_spots:
        if spot not in visited_mass_spots:
            dist_to_spot = ((spot[0] - agent_unit['x'])**2 + (spot[1] - agent_unit['z'])**2)**0.5
            if dist_to_spot < 200:  # Within 200 units
                visited_mass_spots.add(spot)
                reward += 1000  # Big reward for new mass spot
                mass_reward = 100
                reward_components['mass_reward'] = 100
                print(f"Unit {unit_id} mass reward: +100")
                last_mass_visit[unit_id] = 0  # Reset on visit
                break  # Only reward one per step

    # Distance-based reward for approaching unvisited mass spots
    unvisited_mass = [p for p in mass_spots if p not in visited_mass_spots]
    if unvisited_mass:
        nearest_mass = min(unvisited_mass, key=lambda p: (p[0]-agent_unit['x'])**2 + (p[1]-agent_unit['z'])**2)
        current_dist = ((nearest_mass[0]-agent_unit['x'])**2 + (nearest_mass[1]-agent_unit['z'])**2)**0.5
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
        base_dist_reward = max(0, 750 - current_dist) * 0.1
        reward += base_dist_reward
        reward_components['base_distance_reward'] = base_dist_reward
        print(f"Unit {unit_id} base distance reward: +{base_dist_reward:.2f}")
        
        previous_distances[unit_id] = current_dist
    
    # Penalty for not moving (staying in similar position)
    prev_pos = previous_positions.get(unit_id, (agent_unit['x'], agent_unit['z']))
    dist_moved = ((prev_pos[0]-agent_unit['x'])**2 + (prev_pos[1]-agent_unit['z'])**2)**0.5
    if dist_moved < 5:  # If moved less than 5 units
        consecutive_inactive[unit_id] = consecutive_inactive.get(unit_id, 0) + 3
        inactivity_penalty = consecutive_inactive[unit_id]
        reward -= inactivity_penalty
        reward_components['inactivity_penalty'] = -inactivity_penalty
        print(f"Unit {unit_id} inactivity penalty: -{inactivity_penalty} (consecutive: {consecutive_inactive[unit_id]})")
    else:
        consecutive_inactive[unit_id] = 0
    
    # Penalty for not visiting mass spots recently
    last_mass_visit[unit_id] = last_mass_visit.get(unit_id, 0) + 1
    no_mass_penalty = last_mass_visit[unit_id] * 0.25  # Slower scaling than inactivity
    reward -= no_mass_penalty
    reward_components['no_mass_penalty'] = -no_mass_penalty
    print(f"Unit {unit_id} no mass visit penalty: -{no_mass_penalty:.2f} (steps: {last_mass_visit[unit_id]})")
    
    # Penalty for being near map boundaries
    boundary_threshold = 200  # Units within 200 of edge get penalized
    dist_to_boundary = min(
        agent_unit['x'],  # Distance to x=0 edge
        agent_unit['z'],  # Distance to z=0 edge
        map_width - agent_unit['x'],  # Distance to x=max edge
        map_height - agent_unit['z']  # Distance to z=max edge
    )
    
    if dist_to_boundary < boundary_threshold:
        boundary_penalty = (boundary_threshold - dist_to_boundary) * 0.2
        reward -= boundary_penalty
        reward_components['boundary_penalty'] = -boundary_penalty
        print(f"Unit {unit_id} boundary penalty: -{boundary_penalty:.2f} (distance to edge: {dist_to_boundary:.1f})")
    
    previous_positions[unit_id] = (agent_unit['x'], agent_unit['z'])
    
    # === LOG REWARD COMPONENTS TO TENSORBOARD ===
    # This helps us see which parts of the reward function are driving behavior
    global step_counter
    for component_name, component_value in reward_components.items():
        writer.add_scalar(f'Reward_Components/{component_name}', component_value, step_counter)
    writer.add_scalar('Reward/total_reward', reward, step_counter)
    
    print(f"Unit {unit_id} reward: {reward}")
    return reward

def get_action(state, unit_x, unit_z, unit_y):
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
        feature_weights = agent(state.unsqueeze(0))[0]  # Get learned weights
        
        # === LOG LEARNED FEATURE WEIGHTS ===
        # These weights tell us what the agent has learned to value
        # Higher positive weights = agent thinks this feature is important for good outcomes
        global step_counter
        feature_names = ['distance_reduction', 'boundary_proximity', 'terrain_steepness', 
                        'move_magnitude', 'is_noop', 'height_change']
        for i, (name, weight) in enumerate(zip(feature_names, feature_weights)):
            writer.add_scalar(f'Feature_Weights/{name}', weight.item(), step_counter)
        
        # Get unvisited mass spots for feature computation
        unvisited_mass = [p for p in mass_spots if p not in visited_mass_spots]
        
        # Compute score for each action
        action_scores = []
        chosen_action_features = None  # Store features of the chosen action
        
        for action in range(action_size):
            # Compute features for this action
            features = compute_action_features(action, unit_x, unit_z, unit_y, unvisited_mass)
            features_tensor = torch.tensor(features, dtype=torch.float32)
            
            # Score = weighted sum of features
            score = torch.dot(feature_weights, features_tensor)
            action_scores.append(score.item())
        
        # Select best action
        best_action = int(np.argmax(action_scores))
        best_score = action_scores[best_action]
        
        # Get features for the chosen action to log
        chosen_action_features = compute_action_features(best_action, unit_x, unit_z, unit_y, unvisited_mass)
        
        # === LOG ACTION SELECTION DETAILS ===
        # Log statistics about the action scores distribution
        writer.add_scalar('Action_Selection/best_score', best_score, step_counter)
        writer.add_scalar('Action_Selection/mean_score', np.mean(action_scores), step_counter)
        writer.add_scalar('Action_Selection/std_score', np.std(action_scores), step_counter)
        writer.add_scalar('Action_Selection/chosen_action', best_action, step_counter)
        writer.add_scalar('Action_Selection/is_noop', 1.0 if best_action == NOOP_ACTION else 0.0, step_counter)
        
        # Log the features of the chosen action to see what made it attractive
        for i, (name, feature_val) in enumerate(zip(feature_names, chosen_action_features)):
            writer.add_scalar(f'Chosen_Action_Features/{name}', feature_val, step_counter)
        
        # Log action score histogram every 100 steps to see the distribution
        if step_counter % 100 == 0:
            writer.add_histogram('Action_Scores/distribution', np.array(action_scores), step_counter)
        
        return best_action

def train_agent(state, action, reward, next_state, done, unit_x, unit_z, unit_y, next_unit_x, next_unit_z, next_unit_y):
    """
    Train the feature weight network.
    The agent learns which features matter most for maximizing reward.
    
    Now with TensorBoard logging to visualize:
    - Training loss (how well predictions match targets)
    - Q-values (agent's estimate of action quality)
    - TD error (difference between prediction and target)
    - Gradient norms (to detect training instability)
    """
    # Get current feature weights
    current_weights = agent(state.unsqueeze(0))[0]
    
    # Compute features for the action taken
    unvisited_mass = [p for p in mass_spots if p not in visited_mass_spots]
    action_features = compute_action_features(action, unit_x, unit_z, unit_y, unvisited_mass)
    features_tensor = torch.tensor(action_features, dtype=torch.float32)
    
    # Current Q-value = dot(weights, features)
    current_q = torch.dot(current_weights, features_tensor)
    
    # Compute next state's best Q-value
    if not done:
        with torch.no_grad():
            next_weights = agent(next_state.unsqueeze(0))[0]
            next_unvisited = [p for p in mass_spots if p not in visited_mass_spots]
            
            # Find best action's Q-value in next state
            max_next_q = -float('inf')
            for next_action in range(action_size):
                next_features = compute_action_features(next_action, next_unit_x, next_unit_z, next_unit_y, next_unvisited)
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

def load_agent():
    try:
        agent.load_state_dict(torch.load('agent_weights_feature_based.pth'))
        print("Feature-based agent weights loaded.")
    except FileNotFoundError:
        print("No saved feature-based weights found, starting fresh.")

# Load weights at start
load_agent()

def format_action(action, unit_id, unit_x, unit_z, unit_y):    # Return None for NOOP (no command to send)
    if action == NOOP_ACTION:
        return None
    dy_idx = action // num_moves
    dx_idx = action % num_moves
    dx = move_steps[dx_idx]
    dz = move_steps[dy_idx]  # dy is actually dz
    target_x = unit_x + dx
    target_z = unit_z + dz
    distance = (dx**2 + dz**2)**0.5
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
                
                if "REWARD" in message:
                    # Increment step counter for TensorBoard logging
                    step_counter += 1
                    
                    # Compute reward
                    prev_health = previous_healths.get(unit['id'], 100)
                    reward = compute_reward(unit, prev_health)

                    # Print out how many mass spots have been visited
                    print(f"Visited mass spots: {len(visited_mass_spots)}/{len(mass_spots)}")
                    
                    # === LOG GAME STATE METRICS ===
                    # Track overall progress and game state
                    writer.add_scalar('Game_State/visited_mass_spots', len(visited_mass_spots), step_counter)
                    writer.add_scalar('Game_State/total_mass_spots', len(mass_spots), step_counter)
                    writer.add_scalar('Game_State/mass_completion_ratio', 
                                    len(visited_mass_spots) / max(1, len(mass_spots)), step_counter)
                    writer.add_scalar('Game_State/unit_health', unit['health'], step_counter)
                    writer.add_scalar('Game_State/friendly_unit_count', len(friendly_units), step_counter)
                    writer.add_scalar('Game_State/enemy_unit_count', len(enemy_units), step_counter)
                    
                    # Train agent if we have previous state
                    if unit['id'] in previous_states:
                        prev_unit = previous_positions.get(unit['id'], (unit['x'], unit['z']))
                        prev_y = previous_states[unit['id']][2].item()  # Previous y from state
                        train_agent(
                            previous_states[unit['id']], 
                            previous_actions[unit['id']], 
                            reward, 
                            state, 
                            False,
                            prev_unit[0], prev_unit[1], prev_y,
                            unit['x'], unit['z'], unit['y']
                        )
                
                previous_healths[unit['id']] = unit['health']
                previous_states[unit['id']] = state
                
                # Get action from agent (now using feature-based selection)
                action = get_action(state, unit['x'], unit['z'], unit['y'])
                print(f"Unit {unit['id']} chose action {action}")
                previous_actions[unit['id']] = action
                
                # Send action (or skip if NOOP)
                action_command = format_action(action, unit['id'], unit['x'], unit['z'], unit['y'])
                if action_command is not None:
                    conn.sendall(action_command.encode('utf-8'))
                    print(f"{unit['id']} has been sent action: {action_command.strip()}")
                else:
                    print(f"{unit['id']} executing NOOP (no command sent)")
            
            window.write_event_value('-SOCKET-', message)
        except Exception as e:
            print(f"Error in receive_messages: {e}")
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
        map_width = 0
        map_height = 0
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




