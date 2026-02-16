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

# RL Agent
class RTSAgent(nn.Module):
    def __init__(self, input_size, action_size):
        super(RTSAgent, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

agent = RTSAgent(input_size=17, action_size=action_size)
optimizer = optim.Adam(agent.parameters(), lr=0.001)
criterion = nn.MSELoss()

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

def compute_reward(agent_unit, prev_health):
    reward = 0
    unit_id = agent_unit['id']
    
    # Punish damage taken
    if agent_unit['health'] < prev_health:
        damage_penalty = prev_health - agent_unit['health']
        reward -= damage_penalty
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
            print(f"Unit {unit_id} distance improvement: +{dist_improvement:.2f}")
        elif current_dist > prev_dist:
            away_penalty = (current_dist - prev_dist) * 0.05
            reward -= away_penalty
            print(f"Unit {unit_id} moving away penalty: -{away_penalty:.2f}")
        
        # Base reward inversely proportional to distance
        base_dist_reward = max(0, 750 - current_dist) * 0.1
        reward += base_dist_reward
        print(f"Unit {unit_id} base distance reward: +{base_dist_reward:.2f}")
        
        previous_distances[unit_id] = current_dist
    
    # Penalty for not moving (staying in similar position)
    prev_pos = previous_positions.get(unit_id, (agent_unit['x'], agent_unit['z']))
    dist_moved = ((prev_pos[0]-agent_unit['x'])**2 + (prev_pos[1]-agent_unit['z'])**2)**0.5
    if dist_moved < 5:  # If moved less than 5 units
        consecutive_inactive[unit_id] = consecutive_inactive.get(unit_id, 0) + 3
        inactivity_penalty = consecutive_inactive[unit_id]
        reward -= inactivity_penalty
        print(f"Unit {unit_id} inactivity penalty: -{inactivity_penalty} (consecutive: {consecutive_inactive[unit_id]})")
    else:
        consecutive_inactive[unit_id] = 0
    
    # Penalty for not visiting mass spots recently
    last_mass_visit[unit_id] = last_mass_visit.get(unit_id, 0) + 1
    no_mass_penalty = last_mass_visit[unit_id] * 0.25  # Slower scaling than inactivity
    reward -= no_mass_penalty
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
        print(f"Unit {unit_id} boundary penalty: -{boundary_penalty:.2f} (distance to edge: {dist_to_boundary:.1f})")
    
    previous_positions[unit_id] = (agent_unit['x'], agent_unit['z'])
    
    print(f"Unit {unit_id} reward: {reward}")
    return reward

def get_action(state):
    with torch.no_grad():
        q_values = agent(state.unsqueeze(0))
        return torch.argmax(q_values).item()

def train_agent(state, action, reward, next_state, done):
    # Simple Q-learning update
    current_q = agent(state.unsqueeze(0))[0][action]
    next_q = agent(next_state.unsqueeze(0)).max() if not done else 0
    target = reward + 0.99 * next_q  # gamma = 0.99
    
    loss = criterion(current_q, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def save_agent():
    torch.save(agent.state_dict(), 'agent_weights.pth')
    print("Agent weights saved.")

def load_agent():
    try:
        agent.load_state_dict(torch.load('agent_weights.pth'))
        print("Agent weights loaded.")
    except FileNotFoundError:
        print("No saved weights found, starting fresh.")

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
    global units, eUnits, eKUnits, visited_mass_spots
    
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
                    # Compute reward
                    prev_health = previous_healths.get(unit['id'], 100)
                    reward = compute_reward(unit, prev_health)

                    # Print out how many mass spots have been visited
                    print(f"Visited mass spots: {len(visited_mass_spots)}/{len(mass_spots)}")
                    
                    # Train agent if we have previous state
                    if unit['id'] in previous_states:
                        train_agent(previous_states[unit['id']], previous_actions[unit['id']], reward, state, False)
                
                previous_healths[unit['id']] = unit['health']
                previous_states[unit['id']] = state
                
                # Get action from agent
                action = get_action(state)
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




