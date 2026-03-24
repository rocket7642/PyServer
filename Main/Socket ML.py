import socket
import sys
import threading

import FreeSimpleGUI as sg
import numpy as np
import pandas as pd

from Rewards import PeriodicRewards
import config
import runtime_state as state
import map_utils
import agent_core
from Connection.GameMessager import receive_messages



agent_core.load_agent()
print(f"TensorBoard logging to: runs/{state.run_name}")
print("View with: tensorboard --logdir=runs")

layout = [
    [sg.Multiline(size=(60, 10), key='-LOG-', autoscroll=True, disabled=True)],
    [sg.InputText()],
    [sg.Button('Ok'), sg.Button('Cancel')]
]

window = sg.Window("Socket Reader", layout, finalize=True)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    try:
        s.bind((config.HOST, config.PORT))
        s.listen()
        print(f"Server listening on {config.HOST}:{config.PORT}...")

        conn, addr = s.accept()

        state.map_heights = pd.read_csv('F:/BAR Beyond All Reason/Beyond-All-Reason/data/mapHeightInfo.txt', header=None).values
        map_size = state.map_heights.shape[0]

        try:
            with open('F:/BAR Beyond All Reason/Beyond-All-Reason/data/mapInfo.txt', 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('Map Width:'):
                        state.map_width = int(line.split(':')[1]) * 8
                    elif line.startswith('Map Height:'):
                        state.map_height = int(line.split(':')[1]) * 8
        except FileNotFoundError:
            print("mapInfo.txt not found, using height map dimensions")
            state.map_width = map_size
            state.map_height = map_size

        print(f"Map dimensions: {state.map_width} x {state.map_height}")

        map_utils.build_normalized_height_map()

        if state.normalized_map_heights is not None:
            full_map_norm = map_utils.normalize_image(state.normalized_map_heights)
            state.writer.add_image('Map/height_map', full_map_norm, 0, dataformats='HW')

        state.mass_spots = []
        try:
            with open('F:/BAR Beyond All Reason/Beyond-All-Reason/data/massInfo.txt', 'r') as f:
                for line in f:
                    if line.strip():
                        x, y, z = map(float, line.strip().split(','))
                        state.mass_spots.append((x, z))
        except FileNotFoundError:
            print("mass_points.csv not found, using empty mass spots.")
            state.mass_spots = []

        state.map_spots_norm = [
            (map_utils.normalize_x(x), map_utils.normalize_z(z))
            for x, z in state.mass_spots
        ]
        state.visited_mass_spots_norm = set()

        # Check to see if we have cached cost fields for this map to save time on future runs

        cache_loaded = map_utils.load_cached_cost_fields()

        if not cache_loaded:
            # Build cost fields for pathfinding
            print("Building terrain cost map...")
            map_utils.build_terrain_cost_map()
            print("Building mass point cost fields...")
            map_utils.build_mass_cost_fields()

            # Save cost fields for reuse on the same map in the future to save time
            map_utils.save_cached_cost_fields()

        # Print the terrain cost map to tensorboard for visualization
        # used to verify that the map is being generated correctly as to determine where impassable terrain is and where the agent should prefer to move
        if state.terrain_cost_map is not None:

            terrain_cost = state.terrain_cost_map
            passable_mask = np.isfinite(terrain_cost)

            terrain_cost_rgb = np.zeros((terrain_cost.shape[0], terrain_cost.shape[1], 3), dtype=np.float32)
            terrain_cost_rgb[~passable_mask] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

            if np.any(passable_mask):
                passable_vals = terrain_cost[passable_mask]
                min_cost = float(np.min(passable_vals))
                max_cost = float(np.max(passable_vals))

                passable_norm = np.zeros_like(terrain_cost, dtype=np.float32)
                if max_cost > min_cost:
                    passable_norm[passable_mask] = (
                        (terrain_cost[passable_mask] - min_cost) /
                        (max_cost - min_cost)
                    ).astype(np.float32)

                terrain_cost_rgb[..., 0][passable_mask] = passable_norm[passable_mask]
                terrain_cost_rgb[..., 1][passable_mask] = 1.0

            state.writer.add_image('Map/terrain_passability_map', terrain_cost_rgb, 0, dataformats='HWC')

        # Print combined mass cost field reachability to tensorboard
        if state.mass_cost_fields:
            h, w = state.terrain_cost_map.shape
            # Combine all mass cost fields: take the minimum cost across all mass points per cell
            combined = np.full((h, w), np.inf, dtype=np.float32)
            for cf in state.mass_cost_fields.values():
                combined = np.minimum(combined, cf)

            reachable_mask = np.isfinite(combined)
            mass_rgb = np.zeros((h, w, 3), dtype=np.float32)
            mass_rgb[~reachable_mask] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

            if np.any(reachable_mask):
                reachable_vals = combined[reachable_mask]
                min_val = float(np.min(reachable_vals))
                max_val = float(np.max(reachable_vals))

                cost_norm = np.zeros_like(combined, dtype=np.float32)
                if max_val > min_val:
                    cost_norm[reachable_mask] = (
                        (combined[reachable_mask] - min_val) /
                        (max_val - min_val)
                    ).astype(np.float32)

                # Green (near a mass point) → Yellow (far from all mass points)
                mass_rgb[..., 0][reachable_mask] = cost_norm[reachable_mask]
                mass_rgb[..., 1][reachable_mask] = 1.0

            state.writer.add_image('Map/mass_reachability_map', mass_rgb, 0, dataformats='HWC')

        

        server_thread = threading.Thread(target=receive_messages, args=(conn, addr, window), daemon=True)
        server_thread.start()
        print(f"[NEW CONNECTION] {addr} connected.")

        

    except Exception as exc:
        print(f"An error occurred: {exc}")
        sys.exit(1)

    def finalize_match(success, reason):
        if state.match_finalized:
            return

        PeriodicRewards.finalize_all_units(success, reason)
        agent_core.save_agent()
        state.match_finalized = True

    cancel_requested = False
    while True:
        event, values = window.read()
        if event == 'Cancel' or event == sg.WIN_CLOSED:
            cancel_requested = True
            state.forced_terminal_success = True
            print("Cancel received. Finalizing match as success.")
            finalize_match(True, "user_cancelled")
            break

        if event == 'Ok':
            user_input = values[0] + "\n"
            if user_input.strip() != "":
                pass

        if event == '-SOCKET-':
            window['-LOG-'].update(
                f"{state.units}\n\n{state.eUnits}\n\n{state.eKUnits}",
                append=False
            )

    # Finalize/close down before exiting
    if cancel_requested:
        try:
            conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        conn.close()

    if not state.match_finalized:
        print("Closing connection and saving agent (no terminal result finalized).")
        agent_core.save_agent()
    else:
        print("Match outcome already finalized; skipping duplicate save.")

    
    window.close()

    print("Closing TensorBoard writer...")
    state.writer.close()
    print(f"TensorBoard logs saved to: runs/{state.run_name}")
    print("View results with: tensorboard --logdir=runs")




