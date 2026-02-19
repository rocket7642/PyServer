import socket
import sys
import threading

import FreeSimpleGUI as sg
import pandas as pd

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

        server_thread = threading.Thread(target=receive_messages, args=(conn, addr, window), daemon=True)
        server_thread.start()
        print(f"[NEW CONNECTION] {addr} connected.")
    except Exception as exc:
        print(f"An error occurred: {exc}")
        sys.exit(1)

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Cancel':
            break

        if event == 'Ok':
            user_input = values[0] + "\n"
            if user_input.strip() != "":
                pass

        if event == '-SOCKET-':
            window['-LOG-'].update(
                f"{state.units}{state.eUnits}{state.eKUnits}",
                append=False
            )

    window.close()
    agent_core.save_agent()

    print("Closing TensorBoard writer...")
    state.writer.close()
    print(f"TensorBoard logs saved to: runs/{state.run_name}")
    print("View results with: tensorboard --logdir=runs")




