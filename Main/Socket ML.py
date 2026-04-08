import socket
import sys
import threading
import signal
import logging
import os

import numpy as np
import pandas as pd

from Rewards import PeriodicRewards
import config
import runtime_state as state
import map_utils
import agent_core
from Connection.GameMessager import receive_messages


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global shutdown event for graceful termination
shutdown_event = threading.Event()
server_connection = None
server_socket = None
server_thread = None
stop_sentinel_path = os.path.join(os.path.dirname(__file__), 'stop.txt')

agent_core.load_agent()
logger.info(f"TensorBoard logging to: runs/{state.run_name}")
logger.info("View with: tensorboard --logdir=runs")


def signal_handler(signum, frame):
    """Handle SIGINT (Ctrl+C) and SIGTERM for graceful shutdown."""
    logger.info("\n[SIGNAL] Shutdown signal received. Finalizing match...")
    shutdown_event.set()
    state.forced_terminal_success = True


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination signal


try:
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((config.HOST, config.PORT))
    server_socket.listen()
    logger.info(f"Server listening on {config.HOST}:{config.PORT}...")

    server_connection, addr = server_socket.accept()
    logger.info(f"[NEW CONNECTION] {addr} connected.")

    map_heights_path = 'F:/BAR Beyond All Reason/Beyond-All-Reason/data/mapHeightInfo.txt'
    map_info_path = 'F:/BAR Beyond All Reason/Beyond-All-Reason/data/mapInfo.txt'
    mass_info_path = 'F:/BAR Beyond All Reason/Beyond-All-Reason/data/massInfo.txt'

    state.map_heights_source = map_heights_path
    state.map_spots_source = mass_info_path

    state.map_heights = pd.read_csv(map_heights_path, header=None).values
    map_size = state.map_heights.shape[0]

    try:
        with open(map_info_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('Map Width:'):
                    state.map_width = int(line.split(':')[1]) * 8
                elif line.startswith('Map Height:'):
                    state.map_height = int(line.split(':')[1]) * 8
    except FileNotFoundError:
        logger.warning("mapInfo.txt not found, using height map dimensions")
        state.map_width = map_size
        state.map_height = map_size

    logger.info(f"Map dimensions: {state.map_width} x {state.map_height}")

    map_utils.build_normalized_height_map()

    if state.normalized_map_heights is not None:
        full_map_norm = map_utils.normalize_image(state.normalized_map_heights)
        state.writer.add_image('Map/height_map', full_map_norm, 0, dataformats='HW')

    state.mass_spots = []
    try:
        with open(mass_info_path, 'r') as f:
            for line in f:
                if line.strip():
                    x, y, z = map(float, line.strip().split(','))
                    state.mass_spots.append((x, z))
    except FileNotFoundError:
        logger.warning("mass_points.csv not found, using empty mass spots.")
        state.mass_spots = []

    state.map_spots_norm = [
        (map_utils.normalize_x(x), map_utils.normalize_z(z))
        for x, z in state.mass_spots
    ]
    state.visited_mass_spots_norm = set()
    state.mass_cycle_completions = 0
    state.writer.add_scalar('Game_State/mass_cycles_completed', state.mass_cycle_completions, 0)

    # Check to see if we have cached cost fields for this map to save time on future runs

    cache_loaded = map_utils.load_cached_cost_fields()

    if not cache_loaded:
        # Build cost fields for pathfinding
        logger.info("Building terrain cost map...")
        map_utils.build_terrain_cost_map()
        logger.info("Building mass point cost fields...")
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

    # Start the game message receiver thread (no window parameter needed)
    server_thread = threading.Thread(target=receive_messages, args=(server_connection, addr), daemon=True)
    server_thread.start()

except Exception as exc:
    logger.error(f"An error occurred during initialization: {exc}")
    sys.exit(1)


def finalize_match(success, reason):
    """Finalize the match: save rewards and agent if training is enabled."""
    if state.match_finalized or not config.SHOULD_TRAIN:
        return

    logger.info(f"Finalizing match: success={success}, reason={reason}")
    PeriodicRewards.finalize_all_units(success, reason)
    agent_core.save_agent()
    state.match_finalized = True


def cleanup_connection():
    """Gracefully close the server connection."""
    global server_connection, server_socket, server_thread
    
    if server_connection:
        try:
            server_connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        finally:
            server_connection.close()
    
    if server_socket:
        try:
            server_socket.close()
        except OSError:
            pass
    
    if server_thread and server_thread.is_alive():
        server_thread.join(timeout=3.0)


def sentinel_stop_requested(path):
    """Return True when the sentinel file contains a stop command."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip().lower() in {'stop', '1', 'true'}
    except FileNotFoundError:
        return False


# Main execution loop: wait for game to complete or shutdown signal
logger.info("Game server running. Press Ctrl+C to stop gracefully.")
logger.info("Monitoring for game completion or termination signal...\n")

try:
    # Wait for shutdown signal (Ctrl+C) or game completion
    # The daemon threads (server_thread) will run in the background
    while not shutdown_event.is_set():
        if sentinel_stop_requested(stop_sentinel_path):
            logger.info(f"[SENTINEL] Stop requested via {stop_sentinel_path}")
            state.forced_terminal_success = True
            shutdown_event.set()
            break

        # Check periodically if game has ended via state variable
        # (assumes receive_messages thread sets some completion flag)
        shutdown_event.wait(timeout=1.0)

except KeyboardInterrupt:
    logger.info("[KEYBOARD] Interrupted - initiating shutdown...")

finally:
    logger.info("\n[SHUTDOWN] Closing connections and finalizing...")
    
    # Stop the socket connection
    cleanup_connection()
    
    # Determine finalization reason
    if state.forced_terminal_success:
        reason = "terminated"
    else:
        reason = "completed" if state.match_finalized else "interrupted"
    
    # Finalize if not already done
    if not state.match_finalized:
        logger.info(f"Closing connection.")
        if config.SHOULD_TRAIN:
            logger.info("Finalizing rewards and saving agent... (no terminal result finalized)")
            PeriodicRewards.finalize_all_units(False, reason)
            agent_core.save_agent()
        else:
            logger.info("Training disabled; skipping reward finalization and agent save.")
    else:
        logger.info("Match outcome already finalized; skipping duplicate save.")
    
    logger.info("Closing TensorBoard writer...")
    state.writer.close()
    logger.info(f"TensorBoard logs saved to: runs/{state.run_name}")
    logger.info("View results with: tensorboard --logdir=runs")
    logger.info("\n[COMPLETE] Server shutdown successfully.")




