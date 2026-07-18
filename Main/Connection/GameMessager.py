import traceback
import time
import socket
import numpy as np

import config
import runtime_state as state
import map_utils
import agent_core
import unit_defs
from Rewards import PeriodicRewards

def parse_units(message, header):
	"""Parse a socket message into a list of unit dictionaries by extracting lines between a header and END marker."""
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
					if len(parts) >= 9:
						unit = {
							'id': int(parts[0]),
							'name': parts[1],
							'x': float(parts[2]),
							'y': float(parts[3]),
							'z': float(parts[4]),
							#'range': float(parts[5]),
							'health': float(parts[5]),
							'speed': float(parts[6]),
							'is_constructing': int(parts[7]), # 1 or 0
							'active_build_progress': float(parts[8])
						}
						# Enrich with all data so PyTorch tensors have fixed shapes
						unitType = unit_defs.get_unit_type(unit['name'])
						weapon_info = unit_defs.get_weapon_info(unit['name'])
						ranges = unit_defs.get_unit_ranges(unit['name'])
						costs = unit_defs.get_costs(unit['name'])
						sizes = unit_defs.get_unit_size(unit['name'])

						unit.update(weapon_info)
						unit.update(ranges)
						unit.update(costs)
						unit.update(sizes)
						unit.update({"type": unitType})
						unit.update({"cloaked": False}) # Will be updated later based on vision vs known enemy units (we don't currently care about our units cloaking)
						
						units_list.append(unit)
					else:
						print(f"[WARNING] Malformed unit line: '{line}'")
				i += 1
		i += 1

	# Clear out duplicates based on id, keeping the last occurrence (which should be the most recent state)
	units_list = list({unit['id']: unit for unit in units_list}.values())

	return units_list


def format_action(action, unit_id, unit_x, unit_z, unit_y, target_x=None, target_z=None):
	"""Format an action into a command string to send to the game, choosing queued or immediate based on distance."""
	if action == config.NOOP_ACTION or target_x is None or target_z is None:
		return None
	
	if action == "BUILD":
		# Assuming game engine expects a build command, e.g., "BU [unit_id] [target_x] [target_z] [target_y] [building_type]"
		# You can change the "PlaceholderBuilding" string or the actual packet syntax as needed by your C# side.
		command = "C: BU I" 
		return f"{command} {unit_id} {target_x} {target_z} {unit_y} armrad\n"
	
	# Default Move logic
	distance = ((target_x - unit_x) ** 2 + (target_z - unit_z) ** 2) ** 0.5
	command = "C: MU Q" if distance > 50 else "C: MU I"
	return f"{command} {unit_id} {target_x} {target_z} {unit_y}\n"


def perform_handshake(conn, addr):
	"""Send START first and wait for READY with a timeout before entering the main receive loop."""
	handshake_timeout = getattr(config, 'HANDSHAKE_TIMEOUT_SECONDS', 5.0)
	handshake_retries = getattr(config, 'HANDSHAKE_MAX_RETRIES', 30)
	original_timeout = conn.gettimeout()

	try:
		conn.settimeout(handshake_timeout)
		for attempt in range(1, handshake_retries + 1):
			conn.sendall("START\n".encode('utf-8'))
			print(f"[HANDSHAKE] Sent START to {addr} (attempt {attempt}/{handshake_retries})")

			try:
				data = conn.recv(1024)
			except socket.timeout:
				print(f"[HANDSHAKE] Timeout waiting for READY from {addr}")
				continue

			if not data:
				print(f"[HANDSHAKE] {addr} closed the connection during handshake")
				return False

			response = data.decode('utf-8', errors='ignore').strip()
			print(f"[HANDSHAKE] Received from {addr}: {response}")

			if response:
				return True

		print(f"[HANDSHAKE FAILED] No message from {addr} after {handshake_retries} attempts")
		return False
	finally:
		conn.settimeout(original_timeout)


def receive_messages(conn, addr):
	"""Main loop that receives game state messages, runs the agent's decision-making, trains on transitions, and sends commands back."""
	def finalize_match(success, reason):
		if state.match_finalized or not config.SHOULD_TRAIN:
			return

		PeriodicRewards.finalize_all_units(success, reason)
		agent_core.save_agent()
		state.match_finalized = True

	# Load unit definitions once on first connection
	if not state.unit_defs_loaded:
		unit_defs.load_unit_defs()
		state.unit_defs_loaded = True

	if not perform_handshake(conn, addr):
		print(f"[DISCONNECTED] {addr} failed handshake.")
		conn.close()
		return

	while True:
		try:
			data = conn.recv(1024)
			
			if not data:
				print(f"[DISCONNECTED] {addr} closed the connection.")
				break

			message = data.decode('utf-8')
			print(f"[{addr}] {message}")

			if "FRIENDLY_UNITS" in message:
				state.units = parse_units(message, "FRIENDLY_UNITS")
			if "ENEMY_UNITS" in message:
				state.eUnits = parse_units(message, "ENEMY_UNITS")
			if "KNOWN_ENEMY_UNITS" in message:
				state.eKUnits = parse_units(message, "KNOWN_ENEMY_UNITS")
			if "RADAR_ENEMY_UNITS" in message:
				state.eRUnits = parse_units(message, "RADAR_ENEMY_UNITS") # This will need to be integrated into one of the other lists (known likely)
			if "RESOURCES" in message:
				lines = message.strip().split('\n')
				for line in lines:
					if line.startswith("RESOURCES"):
						parts = line.split(' ')
						if len(parts) >= 3:
							try:
								state.fEnergy = float(parts[1])
								state.fMass = float(parts[2])
							except ValueError:
								print(f"[WARNING] Malformed RESOURCES line: '{line}'")
						else:
							print(f"[WARNING] Malformed RESOURCES line: '{line}'")
			finished_units = []
			if "FINISHED" in message:
				in_finished = False
				for line in message.strip().split('\n'):
					if line.startswith("FINISHED"):
						in_finished = True
						continue
					if in_finished:
						if line.startswith("END"):
							break
						parts = line.split(' ')
						if len(parts) >= 2:
							try:
								finished_units.append((int(parts[0]), parts[1]))
								print(f"[FINISHED] Structure {parts[1]} (id {parts[0]}) completed")
							except ValueError:
								print(f"[WARNING] Malformed FINISHED line: '{line}'")
			# Merge radar into known enemy units, ensuring no duplicates (radar may have some units not currently visible in known due to fog of war, but if a unit is in both, we want to avoid duplicates)
			# Need to keep in mind, enemies can enter radar without being identified, this requires us to generalize what they are until confirmed.
			for unit in state.eRUnits:
				if all(unit['id'] != eu['id'] for eu in state.eKUnits):
					state.eKUnits.append(unit)

			# Verify if enemy in eUnits and eKUnits, if so remove from known enemy units (to avoid duplicates)
			# IE known is a subset of enemy, but may have some units not currently visible (fog of war)
			for unit in state.eKUnits:
				if any(unit['id'] == eu['id'] for eu in state.eUnits):
					state.eKUnits.remove(unit)

			# Remove from known if the enemy location has been passed and thus, their location is entirely unknown
			# Requires distance checks between enemy unit loc and friendly LOS

			if "GAMESTOP" in message or "GAME_ENDED" in message:
				# End state, close connection and save agent
				print(f"[GAME STOPPED] {addr} sent GAMESTOP. Closing connection.")
				finalize_match(True, "game_ended")
				break

			if "AI_KILLED" in message or "TEAM_DIED" in message:
				print(f"[AI KILLED] {addr} sent AI_KILLED. Finalizing results and closing connection.")
				finalize_match(False, "death")
				break

			friendly_units = state.units
			enemy_units = state.eUnits

			for unit in state.eKUnits:
				if all(unit['id'] != eu['id'] for eu in enemy_units):
					enemy_units.append(unit)

			print(f"Parsed {len(friendly_units)} friendly units, {len(enemy_units)} enemy units")
			print(f"Current time: {time.time()}")
			print(f"Current step counter: {state.step_counter}")
			if friendly_units:
				print(f"Sample unit: {friendly_units[0]}")

			if "TURN" in message:
				state.step_counter += 1

			for unit in friendly_units:
				# Verify if its a moveable unit or commandable unit (ie fighters or factories)
				if unit['type'] != "UNIT":
					print(f"Unit {unit['id']} ({unit['name']}) is not a commandable unit. Skipping.")
					continue

				state_vec = agent_core.get_state(unit, friendly_units, enemy_units)
				state_no_map = agent_core.get_state_no_map(unit, friendly_units, enemy_units)

				if "TURN" in message:
					try:
						# state.step_counter += 1
						now = time.time()
						PeriodicRewards.init_segment_tracking(unit)

						prev_health = state.previous_healths.get(unit['id'], unit['health'])
						prev_state = state.previous_states_no_map.get(unit['id'], None)
						prev_action = state.previous_actions.get(unit['id'], None)
						prev_pos = state.previous_positions.get(unit['id'], (unit['x'], unit['z']))
						prev_y = state.previous_y_positions.get(unit['id'], unit['y'])
						if unit['is_constructing'] == 1:
							state.previous_build_vision_baseline = np.sum(state.vision_image) if state.vision_image is not None else 0.0 # used to garentee it is maintained
						else:
							state.previous_build_vision_baseline = 0
						damage_taken = max(0.0, prev_health - unit['health'])
						enemy_range_image = map_utils.generate_enemy_range_image(
							enemy_units,
							state.map_width,
							state.map_height,
							state.normalized_map_heights.shape if state.normalized_map_heights is not None else None
						)

						state.previous_vision_scores.append(np.sum(state.vision_image) if state.vision_image is not None else 0.0)
						vision_image = map_utils.generate_vision_image(
							friendly_units,
							state.map_width,
							state.map_height,
							state.normalized_map_heights
						)
						state.vision_image = vision_image

						# Determine if any unknown units from the known unit list are within LOS, if they are they are cloaked
						# This utilizes the vision image created above
						for e_unit in state.eKUnits:
							if any(e_unit['id'] == eu['id'] for eu in enemy_units):
								continue # This unit is currently visible, so skip
							nx = map_utils.normalize_x(e_unit['x'])
							nz = map_utils.normalize_z(e_unit['z'])
							vis_value = vision_image[int(nz * state.vision_image.shape[0]), int(nx * state.vision_image.shape[1])]
							if vis_value > 0.5: # If the vision image has any value greater than 0.5, that means LOS (and thus cloaked), otherwise it is either in radar or out of view.
								e_unit['cloaked'] = True
							else:
								e_unit['cloaked'] = False


						if prev_state is not None and prev_action is not None:
							unvisited_mass = [
								p for p in state.map_spots_norm if (p[0], p[1]) not in state.visited_mass_spots_norm
								if p not in state.visited_mass_spots_norm
							]
							# potential_reward, components = PeriodicRewards.compute_move_potential(
							# 	prev_pos,
							# 	(unit['x'], unit['z']),
							# 	prev_y,
							# 	unit['y'],
							# 	unvisited_mass,
							# 	enemy_range_image=enemy_range_image,
							# 	vision_image=vision_image,
							# 	unit_id=unit['id'],
							# )
							if unit['is_constructing'] == 1:
								potential_reward, components = PeriodicRewards.compute_build_potential(
									prev_pos,
									(unit['x'], unit['z']),
									prev_y,
									unit['y'],
									enemy_range_image=enemy_range_image,
									vision_image=vision_image,
									unit=unit
								)
							else:
								potential_reward, components = PeriodicRewards.compute_move_potential(
									prev_pos,
									(unit['x'], unit['z']),
									prev_y,
									unit['y'],
									unvisited_mass,
									enemy_range_image=enemy_range_image,
									vision_image=vision_image,
									unit_id=unit['id'],
								)

							if damage_taken > 0.0:
								damage_penalty = damage_taken * config.IMMEDIATE_DAMAGE_PENALTY_SCALE
								potential_reward -= damage_penalty
								components['damage_taken'] = -damage_penalty

							cancel_penalty = state.cancel_command_penalties.pop(unit['id'], 0.0)
							if cancel_penalty:
								potential_reward -= cancel_penalty
								state.writer.add_scalar(
									'Move_Potential/cancel_command_penalty',
									-cancel_penalty,
									state.step_counter
								)

							state.segment_stats[unit['id']]['distance'] += PeriodicRewards.terrain_adjusted_distance(
								prev_pos, (unit['x'], unit['z']), prev_y, unit['y']
							)
							state.segment_stats[unit['id']]['height_change'] += abs(
								map_utils.normalize_y(unit['y']) - map_utils.normalize_y(prev_y)
							)
							if damage_taken > 0.0:
								state.segment_stats[unit['id']]['damage_taken'] += damage_taken
							state.segment_stats[unit['id']]['steps'] += 1
							if unit['is_constructing'] == 1:
								state.segment_stats[unit['id']]['build_steps'] += 1

							prev_progress = state.previous_build_progress.get(unit['id'], 0.0)
							current_progress = unit['active_build_progress']

							structure_finished = any(uid != unit['id'] for uid, name in finished_units)
							heuristic_finished = prev_progress > 0.7 and current_progress < 0.1 and unit["is_constructing"] == 0
							if structure_finished or heuristic_finished:
								# This indicates either a build finish, a cancel near completion (I will assume for the sake of the integration that it is a finish however)
								state.segment_stats[unit['id']]['buildings_built'] += 1
								print(f"Unit {unit['id']} likely completed a building. Total buildings built in this segment: {state.segment_stats[unit['id']]['buildings_built']}")
								# Add the vision increase as a reward for completing the building.
								state.segment_stats[unit['id']]['value_from_building'] += (np.sum(state.vision_image) if state.vision_image is not None else 0.0) - state.previous_build_vision_baseline
								# Release the build commitment now that the structure is complete
								state.build_committed_target.pop(unit['id'], None)
								state.build_committed_since_step.pop(unit['id'], None)
								state.build_committed_distance.pop(unit['id'], None)
								print(f"[COMMIT] Unit {unit['id']} commitment released on completion at step {state.step_counter}")
								state.build_released_step[unit['id']] = state.step_counter
								
							state.previous_build_progress[unit['id']] = current_progress

							state.segment_buffers[unit['id']].append({
								'state': prev_state,
								'discrete_action': state.previous_discrete_actions.get(unit['id'], config.ACTION_MOVE),
								'action': prev_action,
								'action_kind': state.previous_action_kinds.get(unit['id'], None),
								'next_state': state_no_map,
								'unit_x': prev_pos[0],
								'unit_z': prev_pos[1],
								'unit_y': prev_y,
								'next_unit_x': unit['x'],
								'next_unit_z': unit['z'],
								'next_unit_y': unit['y'],
								'target_x': state.previous_targets.get(unit['id'], (unit['x'], unit['z']))[0],
								'target_z': state.previous_targets.get(unit['id'], (unit['x'], unit['z']))[1],
								'mass_destination': state.mass_destinations.get(unit['id'], None),
								'potential_reward': potential_reward,
								'forced_build': state.previous_forced_builds.get(unit['id'], False),
								'build_committed_target': state.build_committed_target.get(unit['id']),
								'build_committed_since_step': state.build_committed_since_step.get(unit['id']),
								'build_committed_distance': state.build_committed_distance.get(unit['id']),
								'decision_snapshot': state.decision_snapshots.get(unit['id'], None)
							})

							state.writer.add_scalar('Move_Potential/total', potential_reward, state.step_counter)
							state.writer.add_scalar('Move_Potential/distance', components['distance'], state.step_counter)
							state.writer.add_scalar('Move_Potential/direction', components['direction'], state.step_counter)
							state.writer.add_scalar('Move_Potential/height_jump', components['height_jump'], state.step_counter)
							state.writer.add_scalar('Move_Potential/path_danger', components.get('path_danger', 0.0), state.step_counter)
							state.writer.add_scalar('Move_Potential/path_terrain', components.get('path_terrain', 0.0), state.step_counter)
							state.writer.add_scalar('Move_Potential/enemy_avoidance', components.get('enemy_avoidance', 0.0), state.step_counter)
							state.writer.add_scalar('Move_Potential/damage_taken', components.get('damage_taken', 0.0), state.step_counter)
							state.writer.add_scalar('Move_Potential/vision_coverage', components.get('vision_coverage', 0.0), state.step_counter)
							state.writer.add_scalar('Move_Potential/build_progress', components.get('build_progress', 0.0), state.step_counter)
							state.writer.add_scalar('Move_Potential/build_completion', components.get('build_completion', 0.0), state.step_counter)
						elif unit['id'] not in state.previous_actions:
							print(
								f"[INFO] Skipping move potential for unit {unit['id']} "
								"(no previous action)."
							)

						reached_mass, _ = PeriodicRewards.check_mass_reached(unit)
						if reached_mass:
							state.mass_destinations.pop(unit['id'], None)
							state.mass_destination_distances.pop(unit['id'], None)
							state.mass_spot_blocked_until.pop(unit['id'], None)
							if config.TRAIN_AT_EACH_MASS_POINT and config.SHOULD_TRAIN:
								conn.sendall(f"C: PAUSE {state.pause_time}\n".encode('utf-8'))
								PeriodicRewards.finalize_segment_training(unit['id'], True, "mass_reached")
								conn.sendall("C: UNPAUSE\n".encode('utf-8'))
								print(f"Unit {unit['id']} reached a mass spot. Segment trained.")
							else:
								PeriodicRewards.finalize_segment_training(unit['id'], True, "mass_reached")
								print(f"Unit {unit['id']} reached a mass spot. Segment deferred.")

						if state.mass_spots and len(state.visited_mass_spots) == len(state.mass_spots):
							state.mass_cycle_completions += 1
							state.writer.add_scalar(
								'Game_State/mass_cycles_completed',
								state.mass_cycle_completions,
								state.step_counter
							)
							conn.sendall(f"C: PAUSE {state.pause_time}\n".encode('utf-8')) # Pause during bookkeeping
							if config.END_MATCH_WHEN_ALL_MASS_REACHED or state.evalRun: # Make it so that the eval run does this as it runs a very specific map and we want to see final results at the end of the match, but for training, we want to keep going and gather more data even after all mass spots are reached.
								finalize_match(True, "all_mass_cycle_complete")
								print("All mass spots reached. Match finalized.")
								return
							else:
								PeriodicRewards.finalize_cycle_segments(True, "all_mass_cycle_complete")
							conn.sendall("C: UNPAUSE\n".encode('utf-8')) # Resume Game
							state.visited_mass_spots.clear()
							state.visited_mass_spots_norm.clear()
							state.mass_destinations.clear()
							state.mass_destination_distances.clear()
							state.mass_spot_blocked_until.clear()
							print("All mass spots reached. Cycle reset; continuing match/training set.")
						else:
							last_time = state.segment_stats[unit['id']]['last_mass_time']
							if now - last_time >= config.EPISODE_TIMEOUT_SECONDS:
								if config.TRAIN_AT_EACH_MASS_POINT and config.SHOULD_TRAIN:
									conn.sendall(f"C: PAUSE {state.pause_time}\n".encode('utf-8'))
								PeriodicRewards.finalize_segment_training(unit['id'], False, "timeout")
								if config.TRAIN_AT_EACH_MASS_POINT:
									conn.sendall("C: UNPAUSE\n".encode('utf-8'))
								print(
									f"Unit {unit['id']} timed out without reaching a mass spot. "
									f"Segment {'trained' if config.TRAIN_AT_EACH_MASS_POINT else 'deferred'}."
								)

						print(f"Visited mass spots: {len(state.visited_mass_spots)}/{len(state.mass_spots)}")

						state.writer.add_scalar(
							'Game_State/visited_mass_spots',
							len(state.visited_mass_spots),
							state.step_counter
						)
						state.writer.add_scalar(
							'Game_State/total_mass_spots',
							len(state.mass_spots),
							state.step_counter
						)
						state.writer.add_scalar(
							'Game_State/mass_completion_ratio',
							len(state.visited_mass_spots) / max(1, len(state.mass_spots)),
							state.step_counter
						)
						state.writer.add_scalar('Game_State/unit_health', unit['health'], state.step_counter)
						state.writer.add_scalar('Game_State/friendly_unit_count', len(friendly_units), state.step_counter)
						state.writer.add_scalar('Game_State/enemy_unit_count', len(enemy_units), state.step_counter)
					except Exception as reward_err:
						print(f"[ERROR during reward/training for unit {unit['id']}]")
						print(f"  Exception: {reward_err}")
						traceback.print_exc()

				try:
					state.previous_healths[unit['id']] = unit['health']
					state.previous_states[unit['id']] = state_vec
					state.previous_states_no_map[unit['id']] = state_no_map
					state.previous_y_positions[unit['id']] = unit['y']
					state.previous_positions[unit['id']] = (unit['x'], unit['z'])

					# Bypass weight system, prevent it from doing something while building
					# if unit['is_constructing'] == 1:
					# 	action = config.NOOP_ACTION
					# 	best_target = (map_utils.normalize_x(unit['x']), map_utils.normalize_z(unit['z']))
					# 	best_score = 0.0
					# 	best_candidate_kind = 'noop'
					# 	discrete_action = config.ACTION_BUILD
					# else:
					action, best_target, best_score, best_candidate_kind, discrete_action, forced_build_this_step, decision_snapshot = agent_core.get_action(
						state_vec,
						unit['x'],
						unit['z'],
						unit['y'],
						unit['id']
					)

					# Store chosen discrete action (Move vs Build) for training
					state.previous_discrete_actions[unit['id']] = discrete_action
					state.previous_forced_builds[unit['id']] = forced_build_this_step
					state.decision_snapshots[unit['id']] = decision_snapshot

					denorm_tx = map_utils.denormalize_x(best_target[0])
					denorm_tz = map_utils.denormalize_z(best_target[1])
					print(
						f"Unit {unit['id']} chose action {action} -> target normalized "
						f"({best_target[0]:.2f}, {best_target[1]:.2f}), denormalized "
						f"({denorm_tx:.1f}, {denorm_tz:.1f}) (score {best_score:.2f})" 
						# Best to add the reasons the cost are what it is

					)

					best_target_world = (denorm_tx, denorm_tz)
				except Exception as action_err:
					print(f"[ERROR during action selection for unit {unit['id']}]")
					print(f"  Exception: {action_err}")
					traceback.print_exc()
					continue

				try:
					last_target = state.previous_targets.get(unit['id'], None)
					state.previous_command_steps[unit['id']] = state.previous_command_steps.get(unit['id'], 0) + 1

					if action == config.NOOP_ACTION:
						sample_target_x = unit['x']
						sample_target_y = unit['y']
						sample_target_z = unit['z']
						sample_cmd_id = None
					else:
						sample_target_x = best_target_world[0]
						sample_target_y = unit['y']
						sample_target_z = best_target_world[1]
						sample_cmd_id = 10

					PeriodicRewards.record_match_sample(
						unit_id=unit['id'],
						friendly_units=friendly_units,
						enemy_units=enemy_units,
						action_type=action,
						target_x=sample_target_x,
						target_y=sample_target_y,
						target_z=sample_target_z,
						action_score=best_score,
						cmd_id=sample_cmd_id,
					)

					action_command = format_action(
						action,
						unit['id'],
						unit['x'],
						unit['z'],
						unit['y'],
						best_target_world[0],
						best_target_world[1]
					)

					if discrete_action == config.ACTION_BUILD and action_command is not None:
						state.last_build_step[unit['id']] = state.step_counter
						existing_commit = state.build_committed_target.get(unit['id'])
						same_target = (
							existing_commit is not None
							and ((existing_commit[0] - best_target_world[0]) ** 2 + (existing_commit[1] - best_target_world[1]) ** 2) ** 0.5 <= config.COMMAND_DISTANCE_EPS
						)
						if existing_commit is None or (best_candidate_kind != 'committed_build' and not same_target):
							# New commitment, or a deliberate switch that cleared the margin
							state.build_committed_target[unit['id']] = best_target_world
							state.build_committed_since_step[unit['id']] = state.step_counter
							state.build_committed_distance[unit['id']] = ((unit['x'] - best_target_world[0]) ** 2 + (unit['z'] - best_target_world[1]) ** 2) ** 0.5
							print(f"Unit {unit['id']} committed to building at {best_target_world} (kind={best_candidate_kind}) starting at step {state.step_counter} with distance {state.build_committed_distance[unit['id']]:.1f}")
						# else: continuing toward the committed target — leave since_step/distance intact

					# Suppress re-sending an identical build order: an immediate BU at the
					# same coords each second can cancel/restart construction game-side.
					if (action_command is not None
							and discrete_action == config.ACTION_BUILD
							and state.previous_actions.get(unit['id']) == "BUILD"
							and last_target is not None):
						dist_to_last = ((best_target_world[0] - last_target[0]) ** 2 + (best_target_world[1] - last_target[1]) ** 2) ** 0.5
						prev_pos = state.previous_positions.get(unit['id'])
						unit_moved = prev_pos is not None and ((unit['x'] - prev_pos[0]) ** 2 + (unit['z'] - prev_pos[1]) ** 2) ** 0.5 > 1.0
						order_active = (unit.get('is_constructing', 0) == 1) or unit_moved
						steps_since_send = state.previous_command_steps.get(unit['id'], 9999)
						if dist_to_last <= config.COMMAND_DISTANCE_EPS and (order_active or steps_since_send < config.BUILD_RESEND_COOLDOWN_STEPS):
							action_command = None
							print(f"Unit {unit['id']} continuing existing build order (no resend)")

					if action_command is not None:
						if last_target is not None:
							dist_to_last = (
								(unit['x'] - last_target[0]) ** 2 + (unit['z'] - last_target[1]) ** 2
							) ** 0.5
							if dist_to_last > config.COMMAND_DISTANCE_EPS and state.previous_command_steps[unit['id']] < config.MIN_STEPS_BETWEEN_COMMANDS:
								if best_target != last_target:
									state.cancel_command_penalties[unit['id']] = config.CANCEL_COMMAND_PENALTY
									print(
										f"Unit {unit['id']} cancelling in-flight command "
										f"(penalty {config.CANCEL_COMMAND_PENALTY})"
									)

						conn.sendall(action_command.encode('utf-8'))
						print(f"{unit['id']} has been sent action: {action_command.strip()}")
						state.previous_actions[unit['id']] = action
						state.previous_targets[unit['id']] = best_target_world
						state.previous_action_scores[unit['id']] = best_score
						state.previous_action_kinds[unit['id']] = best_candidate_kind
						state.previous_command_steps[unit['id']] = 0
					else:
						print(f"{unit['id']} executing NOOP (no command sent)")
				except Exception as send_err:
					print(f"[ERROR during command sending for unit {unit['id']}]")
					print(f"  Exception: {send_err}")
					traceback.print_exc()

				try:
					if state.step_counter % 20 == 0:
						local_view = map_utils.get_local_view_image(
							unit['x'],
							unit['z'],
							state.normalized_map_heights,
							view_size=256,
							unit_id=unit['id'],
							enemy_units=enemy_units,
						)
						if local_view is not None:
							local_view_norm = map_utils.normalize_image(local_view)
							state.writer.add_image(
								'Agent_View/local_heights',
								local_view_norm,
								state.step_counter,
								dataformats='HW'
							)
						enemy_img = map_utils.generate_enemy_range_image(
							enemy_units,
							state.map_width,
							state.map_height,
							state.normalized_map_heights.shape if state.normalized_map_heights is not None else None
						)
						if enemy_img is not None:
							state.writer.add_image(
								'Enemy_Ranges/map', 
								enemy_img, 
								state.step_counter, 
								dataformats='HW')

						vision_img = map_utils.generate_vision_image(
							friendly_units,
							state.map_width,
							state.map_height,
							state.normalized_map_heights
						)
						if vision_img is not None:
							state.writer.add_image(
								'Agent_View/vision',
								vision_img,
								state.step_counter,
								dataformats='HW'
							)

						if local_view is not None and enemy_img is not None:
							# use the new map utilities to create a combined visualization of local terrain and enemy ranges
							total_view_img = map_utils.get_total_map_view_image(unit['x'], unit['z'], state.normalized_map_heights, enemy_units)
							if total_view_img is not None:
								state.writer.add_image(
									'Agent_View/local_total',
									total_view_img,
									state.step_counter,
									dataformats='HWC'
								)
				except Exception as img_err:
					print(f"[ERROR during image logging for unit {unit['id']}]")
					print(f"  Exception: {img_err}")
					traceback.print_exc()

			# Log state update to console (replaces GUI event update)
			print(f"[STATE] Units: {len(state.units) if hasattr(state, 'units') and state.units else 0} | " 
					f"Enemy Units: {len(state.eUnits) if hasattr(state, 'eUnits') and state.eUnits else 0} | "
					f"Unknown Units: {len(state.eKUnits) if hasattr(state, 'eKUnits') and state.eKUnits else 0}")
		except Exception as exc:
			print("\n!!! ERROR in receive_messages !!!")
			print(f"Exception type: {type(exc).__name__}")
			print(f"Exception message: {exc}")
			print("Full traceback:")
			traceback.print_exc()
			print("!!!\n")

			if not state.match_finalized and not state.forced_terminal_success:
				print("Finalizing unit results, assuming a loss.")
				finalize_match(False, "disconnect/death")
			else:
				print("Skipping loss finalization because match outcome is already finalized.")

			local_view = map_utils.get_local_view_image(
				unit['x'],
				unit['z'],
				state.normalized_map_heights,
				view_size=256,
				unit_id=unit['id'],
				enemy_units=enemy_units,
			)
			if local_view is not None:
				local_view_norm = map_utils.normalize_image(local_view)
				state.writer.add_image(
					'Agent_View/local_heights',
					local_view_norm,
					state.step_counter,
					dataformats='HW'
				)
			enemy_img = map_utils.generate_enemy_range_image(
				enemy_units,
				state.map_width,
				state.map_height,
				state.normalized_map_heights.shape if state.normalized_map_heights is not None else None
			)
			if enemy_img is not None:
				state.writer.add_image(
					'Enemy_Ranges/map', 
					enemy_img, 
					state.step_counter, 
					dataformats='HW'
				)

			vision_img = map_utils.generate_vision_image(
				friendly_units,
				state.map_width,
				state.map_height,
				state.normalized_map_heights
			)
			if vision_img is not None:
				state.writer.add_image(
					'Agent_View/vision',
					vision_img,
					state.step_counter,
					dataformats='HWC'
				)

			if local_view is not None and enemy_img is not None:
				# use the new map utilities to create a combined visualization of local terrain and enemy ranges
				total_view_img = map_utils.get_total_map_view_image(unit['x'], unit['z'], state.normalized_map_heights, enemy_units)
				if total_view_img is not None:
					state.writer.add_image(
						'Agent_View/local_total',
						total_view_img,
						state.step_counter,
						dataformats='HWC'
					)

			# conn.sendall("C: UNPAUSE\n".encode('utf-8')) # Resume Game
			state.visited_mass_spots.clear()
			state.visited_mass_spots_norm.clear()
			state.mass_destinations.clear()
			state.mass_destination_distances.clear()
			state.mass_spot_blocked_until.clear()
			# print("All mass spots reached. Epoch ended and reset.")

			break
	print(f"[DISCONNECTED] {addr} disconnected.")
	conn.close()
