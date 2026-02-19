import traceback
import time

import config
import runtime_state as state
import map_utils
import agent_core
from Rewards import PeriodicRewards


def parse_units(message, header):
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
					if len(parts) >= 8:
						unit = {
							'id': int(parts[0]),
							'name': parts[1],
							'x': float(parts[2]),
							'y': float(parts[3]),
							'z': float(parts[4]),
							'health': float(parts[6]),
							'speed': float(parts[7])
						}
						units_list.append(unit)
				i += 1
		i += 1
	return units_list


def format_action(action, unit_id, unit_x, unit_z, unit_y, target_x=None, target_z=None):
	if action == config.NOOP_ACTION or target_x is None or target_z is None:
		return None
	distance = ((target_x - unit_x) ** 2 + (target_z - unit_z) ** 2) ** 0.5
	command = "C: MU Q" if distance > 50 else "C: MU I"
	return f"{command} {unit_id} {target_x} {target_z} {unit_y}\n"


def receive_messages(conn, addr, window):
	while True:
		try:
			data = conn.recv(1024)
			if not data:
				break
			message = data.decode('utf-8')
			print(f"[{addr}] {message}")

			if "FRIENDLY_UNITS" in message:
				state.units = parse_units(message, "FRIENDLY_UNITS")
			if "ENEMY_UNITS" in message:
				state.eUnits = parse_units(message, "ENEMY_UNITS")
			if "KNOWN_ENEMY_UNITS" in message:
				state.eKUnits = parse_units(message, "KNOWN_ENEMY_UNITS")

			friendly_units = state.units
			enemy_units = state.eUnits

			for unit in state.eKUnits:
				if all(unit['id'] != eu['id'] for eu in enemy_units):
					enemy_units.append(unit)

			print(f"Parsed {len(friendly_units)} friendly units, {len(enemy_units)} enemy units")
			if friendly_units:
				print(f"Sample unit: {friendly_units[0]}")

			for unit in friendly_units:
				state_vec = agent_core.get_state(unit, friendly_units, enemy_units)
				state_no_map = agent_core.get_state_no_map(unit, friendly_units, enemy_units)

				if "TURN" in message:
					try:
						state.step_counter += 1
						now = time.time()
						PeriodicRewards.init_segment_tracking(unit)

						prev_health = state.previous_healths.get(unit['id'], unit['health'])
						prev_state = state.previous_states_no_map.get(unit['id'], None)
						prev_action = state.previous_actions.get(unit['id'], None)
						prev_pos = state.previous_positions.get(unit['id'], (unit['x'], unit['z']))
						prev_y = state.previous_y_positions.get(unit['id'], unit['y'])
						damage_taken = max(0.0, prev_health - unit['health'])
						enemy_range_image = map_utils.generate_enemy_range_image(
							enemy_units,
							state.map_width,
							state.map_height,
							state.normalized_map_heights.shape if state.normalized_map_heights is not None else None
						)

						if prev_state is not None and prev_action is not None:
							unvisited_mass = [
								p for p in state.map_spots_norm
								if p not in state.visited_mass_spots_norm
							]
							potential_reward, components = PeriodicRewards.compute_move_potential(
								prev_pos,
								(unit['x'], unit['z']),
								prev_y,
								unit['y'],
								unvisited_mass,
								enemy_range_image=enemy_range_image,
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

							state.segment_buffers[unit['id']].append({
								'state': prev_state,
								'action': prev_action,
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
								'potential_reward': potential_reward
							})

							state.writer.add_scalar('Move_Potential/total', potential_reward, state.step_counter)
							state.writer.add_scalar('Move_Potential/distance', components['distance'], state.step_counter)
							state.writer.add_scalar('Move_Potential/direction', components['direction'], state.step_counter)
							state.writer.add_scalar('Move_Potential/height_jump', components['height_jump'], state.step_counter)
							state.writer.add_scalar('Move_Potential/path_danger', components.get('path_danger', 0.0), state.step_counter)
							state.writer.add_scalar('Move_Potential/path_terrain', components.get('path_terrain', 0.0), state.step_counter)
							state.writer.add_scalar('Move_Potential/damage_taken', components.get('damage_taken', 0.0), state.step_counter)
						elif unit['id'] not in state.previous_actions:
							print(
								f"[INFO] Skipping move potential for unit {unit['id']} "
								"(no previous action)."
							)

						reached_mass, _ = PeriodicRewards.check_mass_reached(unit)
						if reached_mass:
							state.mass_destinations.pop(unit['id'], None)
							state.mass_destination_distances.pop(unit['id'], None)
							PeriodicRewards.finalize_segment_training(unit['id'], True, "mass_reached")
							print(f"Unit {unit['id']} reached a mass spot. Segment trained.")

						if state.mass_spots and len(state.visited_mass_spots) == len(state.mass_spots):
							PeriodicRewards.finalize_all_units(True, "all_mass_reached")
							state.visited_mass_spots.clear()
							state.visited_mass_spots_norm.clear()
							state.mass_destinations.clear()
							state.mass_destination_distances.clear()
							print("All mass spots reached. Epoch ended and reset.")
						else:
							last_time = state.segment_stats[unit['id']]['last_mass_time']
							if now - last_time >= config.EPISODE_TIMEOUT_SECONDS:
								PeriodicRewards.finalize_segment_training(unit['id'], False, "timeout")
								print(
									f"Unit {unit['id']} timed out without reaching a mass spot. "
									"Segment punished."
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

					action, best_target, best_score = agent_core.get_action(
						state_vec,
						unit['x'],
						unit['z'],
						unit['y'],
						unit['id']
					)

					denorm_tx = map_utils.denormalize_x(best_target[0])
					denorm_tz = map_utils.denormalize_z(best_target[1])
					print(
						f"Unit {unit['id']} chose action {action} -> target normalized "
						f"({best_target[0]:.2f}, {best_target[1]:.2f}), denormalized "
						f"({denorm_tx:.1f}, {denorm_tz:.1f}) (score {best_score:.2f})"
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

					action_command = format_action(
						action,
						unit['id'],
						unit['x'],
						unit['z'],
						unit['y'],
						best_target_world[0],
						best_target_world[1]
					)
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
							view_size=256
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
							state.writer.add_image('Enemy_Ranges/map', enemy_img, state.step_counter, dataformats='HW')
				except Exception as img_err:
					print(f"[ERROR during image logging for unit {unit['id']}]")
					print(f"  Exception: {img_err}")
					traceback.print_exc()

			window.write_event_value('-SOCKET-', message)
		except Exception as exc:
			print("\n!!! ERROR in receive_messages !!!")
			print(f"Exception type: {type(exc).__name__}")
			print(f"Exception message: {exc}")
			print("Full traceback:")
			traceback.print_exc()
			print("!!!\n")
			break
	print(f"[DISCONNECTED] {addr} disconnected.")
	conn.close()
