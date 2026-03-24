import time
from collections import deque

import numpy as np

import config
import map_utils
import runtime_state as state
import agent_core
import unit_defs
from Rewards import MoveJudger


def compute_reward(agent_unit, prev_health):
	"""Calculate the total step reward for a unit based on damage, mass collection, distance, inactivity, and danger."""
	reward = 0
	unit_id = agent_unit['id']

	reward_components = {
		'damage_penalty': 0,
		'mass_reward': 0,
		'distance_improvement': 0,
		'base_distance_reward': 0,
		'inactivity_penalty': 0,
		'no_mass_penalty': 0,
		#'boundary_penalty': 0,
		'danger_zone_penalty': 0,
		'cancel_command_penalty': 0
	}

	if agent_unit['health'] < prev_health:
		damage_penalty = prev_health - agent_unit['health']
		reward -= damage_penalty
		reward_components['damage_penalty'] = -damage_penalty
		print(f"Unit {unit_id} damage penalty: -{damage_penalty}")

	unit_nx = map_utils.normalize_x(agent_unit['x'])
	unit_nz = map_utils.normalize_z(agent_unit['z'])

	#mass_reward = 0
	for spot in state.mass_spots:
		if spot not in state.visited_mass_spots:
			dist_to_spot = ((spot[0] - agent_unit['x']) ** 2 + (spot[1] - agent_unit['z']) ** 2) ** 0.5
			if map_utils.normalize_distance(dist_to_spot) < map_utils.normalize_distance(200):
				state.visited_mass_spots.add(spot)
				state.visited_mass_spots_norm.add((map_utils.normalize_x(spot[0]), map_utils.normalize_z(spot[1])))
				reward += 1000
				#mass_reward = 100
				reward_components['mass_reward'] = 1000
				print(f"Unit {unit_id} mass reward: +1000")
				state.last_mass_visit[unit_id] = 0
				break

	unvisited_mass = [p for p in state.map_spots_norm if p not in state.visited_mass_spots_norm]
	if unvisited_mass:
		nearest_mass = min(unvisited_mass, key=lambda p: (p[0] - unit_nx) ** 2 + (p[1] - unit_nz) ** 2)
		current_dist = ((nearest_mass[0] - unit_nx) ** 2 + (nearest_mass[1] - unit_nz) ** 2) ** 0.5
		prev_dist = state.previous_distances.get(unit_id, current_dist)
		print(f"Unit {unit_id} nearest mass: {nearest_mass}, dist: {current_dist:.2f}, prev_dist: {prev_dist:.2f}")

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

		base_dist_reward = max(0, map_utils.normalize_distance(750) - current_dist) * 0.1
		reward += base_dist_reward
		reward_components['base_distance_reward'] = base_dist_reward
		print(f"Unit {unit_id} base distance reward: +{base_dist_reward:.2f}")

		state.previous_distances[unit_id] = current_dist

	prev_pos = state.previous_positions.get(unit_id, (agent_unit['x'], agent_unit['z']))
	dist_moved = ((prev_pos[0] - agent_unit['x']) ** 2 + (prev_pos[1] - agent_unit['z']) ** 2) ** 0.5
	if map_utils.normalize_distance(dist_moved) < map_utils.normalize_distance(5):
		state.consecutive_inactive[unit_id] = state.consecutive_inactive.get(unit_id, 0) + 3
		inactivity_penalty = state.consecutive_inactive[unit_id]
		reward -= inactivity_penalty
		reward_components['inactivity_penalty'] = -inactivity_penalty
		print(
			f"Unit {unit_id} inactivity penalty: -{inactivity_penalty} "
			f"(consecutive: {state.consecutive_inactive[unit_id]})"
		)
	else:
		state.consecutive_inactive[unit_id] = 0

	state.last_mass_visit[unit_id] = state.last_mass_visit.get(unit_id, 0) + 1
	no_mass_penalty = state.last_mass_visit[unit_id] * 0.05
	reward -= no_mass_penalty
	reward_components['no_mass_penalty'] = -no_mass_penalty
	print(
		f"Unit {unit_id} no mass visit penalty: -{no_mass_penalty:.2f} "
		f"(steps: {state.last_mass_visit[unit_id]})"
	)

    # As long as no commands are allowed outside the map, this is not required
	# boundary_threshold = map_utils.normalize_distance(200)
	# dist_to_boundary = min(
	# 	unit_nx,
	# 	unit_nz,
	# 	config.STANDARD_MAP_WIDTH - unit_nx,
	# 	config.STANDARD_MAP_HEIGHT - unit_nz
	# )

	# if dist_to_boundary < boundary_threshold:
	# 	boundary_penalty = (boundary_threshold - dist_to_boundary) * 0.2
	# 	reward -= boundary_penalty
	# 	reward_components['boundary_penalty'] = -boundary_penalty
	# 	print(
	# 		f"Unit {unit_id} boundary penalty: -{boundary_penalty:.2f} "
	# 		f"(distance to edge: {dist_to_boundary:.1f})"
	# 	)

	enemy_range_image = map_utils.generate_enemy_range_image(
		state.eUnits,
		state.map_width,
		state.map_height,
		state.normalized_map_heights.shape if state.normalized_map_heights is not None else None
	)
	if enemy_range_image is not None:
		map_x = int(map_utils.normalize_x(agent_unit['x']))
		map_z = int(map_utils.normalize_z(agent_unit['z']))
		if 0 <= map_z < enemy_range_image.shape[0] and 0 <= map_x < enemy_range_image.shape[1]:
			if enemy_range_image[map_z, map_x] > 0:
				danger_penalty = -50.0
				reward -= danger_penalty
				reward_components['danger_zone_penalty'] = -danger_penalty
				print(f"Unit {unit_id} danger zone penalty: -{danger_penalty}")

	cancel_penalty = state.cancel_command_penalties.pop(unit_id, 0.0)
	if cancel_penalty:
		reward -= cancel_penalty
		reward_components['cancel_command_penalty'] = -cancel_penalty
		print(f"Unit {unit_id} cancel command penalty: -{cancel_penalty}")

	state.previous_positions[unit_id] = (agent_unit['x'], agent_unit['z'])

	for component_name, component_value in reward_components.items():
		state.writer.add_scalar(f'Reward_Components/{component_name}', component_value, state.step_counter)
	state.writer.add_scalar('Reward/total_reward', reward, state.step_counter)

	print(f"Unit {unit_id} reward: {reward}")
	return reward


def init_segment_tracking(unit):
	"""Initialize per-unit segment tracking stats and buffer if not already present."""
	unit_id = unit['id']
	if unit_id not in state.segment_stats:
		now = time.time()
		state.segment_stats[unit_id] = {
			'start_time': now,
			'last_mass_time': now,
			'distance': 0.0,
			'height_change': 0.0,
			'damage_taken': 0.0,
			'steps': 0
		}
		state.segment_buffers[unit_id] = deque(maxlen=config.MAX_SEGMENT_STEPS)
		state.last_mass_visit[unit_id] = now


def terrain_adjusted_distance(prev_pos, curr_pos, prev_y, curr_y):
	"""Compute the movement distance between positions, adding a height-change penalty factor."""
	raw_dist = ((prev_pos[0] - curr_pos[0]) ** 2 + (prev_pos[1] - curr_pos[1]) ** 2) ** 0.5
	raw_dist = map_utils.normalize_distance(raw_dist)
	height_delta = abs(map_utils.normalize_y(curr_y) - map_utils.normalize_y(prev_y))
	return raw_dist + height_delta * config.HEIGHT_DISTANCE_FACTOR


def _compute_path_danger_penalty(prev_pos, curr_pos, enemy_range_image):
	"""Compute a penalty based on the fraction of the movement path that passes through enemy weapon range."""
	if enemy_range_image is None:
		return 0.0

	prev_nx = map_utils.normalize_x(prev_pos[0])
	prev_nz = map_utils.normalize_z(prev_pos[1])
	curr_nx = map_utils.normalize_x(curr_pos[0])
	curr_nz = map_utils.normalize_z(curr_pos[1])

	path_values = map_utils.sample_path_values(
		enemy_range_image,
		prev_nx,
		prev_nz,
		curr_nx,
		curr_nz,
		sample_count=config.PATH_SAMPLE_COUNT,
	)
	if path_values is None or path_values.size == 0:
		return 0.0

	danger_ratio = float((path_values > 0).mean())
	if danger_ratio <= 0.0:
		return 0.0

	return -(danger_ratio * config.PATH_DANGER_PENALTY_SCALE + config.PATH_DANGER_MIN_HIT_PENALTY)


def _compute_path_terrain_penalty(prev_pos, curr_pos):
	"""Compute a terrain traversal penalty along the movement path using the cost map."""
	prev_nx = map_utils.normalize_x(prev_pos[0])
	prev_nz = map_utils.normalize_z(prev_pos[1])
	curr_nx = map_utils.normalize_x(curr_pos[0])
	curr_nz = map_utils.normalize_z(curr_pos[1])

	return map_utils.estimate_path_terrain_penalty(
		prev_nx,
		prev_nz,
		curr_nx,
		curr_nz,
		sample_count=config.PATH_SAMPLE_COUNT,
	)


def compute_move_potential(prev_pos, curr_pos, prev_y, curr_y, unvisited_mass, enemy_range_image=None):
	"""Compute the potential-based shaping reward for a movement step, combining distance, direction, height, danger, terrain, and enemy avoidance."""
	unit_nx = map_utils.normalize_x(prev_pos[0])
	unit_nz = map_utils.normalize_z(prev_pos[1])
	curr_nx = map_utils.normalize_x(curr_pos[0])
	curr_nz = map_utils.normalize_z(curr_pos[1])

	move_dx = curr_nx - unit_nx
	move_dz = curr_nz - unit_nz
	move_dist = (move_dx ** 2 + move_dz ** 2) ** 0.5

	# Combine all known enemies for avoidance calculations
	all_enemies = list(state.eUnits)
	for u in state.eKUnits:
		if all(u['id'] != eu['id'] for eu in all_enemies):
			all_enemies.append(u)

	# Proactive enemy avoidance: reward for increasing distance from nearest enemy when in danger
	# Scaled by the nearest enemy's DPS so higher-threat enemies produce stronger avoidance signal
	enemy_avoidance_reward = 0.0
	if all_enemies:
		prev_enemy_info = map_utils.find_nearest_enemy(unit_nx, unit_nz, all_enemies)
		curr_enemy_info = map_utils.find_nearest_enemy(curr_nx, curr_nz, all_enemies)
		if prev_enemy_info is not None and curr_enemy_info is not None:
			prev_enemy_dist = prev_enemy_info[0]
			curr_enemy_dist = curr_enemy_info[0]
			# Only reward avoidance when the unit is within the proximity threshold
			if prev_enemy_dist < config.ENEMY_PROXIMITY_THRESHOLD:
				dist_change = curr_enemy_dist - prev_enemy_dist
				# Find DPS of the nearest enemy for threat scaling
				_, nearest_ex, nearest_ez = prev_enemy_info
				nearest_dps = 50.0  # default
				best_match_dist = float('inf')
				for eu in all_enemies:
					eu_nx = map_utils.normalize_x(eu['x'])
					eu_nz = map_utils.normalize_z(eu['z'])
					d = (eu_nx - nearest_ex) ** 2 + (eu_nz - nearest_ez) ** 2
					if d < best_match_dist:
						best_match_dist = d
						nearest_dps = eu.get('dps', 50.0)
				dps_multiplier = 1.0 + unit_defs.normalize_dps(nearest_dps) * config.DPS_THREAT_SCALE
				enemy_avoidance_reward = dist_change * config.ENEMY_AVOIDANCE_REWARD_SCALE * dps_multiplier

	if not unvisited_mass:
		path_danger_penalty = _compute_path_danger_penalty(prev_pos, curr_pos, enemy_range_image)
		path_terrain_penalty = _compute_path_terrain_penalty(prev_pos, curr_pos)
		total = path_danger_penalty + path_terrain_penalty + enemy_avoidance_reward
		return total, {
			'distance': 0.0,
			'direction': 0.0,
			'height_jump': 0.0,
			'path_danger': path_danger_penalty,
			'path_terrain': path_terrain_penalty,
			'enemy_avoidance': enemy_avoidance_reward,
		}

	nearest = MoveJudger.select_best_mass_spot(unit_nx, unit_nz, unvisited_mass)
	
	# If no reachable mass spot, treat like no mass spots
	if nearest is None:
		path_danger_penalty = _compute_path_danger_penalty(prev_pos, curr_pos, enemy_range_image)
		path_terrain_penalty = _compute_path_terrain_penalty(prev_pos, curr_pos)
		total = path_danger_penalty + path_terrain_penalty + enemy_avoidance_reward
		return total, {
			'distance': 0.0,
			'direction': 0.0,
			'height_jump': 0.0,
			'path_danger': path_danger_penalty,
			'path_terrain': path_terrain_penalty,
			'enemy_avoidance': enemy_avoidance_reward,
		}
	
	prev_dist = MoveJudger.compute_mass_spot_score(unit_nx, unit_nz, nearest)
	curr_dist = MoveJudger.compute_mass_spot_score(curr_nx, curr_nz, nearest)

	# Clamp infinite values to prevent NaN in loss computation
	MAX_DIST = 10000.0
	prev_dist = min(prev_dist, MAX_DIST) if not np.isinf(prev_dist) else MAX_DIST
	curr_dist = min(curr_dist, MAX_DIST) if not np.isinf(curr_dist) else MAX_DIST

	distance_reduction = prev_dist - curr_dist
	distance_reward = distance_reduction * config.POTENTIAL_DISTANCE_SCALE

	dir_dx = nearest[0] - unit_nx
	dir_dz = nearest[1] - unit_nz
	dir_mag = (dir_dx ** 2 + dir_dz ** 2) ** 0.5
	if move_dist > 1e-6 and dir_mag > 1e-6:
		alignment = (move_dx * dir_dx + move_dz * dir_dz) / ((move_dist * dir_mag) + 1e-6)
	else:
		alignment = 0.0
	direction_reward = alignment * move_dist * config.POTENTIAL_DIRECTION_SCALE

	height_delta = abs(map_utils.normalize_y(curr_y) - map_utils.normalize_y(prev_y))
	height_jump_penalty = -max(0.0, height_delta - config.HEIGHT_JUMP_TOLERANCE) * config.HEIGHT_JUMP_PENALTY_SCALE
	path_danger_penalty = _compute_path_danger_penalty(prev_pos, curr_pos, enemy_range_image)
	path_terrain_penalty = _compute_path_terrain_penalty(prev_pos, curr_pos)

	total = distance_reward + direction_reward + height_jump_penalty + path_danger_penalty + path_terrain_penalty + enemy_avoidance_reward
	components = {
		'distance': distance_reward,
		'direction': direction_reward,
		'height_jump': height_jump_penalty,
		'path_danger': path_danger_penalty,
		'path_terrain': path_terrain_penalty,
		'enemy_avoidance': enemy_avoidance_reward,
	}
	return total, components


def check_mass_reached(unit):
	"""Check if the unit is close enough to any unvisited mass spot to mark it as collected."""
	unit_id = unit['id']
	for spot in state.mass_spots:
		if spot not in state.visited_mass_spots:
			dist_to_spot = ((spot[0] - unit['x']) ** 2 + (spot[1] - unit['z']) ** 2) ** 0.5
			if map_utils.normalize_distance(dist_to_spot) < map_utils.normalize_distance(config.MASS_REACH_RADIUS):
				state.visited_mass_spots.add(spot)
				state.visited_mass_spots_norm.add((map_utils.normalize_x(spot[0]), map_utils.normalize_z(spot[1])))
				now = time.time()
				state.last_mass_visit[unit_id] = now
				if unit_id in state.segment_stats:
					state.segment_stats[unit_id]['last_mass_time'] = now
				return True, spot
	return False, None


def compute_segment_reward(unit_id, success, now):
	"""Compute the end-of-segment reward or penalty based on time, distance traveled, and damage sustained."""
	stats = state.segment_stats.get(unit_id, None)
	if not stats:
		return 0.0
	time_taken = max(0.01, now - stats['last_mass_time'])
	distance = stats['distance']
	damage_taken = stats['damage_taken']

	time_penalty = time_taken * config.SEGMENT_TIME_PENALTY
	distance_penalty = distance * config.SEGMENT_DISTANCE_PENALTY
	damage_penalty = damage_taken * config.SEGMENT_DAMAGE_PENALTY

	if success:
		return config.SEGMENT_BASE_REWARD - time_penalty - distance_penalty - damage_penalty
	return -config.FAILURE_BASE_PENALTY - time_penalty - distance_penalty - damage_penalty


def _train_buffer(buffer, unit_id, segment_reward):
	"""Run TD training on a list of transitions with a pre-computed segment reward."""
	per_step_bonus = segment_reward / max(1, len(buffer))
	for i, transition in enumerate(list(buffer)):
		state_full = map_utils.reconstruct_state_with_map(agent_core.agent, transition['state'])
		next_state_full = map_utils.reconstruct_state_with_map(agent_core.agent, transition['next_state'])
		total_reward = transition['potential_reward'] + per_step_bonus
		done = (i == len(buffer) - 1)
		agent_core.train_agent(
			state_full,
			transition['action'],
			total_reward,
			next_state_full,
			done,
			transition['unit_x'], transition['unit_z'], transition['unit_y'],
			transition['next_unit_x'], transition['next_unit_z'], transition['next_unit_y'],
			transition['target_x'], transition['target_z'],
			transition.get('mass_destination'),
			unit_id
		)


def finalize_segment_training(unit_id, success, reason):
	"""Train the agent on all buffered transitions for a segment, distributing the segment reward evenly across steps.
	When TRAIN_AT_EACH_MASS_POINT is False, transitions are deferred to the match buffer instead of being trained immediately."""
	buffer = state.segment_buffers.get(unit_id, [])
	if not buffer:
		return

	# Track time it takes to finish the full training loop for this segment
	start_time = time.time()

	now = time.time()
	segment_reward = compute_segment_reward(unit_id, success, now)

	if config.TRAIN_AT_EACH_MASS_POINT:
		# Immediate training mode: train on the segment now
		_train_buffer(list(buffer), unit_id, segment_reward)
	else:
		# Deferred training mode: stash transitions with their computed reward for end-of-match training
		state.match_buffer.append({
			'unit_id': unit_id,
			'transitions': list(buffer),
			'segment_reward': segment_reward,
			'success': success,
			'reason': reason,
		})
		print(f"Deferred segment for unit {unit_id} ({len(buffer)} steps, reward {segment_reward:.2f}) to match buffer. Reason: {reason}")

	state.writer.add_scalar('Segment/segment_reward', segment_reward, state.step_counter)
	state.writer.add_scalar('Segment/segment_steps', len(buffer), state.step_counter)
	state.writer.add_scalar('Segment/success', 1.0 if success else 0.0, state.step_counter)
	state.writer.add_text('Segment/reason', reason, state.step_counter)

	state.segment_buffers[unit_id] = deque(maxlen=config.MAX_SEGMENT_STEPS)
	if unit_id in state.segment_stats:
		state.segment_stats[unit_id]['distance'] = 0.0
		state.segment_stats[unit_id]['height_change'] = 0.0
		state.segment_stats[unit_id]['damage_taken'] = 0.0
		state.segment_stats[unit_id]['steps'] = 0
		state.segment_stats[unit_id]['last_mass_time'] = now

	end_time = time.time()
	state.process_times.append(end_time - start_time)
	state.pause_time = sum(state.process_times) / len(state.process_times)
	print(f"Finalized segment for unit {unit_id} with reward {segment_reward:.2f} in {end_time - start_time:.2f} seconds. Reason: {reason}")


def finalize_all_units(success, reason):
	"""Finalize segment training for every tracked unit, used when all mass spots are reached.
	When TRAIN_AT_EACH_MASS_POINT is False, this also drains the deferred match buffer."""
	# Finalize any in-progress segments (will defer if TRAIN_AT_EACH_MASS_POINT is False)
	for uid in list(state.segment_buffers.keys()):
		finalize_segment_training(uid, success, reason)

	# In deferred mode, now train on the entire accumulated match buffer
	if not config.TRAIN_AT_EACH_MASS_POINT and state.match_buffer:
		start_time = time.time()
		total_transitions = sum(len(seg['transitions']) for seg in state.match_buffer)
		print(f"[Deferred Training] Training on {len(state.match_buffer)} segments, {total_transitions} total transitions.")
		for seg in state.match_buffer:
			_train_buffer(seg['transitions'], seg['unit_id'], seg['segment_reward'])
		state.match_buffer.clear()
		end_time = time.time()
		print(f"[Deferred Training] Completed in {end_time - start_time:.2f} seconds.")
