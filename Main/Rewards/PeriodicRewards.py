import time
import json
import random
import datetime
from collections import deque
import re
from pathlib import Path

import numpy as np

import config
import map_utils
import runtime_state as state
import agent_core
import unit_defs
from Rewards import MoveJudger


def _sanitize_units_for_json(units):
	"""Return a JSON-safe shallow copy of unit dicts, converting tuple values to lists."""
	clean_units = []
	for unit in units:
		clean = {}
		for key, value in unit.items():
			if isinstance(value, tuple):
				clean[key] = list(value)
			else:
				clean[key] = value
		clean_units.append(clean)
	return clean_units


def record_match_sample(unit_id, friendly_units, enemy_units, action_type, target_x, target_y, target_z, action_score, cmd_id=None):
	"""Record one live decision in replay-compatible format for optional top-match export."""
	if not getattr(config, 'SAVE_TOP_MATCH_DATASET', False):
		return

	action_payload = {
		'type': action_type,
		'x': float(target_x),
		'y': float(target_y),
		'z': float(target_z),
		'score': float(action_score),
	}
	if cmd_id is not None:
		action_payload['cmd_id'] = int(cmd_id)

	state.current_match_samples.append({
		'timestamp': float(time.time()),
		'step': int(state.step_counter),
		'unit_id': int(unit_id),
		'friendly_units': _sanitize_units_for_json(friendly_units),
		'enemy_units': _sanitize_units_for_json(enemy_units),
		'action': action_payload,
	})


def _compute_match_score():
	"""Compute a scalar match score from finalized segment rewards."""
	if not state.match_segment_summaries:
		return 0.0
	return float(sum(seg['segment_reward'] for seg in state.match_segment_summaries))


def _run_log_prefix():
	"""Return the TensorBoard prefix for the current run mode."""
	return 'Eval' if getattr(state, 'evalRun', False) else 'Run'


def _filtered_samples_for_export(samples, seed_value):
	"""Downsample NOOP actions to align online exports with manual replay class balance."""
	noop_ratio = max(0.0, min(1.0, float(getattr(config, 'AGENT_REPLAY_NOOP_KEEP_RATIO', 0.2))))
	rng = random.Random(seed_value)
	filtered = []
	for sample in samples:
		action_type = sample.get('action', {}).get('type', config.NOOP_ACTION)
		if action_type == config.NOOP_ACTION and rng.random() > noop_ratio:
			continue
		filtered.append(sample)
	return filtered


def _sanitize_map_slug(map_name):
	"""Convert a map label into a filesystem-safe slug."""
	slug = re.sub(r'[^A-Za-z0-9]+', '_', str(map_name).strip()).strip('_').lower()
	return slug or 'unknown_map'


def _get_current_map_name():
	"""Resolve the current map name from runtime state or the parsed map metadata file."""
	map_name = str(getattr(state, 'map_name', '')).strip()
	if map_name:
		return map_name

	map_info_source = str(getattr(state, 'map_info_source', '')).strip()
	if map_info_source:
		map_info_path = Path(map_info_source)
		if map_info_path.exists():
			try:
				with map_info_path.open('r', encoding='utf-8', errors='ignore') as f:
					for line in f:
						stripped = line.strip()
						if not stripped or ':' not in stripped:
							continue
						label, value = stripped.split(':', 1)
						label = label.strip().lower()
						value = value.strip()
						if label in {'map name', 'mapname', 'mission name', 'missionname', 'name'} and value:
							return value
			except Exception:
				pass

	map_heights_source = str(getattr(state, 'map_heights_source', '')).strip()
	if map_heights_source:
		map_signature_fn = getattr(map_utils, '_get_map_signature', None)
		if callable(map_signature_fn):
			map_signature = map_signature_fn()
			if map_signature:
				return map_signature
		return Path(map_heights_source).stem

	return 'unknown_map'


def _get_map_export_paths(main_dir):
	"""Return the per-map export directory and index path."""
	map_name = _get_current_map_name()
	map_slug = _sanitize_map_slug(map_name)
	export_root = main_dir / getattr(config, 'AGENT_REPLAY_EXPORT_DIR', 'Recordings/AgentReplayTop')
	map_export_dir = export_root / map_slug
	index_path = map_export_dir / 'top_matches_index.json'
	return map_name, map_slug, map_export_dir, index_path


def _save_top_match_dataset(success, reason):
	"""Save this match in replay JSON format and keep only the top-K matches by score."""
	if not getattr(config, 'SAVE_TOP_MATCH_DATASET', False):
		state.current_match_samples.clear()
		state.match_segment_summaries.clear()
		return

	if not state.current_match_samples:
		state.match_segment_summaries.clear()
		return

	main_dir = Path(__file__).resolve().parents[1]
	map_name, map_slug, export_dir, index_path = _get_map_export_paths(main_dir)
	export_dir.mkdir(parents=True, exist_ok=True)

	match_score = _compute_match_score()
	now = datetime.datetime.now()
	timestamp = now.strftime('%Y%m%d_%H%M%S')
	seed_value = f"{timestamp}_{match_score:.4f}_{len(state.current_match_samples)}"

	samples_to_save = _filtered_samples_for_export(state.current_match_samples, seed_value)
	if not samples_to_save:
		samples_to_save = list(state.current_match_samples)

	score_token = f"{match_score:.2f}".replace('-', 'm').replace('.', 'p')
	match_filename = f"agent_match_{timestamp}_score_{score_token}.json"
	match_path = export_dir / match_filename

	dataset = {
		'metadata': {
			'source': 'online_agent_export',
			'exported_at': now.isoformat(),
			'map_name': map_name,
			'map_slug': map_slug,
			'match_score': match_score,
			'success': bool(success),
			'reason': reason,
			'total_segments': len(state.match_segment_summaries),
			'total_samples': len(samples_to_save),
			'map_width': int(state.map_width),
			'map_height': int(state.map_height),
			'map_heights_file': state.map_heights_source,
			'map_spots_file': state.map_spots_source,
			'runtime_file': 'online_agent_runtime',
		},
		'samples': samples_to_save,
	}

	with match_path.open('w', encoding='utf-8') as f:
		json.dump(dataset, f, indent=2)

	max_keep = max(1, int(getattr(config, 'TOP_MATCHES_TO_KEEP', 15)))
	index_entries = []
	if index_path.exists():
		try:
			with index_path.open('r', encoding='utf-8') as f:
				loaded = json.load(f)
				if isinstance(loaded, dict):
					index_entries = loaded.get('top_matches', [])
				elif isinstance(loaded, list):
					index_entries = loaded
		except Exception:
			index_entries = []

	new_entry = {
		'file': match_filename,
		'map_name': map_name,
		'map_slug': map_slug,
		'match_score': match_score,
		'success': bool(success),
		'reason': reason,
		'samples': len(samples_to_save),
		'segments': len(state.match_segment_summaries),
		'exported_at': now.isoformat(),
	}
	index_entries.append(new_entry)
	index_entries.sort(key=lambda item: float(item.get('match_score', 0.0)), reverse=True)
	kept_entries = index_entries[:max_keep]
	kept_files = {entry.get('file') for entry in kept_entries if entry.get('file')}

	if match_filename not in kept_files:
		if match_path.exists():
			match_path.unlink()
		print(
			f"[Replay Export] Match score {match_score:.2f} not in top {max_keep} for {map_name}. "
			f"Discarded {match_filename}."
		)
	else:
		print(
			f"[Replay Export] Saved top-match dataset {match_filename} for {map_name} "
			f"with {len(samples_to_save)} samples (score={match_score:.2f})."
		)

	# Remove stale files not present in the top list.
	for entry in index_entries[max_keep:]:
		stale_name = entry.get('file')
		if not stale_name:
			continue
		stale_path = export_dir / stale_name
		if stale_path.exists():
			stale_path.unlink()

	with index_path.open('w', encoding='utf-8') as f:
		json.dump({'top_matches': kept_entries}, f, indent=2)

	state.current_match_samples.clear()
	state.match_segment_summaries.clear()


# def compute_reward(agent_unit, prev_health):
# 	"""Calculate the total step reward for a unit based on damage, mass collection, distance, inactivity, and danger."""
# 	reward = 0
# 	unit_id = agent_unit['id']

# 	reward_components = {
# 		'damage_penalty': 0,
# 		'mass_reward': 0,
# 		'distance_improvement': 0,
# 		'base_distance_reward': 0,
# 		'inactivity_penalty': 0,
# 		'no_mass_penalty': 0,
# 		#'boundary_penalty': 0,
# 		'danger_zone_penalty': 0,
# 		'cancel_command_penalty': 0
# 	}

# 	if agent_unit['health'] < prev_health:
# 		damage_penalty = prev_health - agent_unit['health']
# 		reward -= damage_penalty
# 		reward_components['damage_penalty'] = -damage_penalty
# 		print(f"Unit {unit_id} damage penalty: -{damage_penalty}")

# 	unit_nx = map_utils.normalize_x(agent_unit['x'])
# 	unit_nz = map_utils.normalize_z(agent_unit['z'])

# 	#mass_reward = 0
# 	for spot in state.mass_spots:
# 		if spot not in state.visited_mass_spots:
# 			dist_to_spot = ((spot[0] - agent_unit['x']) ** 2 + (spot[1] - agent_unit['z']) ** 2) ** 0.5
# 			if map_utils.normalize_distance(dist_to_spot) < map_utils.normalize_distance(200):
# 				state.visited_mass_spots.add(spot)
# 				state.visited_mass_spots_norm.add((map_utils.normalize_x(spot[0]), map_utils.normalize_z(spot[1])))
# 				reward += 100
# 				#mass_reward = 100
# 				reward_components['mass_reward'] = 100
# 				print(f"Unit {unit_id} mass reward: +100")
# 				state.last_mass_visit[unit_id] = 0
# 				break

# 	unvisited_mass = [p for p in state.map_spots_norm if (p[0], p[1]) not in state.visited_mass_spots_norm]
# 	if unvisited_mass:
# 		nearest_mass = min(unvisited_mass, key=lambda p: (p[0] - unit_nx) ** 2 + (p[1] - unit_nz) ** 2)
# 		current_dist = ((nearest_mass[0] - unit_nx) ** 2 + (nearest_mass[1] - unit_nz) ** 2) ** 0.5
# 		prev_dist = state.previous_distances.get(unit_id, current_dist)
# 		print(f"Unit {unit_id} nearest mass: {nearest_mass}, dist: {current_dist:.2f}, prev_dist: {prev_dist:.2f}")

# 		if current_dist < prev_dist:
# 			dist_improvement = (prev_dist - current_dist) * 0.1
# 			reward += dist_improvement
# 			reward_components['distance_improvement'] = dist_improvement
# 			print(f"Unit {unit_id} distance improvement: +{dist_improvement:.2f}")
# 		elif current_dist > prev_dist:
# 			away_penalty = (current_dist - prev_dist) * 0.05
# 			reward -= away_penalty
# 			reward_components['distance_improvement'] = -away_penalty
# 			print(f"Unit {unit_id} moving away penalty: -{away_penalty:.2f}")

# 		base_dist_reward = max(0, map_utils.normalize_distance(750) - current_dist) * 0.1
# 		reward += base_dist_reward
# 		reward_components['base_distance_reward'] = base_dist_reward
# 		print(f"Unit {unit_id} base distance reward: +{base_dist_reward:.2f}")

# 		state.previous_distances[unit_id] = current_dist

# 	prev_pos = state.previous_positions.get(unit_id, (agent_unit['x'], agent_unit['z']))
# 	dist_moved = ((prev_pos[0] - agent_unit['x']) ** 2 + (prev_pos[1] - agent_unit['z']) ** 2) ** 0.5
# 	if map_utils.normalize_distance(dist_moved) < map_utils.normalize_distance(5):
# 		state.consecutive_inactive[unit_id] = state.consecutive_inactive.get(unit_id, 0) + 3
# 		inactivity_penalty = state.consecutive_inactive[unit_id]
# 		reward -= inactivity_penalty
# 		reward_components['inactivity_penalty'] = -inactivity_penalty
# 		print(
# 			f"Unit {unit_id} inactivity penalty: -{inactivity_penalty} "
# 			f"(consecutive: {state.consecutive_inactive[unit_id]})"
# 		)
# 	else:
# 		state.consecutive_inactive[unit_id] = 0

# 	state.last_mass_visit[unit_id] = state.last_mass_visit.get(unit_id, 0) + 1
# 	no_mass_penalty = state.last_mass_visit[unit_id] * 0.05
# 	reward -= no_mass_penalty
# 	reward_components['no_mass_penalty'] = -no_mass_penalty
# 	print(
# 		f"Unit {unit_id} no mass visit penalty: -{no_mass_penalty:.2f} "
# 		f"(steps: {state.last_mass_visit[unit_id]})"
# 	)

#     # As long as no commands are allowed outside the map, this is not required
# 	# boundary_threshold = map_utils.normalize_distance(200)
# 	# dist_to_boundary = min(
# 	# 	unit_nx,
# 	# 	unit_nz,
# 	# 	config.STANDARD_MAP_WIDTH - unit_nx,
# 	# 	config.STANDARD_MAP_HEIGHT - unit_nz
# 	# )

# 	# if dist_to_boundary < boundary_threshold:
# 	# 	boundary_penalty = (boundary_threshold - dist_to_boundary) * 0.2
# 	# 	reward -= boundary_penalty
# 	# 	reward_components['boundary_penalty'] = -boundary_penalty
# 	# 	print(
# 	# 		f"Unit {unit_id} boundary penalty: -{boundary_penalty:.2f} "
# 	# 		f"(distance to edge: {dist_to_boundary:.1f})"
# 	# 	)

# 	enemy_range_image = map_utils.generate_enemy_range_image(
# 		state.eUnits,
# 		state.map_width,
# 		state.map_height,
# 		state.normalized_map_heights.shape if state.normalized_map_heights is not None else None
# 	)
# 	if enemy_range_image is not None:
# 		map_x = int(map_utils.normalize_x(agent_unit['x']))
# 		map_z = int(map_utils.normalize_z(agent_unit['z']))
# 		if 0 <= map_z < enemy_range_image.shape[0] and 0 <= map_x < enemy_range_image.shape[1]:
# 			if enemy_range_image[map_z, map_x] > 0:
# 				danger_penalty = -50.0
# 				reward -= danger_penalty
# 				reward_components['danger_zone_penalty'] = -danger_penalty
# 				print(f"Unit {unit_id} danger zone penalty: -{danger_penalty}")

# 	cancel_penalty = state.cancel_command_penalties.pop(unit_id, 0.0)
# 	if cancel_penalty:
# 		reward -= cancel_penalty
# 		reward_components['cancel_command_penalty'] = -cancel_penalty
# 		print(f"Unit {unit_id} cancel command penalty: -{cancel_penalty}")

# 	state.previous_positions[unit_id] = (agent_unit['x'], agent_unit['z'])

# 	for component_name, component_value in reward_components.items():
# 		state.writer.add_scalar(f'Reward_Components/{component_name}', component_value, state.step_counter)
# 	state.writer.add_scalar('Reward/total_reward', reward, state.step_counter)

# 	print(f"Unit {unit_id} reward: {reward}")
# 	return reward


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
			'steps': 0,
			'buildings_built': 0,
			'value_from_building': 0.0,
			'build_steps': 0
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


def compute_move_potential(prev_pos, curr_pos, prev_y, curr_y, unvisited_mass, enemy_range_image=None, vision_image=None, unit_id=None):
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
				nearest_wtype = 'projectile' # default
				best_match_dist = float('inf')
				for eu in all_enemies:
					eu_nx = map_utils.normalize_x(eu['x'])
					eu_nz = map_utils.normalize_z(eu['z'])
					d = (eu_nx - nearest_ex) ** 2 + (eu_nz - nearest_ez) ** 2
					if d < best_match_dist:
						best_match_dist = d
						nearest_dps = eu.get('dps', 50.0)
						nearest_wtype = eu.get('weapon_type', 'projectile')
				dps_multiplier = 1.0 + unit_defs.normalize_dps(nearest_dps) * config.DPS_THREAT_SCALE
				# Range-keeping is the correct evasion vs instant-hit weapons; vs travel-time
				# weapons, lateral dodging should compete, so radial retreat pays less.
				type_multiplier = 1.0 if nearest_wtype in ('hitscan', 'beam') else 0.4
				enemy_avoidance_reward = dist_change * config.ENEMY_AVOIDANCE_REWARD_SCALE * dps_multiplier * type_multiplier

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

	# Building reward parameters
	
	# Compare vision delta between rewards (ie, if a building was finished and it can spot a lot, reward it)
	currentVisionScore = np.sum(vision_image) if vision_image is not None else 0.0
	# Grab the past few (to get 5 seconds of it) vision scores and average them to get a more stable "prior" vision score, then compare to current
	# Use a frozen pre-build baseline during construction so the full
	# vision improvement is credited across the entire build period,
	# not just the single step the radar comes online.

	pre_build_baseline = state.previous_build_vision_baseline

	if pre_build_baseline != 0 :
		# During active construction: compare against the moment building began
		vision_delta = currentVisionScore - pre_build_baseline
	else:
		# Normal movement: compare against rolling 10-step average
		priorVisionScore = 0.0
		vision_scores = state.previous_vision_scores[-10:]
		if vision_scores:
			priorVisionScore = np.mean(vision_scores)
		vision_delta = currentVisionScore - priorVisionScore

	vision_reward = max(0.0, vision_delta) * config.VISION_REWARD_SCALE

	total = distance_reward + direction_reward + height_jump_penalty + path_danger_penalty + path_terrain_penalty + enemy_avoidance_reward + vision_reward
	components = {
		'distance': distance_reward,
		'direction': direction_reward,
		'height_jump': height_jump_penalty,
		'path_danger': path_danger_penalty,
		'path_terrain': path_terrain_penalty,
		'enemy_avoidance': enemy_avoidance_reward,
		'vision_coverage': vision_reward,
	}
	return total, components

def compute_build_potential(prev_pos, curr_pos, prev_y, curr_y,
                             enemy_range_image=None, vision_image=None, unit=None):
    """Compute the potential-based shaping reward for a build action step.
    Rewards construction progress, vision expansion, and safe placement.
    Does not penalize standing still — that is intentional during builds.
    """
    unit_id = unit['id']
    is_constructing = int(unit.get('is_constructing', 0))
    build_progress = float(unit.get('active_build_progress', 0.0))

    components = {
        'build_progress': 0.0,
        'build_completion': 0.0,
		'build_approach': 0.0,
        'vision_coverage': 0.0,
        'path_danger': 0.0,
        'distance': 0.0,
        'direction': 0.0,
        'height_jump': 0.0,
        'enemy_avoidance': 0.0,
    }

    total = 0.0

    # --- Build progress reward ---
    # Reward each step of active construction proportional to progress made.
    # This gives the agent a dense signal during the 20-100 steps construction takes.
    if is_constructing == 1 and build_progress > 0.0:
        prev_progress = state.previous_build_progress.get(unit_id, 0.0)
        progress_delta = max(0.0, build_progress - prev_progress)
        build_progress_reward = progress_delta * config.BUILD_PROGRESS_REWARD_SCALE
        total += build_progress_reward
        components['build_progress'] = build_progress_reward

	# --- A5: approach shaping toward the committed build site ---
    # Mirrors the move head's distance potential so a build decision starts
    # paying immediately during transit, not only once construction begins.
    committed = state.build_committed_target.get(unit_id)
    if committed is not None and is_constructing == 0:
        prev_d = ((prev_pos[0] - committed[0]) ** 2 + (prev_pos[1] - committed[1]) ** 2) ** 0.5
        curr_d = ((curr_pos[0] - committed[0]) ** 2 + (curr_pos[1] - committed[1]) ** 2) ** 0.5
        approach_reward = map_utils.normalize_distance(prev_d - curr_d) * config.POTENTIAL_DISTANCE_SCALE
        total += approach_reward
        components['build_approach'] = approach_reward

    # --- Vision coverage reward (sustained, using frozen baseline) ---
    if vision_image is not None:
        currentVisionScore = float(np.sum(vision_image > 0))
        pre_build_baseline = state.previous_build_vision_baseline
        if pre_build_baseline != 0:
            vision_delta = max(0.0, currentVisionScore - pre_build_baseline)
        else:
            prior_scores = state.previous_vision_scores[-10:]
            vision_delta = max(0.0, currentVisionScore - (np.mean(prior_scores) if prior_scores else 0.0))
        vision_reward = vision_delta * config.VISION_REWARD_SCALE
        total += vision_reward
        components['vision_coverage'] = vision_reward

    # --- Danger penalty (same as move) ---
    # Unit should not be standing in enemy range while building.
    unit_nx = map_utils.normalize_x(curr_pos[0])
    unit_nz = map_utils.normalize_z(curr_pos[1])
    if enemy_range_image is not None:
        map_x = int(np.clip(unit_nx, 0, enemy_range_image.shape[1] - 1))
        map_z = int(np.clip(unit_nz, 0, enemy_range_image.shape[0] - 1))
        danger = float(enemy_range_image[map_z, map_x])
        if danger > 0:
            danger_penalty = -danger * config.PATH_DANGER_PENALTY_SCALE
            total += danger_penalty
            components['path_danger'] = danger_penalty

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
	build_steps = stats['build_steps']
	build_value = stats['value_from_building']
	buildings_built = stats['buildings_built']
	total_steps = max(1, stats['steps'])

	build_fraction = min(1.0, build_steps / total_steps) if total_steps > 0 else 0.0
	penalty_scale = 1.0 - (build_fraction * config.BUILDING_REWARD_SCALE)

	time_penalty = time_taken * config.SEGMENT_TIME_PENALTY * penalty_scale
	distance_penalty = distance * config.SEGMENT_DISTANCE_PENALTY * penalty_scale
	damage_penalty = damage_taken * config.SEGMENT_DAMAGE_PENALTY

	build_reward = build_value / max(1, buildings_built) * config.VISION_REWARD_SCALE

	if success:
		return config.SEGMENT_BASE_REWARD - time_penalty - distance_penalty - damage_penalty + build_reward
	return -config.FAILURE_BASE_PENALTY - time_penalty - distance_penalty - damage_penalty + build_reward


def _train_buffer(buffer, unit_id, segment_reward):
	"""Train on a segment using discounted Monte-Carlo returns (no bootstrapping)."""
	per_step_bonus = segment_reward / max(1, len(buffer))
	transitions = list(buffer)

	# Backward pass: G_t = r_t + gamma * G_{t+1}
	G = 0.0
	returns = [0.0] * len(transitions)
	for i in range(len(transitions) - 1, -1, -1):
		r = transitions[i]['potential_reward'] + per_step_bonus
		G = r + 0.95 * G
		returns[i] = G

	for i, transition in enumerate(transitions):
		state_full = map_utils.reconstruct_state_with_map(agent_core.agent, transition['state'])
		agent_core.train_agent(
			state_full,
			transition.get('discrete_action', config.ACTION_MOVE),
			transition['action'],
			returns[i],
			unit_id=unit_id,
			forced_build=transition.get('forced_build', False),
			decision_snapshot=transition['decision_snapshot'],
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
	state.match_segment_summaries.append({
		'unit_id': unit_id,
		'segment_reward': float(segment_reward),
		'success': bool(success),
		'reason': reason,
		'steps': len(buffer),
	})

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
		state.segment_stats[unit_id]['buildings_built'] = 0
		state.segment_stats[unit_id]['value_from_building'] = 0.0
		state.segment_stats[unit_id]['build_steps'] = 0

	end_time = time.time()
	state.process_times.append(end_time - start_time)
	state.pause_time = sum(state.process_times) / len(state.process_times)
	print(f"Finalized segment for unit {unit_id} with reward {segment_reward:.2f} in {end_time - start_time:.2f} seconds. Reason: {reason}")


def finalize_all_units(success, reason):
	"""Finalize segment training for every tracked unit, used when all mass spots are reached.
	When TRAIN_AT_EACH_MASS_POINT is False, this also drains the deferred match buffer."""

	# Check sentinel file for survival time (a number)
	# This can override success/failure based on whether the agent survived for a certain duration, even if not all mass spots were reached.
	# However if there is a low or no number, use the original success/failure signal to avoid false negatives.
	survival_time = None
	sentinel_path = Path(state.sentinel_time_path)
	if sentinel_path.exists():
		try:
			with sentinel_path.open('r') as f:
				content = f.read().strip()
				survival_time = float(content)
				print(f"Read survival time from sentinel: {survival_time:.2f} minutes")
		except Exception as e:
			print(f"Error reading survival time sentinel: {e}")
			survival_time = None

	if survival_time is not None:
		if survival_time >= config.SURVIVAL_TIME_THRESHOLD:
			success = True
			reason = f"Survived for {survival_time:.2f} minutes (threshold {config.SURVIVAL_TIME_THRESHOLD}s)"
		else:
			# No need to set it to failure here, it already is, but we can update the reason to reflect the survival time outcome.
			# Add survival time onto the existing reason for more context in the logs and replay exports.
			reason = reason + f" |> Survival time {survival_time:.2f}s below threshold {config.SURVIVAL_TIME_THRESHOLD}s"

		print(f"Overriding match outcome based on survival time: success={success}, reason='{reason}'")

	# Finalize any in-progress segments (will defer if TRAIN_AT_EACH_MASS_POINT is False)
	for uid in list(state.segment_buffers.keys()):
		finalize_segment_training(uid, success, reason)

	# In deferred mode, now train on the entire accumulated match buffer
	if not getattr(state, 'evalRun', False) and not config.TRAIN_AT_EACH_MASS_POINT and state.match_buffer:
		start_time = time.time()
		total_transitions = sum(len(seg['transitions']) for seg in state.match_buffer)
		print(f"[Deferred Training] Training on {len(state.match_buffer)} segments, {total_transitions} total transitions.")
		for seg in state.match_buffer:
			_train_buffer(seg['transitions'], seg['unit_id'], seg['segment_reward'])
		state.match_buffer.clear()
		end_time = time.time()
		print(f"[Deferred Training] Completed in {end_time - start_time:.2f} seconds.")
	elif getattr(state, 'evalRun', False):
		state.match_buffer.clear()

	# Log explicit run-level summary metrics for cross-run analysis.
	final_match_score = _compute_match_score()
	runtime_seconds = max(0.0, float(time.time() - getattr(state, 'run_started_at', time.time())))
	# Expose the most recent run metrics on the global state so the runner
	# (Socket ML) can make decisions such as best-checkpoint promotion.
	setattr(state, 'last_run_final_match_score', float(final_match_score))
	setattr(state, 'last_run_runtime_seconds', float(runtime_seconds))
	prefix = _run_log_prefix()
	state.writer.add_scalar(f'{prefix}/score', final_match_score, state.step_counter)
	state.writer.add_scalar(f'{prefix}/final_match_score', final_match_score, state.step_counter)
	state.writer.add_scalar(f'{prefix}/runtime_seconds', runtime_seconds, state.step_counter)
	state.writer.add_scalar(f'{prefix}/success', 1.0 if success else 0.0, state.step_counter)

	_save_top_match_dataset(success, reason)


def finalize_cycle_segments(success, reason):
	"""Finalize active segment buffers for a cycle boundary without ending the full match."""
	for uid in list(state.segment_buffers.keys()):
		finalize_segment_training(uid, success, reason)
