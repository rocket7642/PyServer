import config
import map_utils
import numpy as np


def compute_mass_spot_score(unit_x, unit_z, spot):
	"""Compute terrain-aware cost to reach a mass spot using precomputed cost field."""
	return map_utils.get_mass_cost_at(spot, unit_x, unit_z)


def select_best_mass_spot(unit_x, unit_z, unvisited_mass, return_score=False):
	if not unvisited_mass:
		return (None, None) if return_score else None
	
	best_score = None
	best_spot = None

	for spot in unvisited_mass:
		score = compute_mass_spot_score(unit_x, unit_z, spot)
		# Skip unreachable mass spots
		if np.isinf(score):
			continue
		if best_score is None or score < best_score:
			best_score = score
			best_spot = spot

	if return_score:
		return best_spot, best_score
	return best_spot


def get_top_mass_spots(unit_x, unit_z, unvisited_mass, limit=4):
	if not unvisited_mass:
		return []

	ranked = []
	for spot in unvisited_mass:
		score = compute_mass_spot_score(unit_x, unit_z, spot)
		# Skip unreachable mass spots
		if not np.isinf(score):
			ranked.append((spot, score))

	ranked.sort(key=lambda item: item[1])
	return ranked[:limit]


def compute_action_features(action, unit_x, unit_z, unit_y, unvisited_mass, target_x=None, target_z=None, enemy_range_image=None):
	features = []

	if action == config.NOOP_ACTION:
		features.append(0.0)
		#features.append(0.0)
		features.append(0.0)
		#features.append(0.0)
		features.append(1.0)
		#features.append(0.0)
		features.append(0.0)
		return features

	if target_x is None or target_z is None:
		target_x = unit_x
		target_z = unit_z

	# unvisted mass reduction which is the prime goal
	best_spot = select_best_mass_spot(unit_x, unit_z, unvisited_mass)
	if best_spot is not None:
		current_cost = compute_mass_spot_score(unit_x, unit_z, best_spot)
		target_cost = compute_mass_spot_score(target_x, target_z, best_spot)
		# Handle unreachable paths - assign large negative penalty to discourage unreachable moves
		if np.isinf(current_cost) or np.isinf(target_cost):
			dist_reduction = -1000.0
		else:
			dist_reduction = current_cost - target_cost
		features.append(dist_reduction)
	else:
		features.append(0.0)

	# avoid boundaries which is not needed as long as moves are not allowed out of bounds
	# dist_to_boundary = min(
	# 	target_x,
	# 	target_z,
	# 	config.STANDARD_MAP_WIDTH - target_x,
	# 	config.STANDARD_MAP_HEIGHT - target_z
	# )
	# boundary_feature = -max(0, 100 - dist_to_boundary)
	# features.append(boundary_feature)

	# perfer closer targets
	move_magnitude = ((target_x - unit_x) ** 2 + (target_z - unit_z) ** 2) ** 0.5
	terrain_path_penalty = map_utils.estimate_path_terrain_penalty(
		unit_x,
		unit_z,
		target_x,
		target_z,
		sample_count=config.PATH_SAMPLE_COUNT,
	)
	magnitude_feature = -move_magnitude / 100.0 + terrain_path_penalty
	features.append(magnitude_feature)

	# Remove terrain penalty for now since heights don't matter, only path up them, which this does not convey
	# try:
	# 	target_height = map_utils.height_at_normalized(target_x, target_z)
	# 	current_height = map_utils.height_at_normalized(unit_x, unit_z)
	# 	height_diff = abs(target_height - current_height)
	# 	terrain_penalty = -height_diff * 0.05
	# 	terrain_penalty = max(terrain_penalty, -5.0)
	# 	terrain_penalty = min(terrain_penalty, 0.0)
	# 	terrain_penalty = -(height_diff / max(move_magnitude, 1.0)) * 0.1
	# except (IndexError, TypeError):
	# 	terrain_penalty = -100.0
	# features.append(terrain_penalty)

	features.append(0.0)

	# Remove terrain penalty for now since heights don't matter, only path up them, which this does not convey
	# try:
	# 	target_height = map_utils.height_at_normalized(target_x, target_z)
	# 	height_change = target_height - unit_y
	# 	features.append(height_change)
	# except (IndexError, TypeError):
	# 	features.append(0.0)

	# Danger feature based on enemy range image, might be worth moving to a seperate weight system
	# as to allow the agent to train seperate actions correctly
	if enemy_range_image is not None and target_x is not None and target_z is not None:
		path_values = map_utils.sample_path_values(
			enemy_range_image,
			unit_x,
			unit_z,
			target_x,
			target_z,
			sample_count=config.PATH_SAMPLE_COUNT,
		)
		if path_values is not None and path_values.size > 0:
			danger_ratio = float((path_values > 0).mean())
			features.append(-danger_ratio)
		else:
			features.append(0.0)
	else:
		features.append(0.0)

	return features
