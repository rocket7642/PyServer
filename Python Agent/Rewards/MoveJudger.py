import config
import map_utils
import unit_defs
import numpy as np


def compute_mass_spot_score(unit_x, unit_z, spot):
	"""Compute terrain-aware cost to reach a mass spot using precomputed cost field."""
	return map_utils.get_mass_cost_at(spot, unit_x, unit_z)


def select_best_mass_spot(unit_x, unit_z, unvisited_mass, return_score=False):
	"""Find the unvisited mass spot with the lowest value-weighted cost from the unit's position.
	
	Weighting combines terrain cost with mass point value: higher-value spots receive a reduced effective cost,
	making them more attractive when terrain costs are similar.
	Spot format: (mx, mz) for key or (mx, mz, value_norm) for full triple.
	"""
	if not unvisited_mass:
		return (None, None) if return_score else None
	
	best_weighted_score = None
	best_spot = None
	best_terrain_score = None

	for spot in unvisited_mass:
		terrain_score = compute_mass_spot_score(unit_x, unit_z, spot)
		# Skip unreachable mass spots
		if np.isinf(terrain_score):
			continue
		
		# Extract value normalization from spot (mx, mz, value_norm)
		# If spot is a 2-tuple (key), assume value_norm = 1.0 (no weighting)
		try:
			value_norm = float(spot[2]) if len(spot) > 2 else 1.0
		except (IndexError, TypeError, ValueError):
			value_norm = 1.0
		
		# Apply value weighting: higher value reduces the effective cost
		# weighted_score = terrain_cost * (1 - alpha * value_norm)
		# When value_norm=1.0 (highest), scale = 1 - alpha, reducing cost
		# When value_norm=0.0 (lowest), scale = 1.0, cost unchanged
		try:
			alpha = float(getattr(config, 'MASS_VALUE_ALPHA', 0.5))
		except Exception:
			alpha = 0.5
		
		scale = 1.0 - alpha * value_norm
		if scale <= 0.0:
			scale = 0.01  # Prevent zero/negative scaling
		weighted_score = terrain_score * scale
		
		if best_weighted_score is None or weighted_score < best_weighted_score:
			best_weighted_score = weighted_score
			best_spot = spot
			best_terrain_score = terrain_score

	if return_score:
		# Return terrain score for logging (ground-truth distance metric)
		return best_spot, best_terrain_score if best_terrain_score is not None else best_weighted_score
	return best_spot


def get_top_mass_spots(unit_x, unit_z, unvisited_mass, limit=4):
	"""Return the top N nearest reachable mass spots ranked by value-weighted terrain cost.
	
	Weighting prefers higher-value spots: weighted_score = terrain_cost * (1 - alpha * value_norm).
	"""
	if not unvisited_mass:
		return []

	ranked = []
	for spot in unvisited_mass:
		terrain_score = compute_mass_spot_score(unit_x, unit_z, spot)
		# Skip unreachable mass spots
		if np.isinf(terrain_score):
			continue
		
		# Extract and apply value weighting
		try:
			value_norm = float(spot[2]) if len(spot) > 2 else 1.0
		except (IndexError, TypeError, ValueError):
			value_norm = 1.0
		
		try:
			alpha = float(getattr(config, 'MASS_VALUE_ALPHA', 0.5))
		except Exception:
			alpha = 0.5
		
		scale = 1.0 - alpha * value_norm
		if scale <= 0.0:
			scale = 0.01
		weighted_score = terrain_score * scale
		
		ranked.append((spot, weighted_score))

	ranked.sort(key=lambda item: item[1])
	return ranked[:limit]


def compute_action_features(action, unit_x, unit_z, unit_y, unvisited_mass, target_x=None, target_z=None, enemy_range_image=None, enemy_units=None, vision_image=None):
	"""Compute the feature vector for an action (NOOP or MOVE) used to score candidates via dot product with learned weights."""
	features = []

	if action == config.NOOP_ACTION:
		features.append(0.0)  # distance_reduction
		#features.append(0.0)  # boundary_proximity
		features.append(0.0)  # move_magnitude
		#features.append(0.0)  # terrain_steepness
		features.append(1.0)  # is_noop
		#features.append(0.0)  # height_change
		features.append(0.0)  # danger_zone
		features.append(0.0)  # enemy_distance_change
		# nearest_enemy_proximity: even while standing still, report how close the nearest enemy is
		nearest_info = map_utils.find_nearest_enemy(unit_x, unit_z, enemy_units) if enemy_units else None
		if nearest_info is not None:
			nearest_dist = nearest_info[0]
			# Negative value when close (penalizes staying near enemies), 0 when far
			proximity_feature = -max(0.0, config.ENEMY_PROXIMITY_THRESHOLD - nearest_dist) / config.ENEMY_PROXIMITY_THRESHOLD
		else:
			proximity_feature = 0.0
		features.append(proximity_feature)  # nearest_enemy_proximity
		features.append(0.0)  # escape_alignment
		features.append(0.0)  # skirt_alignment
		features.append(0.0)  # dodge_viability (standing still = no dodge)
		features.append(0.0)  # hazard_prediction (filled by caller in agent_core)
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
			dist_reduction = -1.0
		else:
			dist_reduction = float(np.tanh((current_cost - target_cost) / 200.0))  # Squash to [-1, 1] range for stability
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
	magnitude_feature = max(-1.0,  min(magnitude_feature, 1.0))  # Cap the penalty to prevent extreme values from dominating
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

	# Danger feature based on enemy range image - now uses gradient intensity
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
			# Use mean danger intensity along the path (gradient, not binary)
			danger_intensity = float(np.mean(path_values))
			features.append(-danger_intensity)
		else:
			features.append(0.0)
	else:
		features.append(0.0)

	# === NEW ENEMY AVOIDANCE FEATURES ===

	# enemy_distance_change: positive when moving away from nearest enemy, negative when approaching
	nearest_info = map_utils.find_nearest_enemy(unit_x, unit_z, enemy_units) if enemy_units else None
	if nearest_info is not None:
		current_enemy_dist = nearest_info[0]
		target_enemy_dist_info = map_utils.find_nearest_enemy(target_x, target_z, enemy_units)
		if target_enemy_dist_info is not None:
			target_enemy_dist = target_enemy_dist_info[0]
			enemy_dist_change = target_enemy_dist - current_enemy_dist
		else:
			enemy_dist_change = 0.0
		features.append(max(-1.0, min(1.0, enemy_dist_change / 100.0)))  # scale down, clamped
	else:
		features.append(0.0)

	# nearest_enemy_proximity: how close the target position is to the nearest enemy
	# Negative when close (within threshold), 0 when far away
	if nearest_info is not None:
		target_enemy_info = map_utils.find_nearest_enemy(target_x, target_z, enemy_units) if enemy_units else None
		if target_enemy_info is not None:
			target_nearest_dist = target_enemy_info[0]
			proximity_feature = -max(0.0, config.ENEMY_PROXIMITY_THRESHOLD - target_nearest_dist) / config.ENEMY_PROXIMITY_THRESHOLD
		else:
			proximity_feature = 0.0
		features.append(proximity_feature)
	else:
		features.append(0.0)

	# escape_alignment: how well the move direction aligns with the escape direction (away from enemies)
	escape_dx, escape_dz = map_utils.compute_enemy_escape_direction(unit_x, unit_z, enemy_units) if enemy_units else (0.0, 0.0)
	move_dx = target_x - unit_x
	move_dz = target_z - unit_z
	move_mag = (move_dx ** 2 + move_dz ** 2) ** 0.5
	escape_mag = (escape_dx ** 2 + escape_dz ** 2) ** 0.5
	if move_mag > 1e-6 and escape_mag > 1e-6:
		escape_alignment = (move_dx * escape_dx + move_dz * escape_dz) / (move_mag * escape_mag + 1e-6)
		skirt_alignment = abs(move_dx * escape_dz - move_dz * escape_dx) / (move_mag * escape_mag + 1e-6)
	else:
		escape_alignment = 0.0
		skirt_alignment = 0.0
	features.append(escape_alignment)
	features.append(skirt_alignment)

	# dodge_viability: scores lateral (perpendicular) movement relative to projectile/missile enemies.
	# Hitscan/beam enemies can't be dodged, so only dodgeable enemies contribute.
	dodge_score = 0.0
	if enemy_units and move_mag > 1e-6:
		dodgeable_count = 0
		for eu in enemy_units:
			wtype = eu.get('weapon_type', 'projectile')
			if wtype in ('projectile', 'missile'):
				# Vector from enemy to our current position
				eu_nx = map_utils.normalize_x(eu['x'])
				eu_nz = map_utils.normalize_z(eu['z'])
				fire_dx = unit_x - eu_nx
				fire_dz = unit_z - eu_nz
				fire_mag = (fire_dx ** 2 + fire_dz ** 2) ** 0.5
				if fire_mag > 1e-6:
					# Perpendicular component of move relative to incoming fire line
					# dot = parallel, cross magnitude = perpendicular
					cross = abs(move_dx * fire_dz - move_dz * fire_dx) / (move_mag * fire_mag + 1e-6)
					# Scale by inverse of projectile speed (slower = easier to dodge)
					proj_speed = eu.get('projectile_speed', 200)
					speed_factor = 1.0 - unit_defs.normalize_projectile_speed(proj_speed)
					
					# High priority: enemies closer to us require dodging more urgently
					dist_factor = 1.0 / (1.0 + (fire_mag / 100.0))
					
					dodge_score += cross * (1.0 + speed_factor) * dist_factor
					dodgeable_count += 1
		if dodgeable_count > 0:
			dodge_score /= dodgeable_count
	dodge_score = min(dodge_score, 1.0)  # Cap to prevent extreme values
	features.append(dodge_score)
	features.append(0.0)  # hazard_prediction (filled by caller in agent_core)

	return features
