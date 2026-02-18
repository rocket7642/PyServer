import config
import map_utils


def compute_mass_spot_score(unit_x, unit_z, spot):
	terrain_weight = config.PATH_TERRAIN_WEIGHT
	spike_threshold = config.PATH_SPIKE_THRESHOLD
	sample_count = config.PATH_SAMPLE_COUNT
	path_penalty = 0.0
	try:
		prev_height = None
		max_delta = 0.0
		for step in range(sample_count + 1):
			step_ratio = step / float(sample_count)
			sample_x = unit_x + (spot[0] - unit_x) * step_ratio
			sample_z = unit_z + (spot[1] - unit_z) * step_ratio
			sample_height = map_utils.height_at_normalized(sample_x, sample_z)
			if prev_height is not None:
				delta = abs(sample_height - prev_height)
				if delta > max_delta:
					max_delta = delta
			prev_height = sample_height

		if max_delta > spike_threshold:
			path_penalty = (max_delta - spike_threshold) * terrain_weight
	except (IndexError, TypeError):
		path_penalty = 0.0

	dist = ((spot[0] - unit_x) ** 2 + (spot[1] - unit_z) ** 2) ** 0.5
	return dist + path_penalty


def select_best_mass_spot(unit_x, unit_z, unvisited_mass, return_score=False):
	if not unvisited_mass:
		return (None, None) if return_score else None
	spots_with_dist = []
	for spot in unvisited_mass:
		dist = ((spot[0] - unit_x) ** 2 + (spot[1] - unit_z) ** 2) ** 0.5
		spots_with_dist.append((dist, spot))
	spots_with_dist.sort(key=lambda item: item[0])
	closest_spots = spots_with_dist[:4]

	best_score = None
	best_spot = None

	for dist, spot in closest_spots:
		score = compute_mass_spot_score(unit_x, unit_z, spot)
		if best_score is None or score < best_score:
			best_score = score
			best_spot = spot

	if return_score:
		return best_spot, best_score
	return best_spot


def get_top_mass_spots(unit_x, unit_z, unvisited_mass, limit=4):
	if not unvisited_mass:
		return []

	spots_with_dist = []
	for spot in unvisited_mass:
		dist = ((spot[0] - unit_x) ** 2 + (spot[1] - unit_z) ** 2) ** 0.5
		spots_with_dist.append((dist, spot))
	spots_with_dist.sort(key=lambda item: item[0])
	closest_spots = spots_with_dist[:max(1, limit)]

	ranked = []
	for _, spot in closest_spots:
		score = compute_mass_spot_score(unit_x, unit_z, spot)
		ranked.append((spot, score))

	ranked.sort(key=lambda item: item[1])
	return ranked


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
		current_dist = ((best_spot[0] - unit_x) ** 2 + (best_spot[1] - unit_z) ** 2) ** 0.5
		target_dist = ((best_spot[0] - target_x) ** 2 + (best_spot[1] - target_z) ** 2) ** 0.5
		dist_reduction = current_dist - target_dist
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
	magnitude_feature = -move_magnitude / 100.0
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
		map_x = int(target_x)
		map_z = int(target_z)
		if 0 <= map_z < enemy_range_image.shape[0] and 0 <= map_x < enemy_range_image.shape[1]:
			danger_value = enemy_range_image[map_z, map_x]
			danger_feature = -danger_value
			features.append(danger_feature)
		else:
			features.append(0.0)
	else:
		features.append(0.0)

	return features
