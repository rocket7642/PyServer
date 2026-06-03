import numpy as np
import torch
from pathlib import Path
import hashlib

import unit_defs
import config
import runtime_state as state


def normalize_x(x):
    """Convert a world X coordinate to the standardized map coordinate space."""
    return (x / state.map_width) * config.STANDARD_MAP_WIDTH if state.map_width > 0 else x


def normalize_z(z):
    """Convert a world Z coordinate to the standardized map coordinate space."""
    return (z / state.map_height) * config.STANDARD_MAP_HEIGHT if state.map_height > 0 else z


def denormalize_x(nx):
    """Convert a standardized X coordinate back to world space."""
    return (nx / config.STANDARD_MAP_WIDTH) * state.map_width if state.map_width > 0 else nx


def denormalize_z(nz):
    """Convert a standardized Z coordinate back to world space."""
    return (nz / config.STANDARD_MAP_HEIGHT) * state.map_height if state.map_height > 0 else nz


def normalize_range(rng):
    """Scale a range value (e.g. weapon range) to the standardized map coordinate space."""
    if state.map_width > 0 and state.map_height > 0:
        scale = (config.STANDARD_MAP_WIDTH / state.map_width + config.STANDARD_MAP_HEIGHT / state.map_height) / 2.0
        return rng * scale
    return rng


def normalize_distance(dist):
    """Scale a distance value to the standardized map coordinate space."""
    if state.map_width > 0 and state.map_height > 0:
        scale = (config.STANDARD_MAP_WIDTH / state.map_width + config.STANDARD_MAP_HEIGHT / state.map_height) / 2.0
        return dist * scale
    return dist


def normalize_y(y):
    """Normalize a world Y (height) value to the standard range using the map's min/max heights."""
    if np.isfinite(state.map_height_min) and np.isfinite(state.map_height_max) and state.map_height_max > state.map_height_min:
        return (y - state.map_height_min) / (state.map_height_max - state.map_height_min) * config.STANDARD_MAP_Y
    return y


def denormalize_y(ny):
    """Convert a normalized Y (height) value back to the original world height."""
    if np.isfinite(state.map_height_min) and np.isfinite(state.map_height_max) and state.map_height_max > state.map_height_min:
        return state.map_height_min + (ny / config.STANDARD_MAP_Y) * (state.map_height_max - state.map_height_min)
    return ny


def build_normalized_height_map():
    """Resample the raw height map to the standard resolution and normalize heights for the neural network."""
    if state.map_heights is None or state.map_heights.size == 0:
        state.normalized_map_heights = None
        return

    heights = np.array(state.map_heights, dtype=np.float32)
    heights = np.nan_to_num(heights, nan=0.0, posinf=0.0, neginf=0.0)

    state.map_height_min = float(np.nanmin(heights))
    state.map_height_max = float(np.nanmax(heights))
    if not np.isfinite(state.map_height_min) or not np.isfinite(state.map_height_max):
        state.map_height_min = 0.0
        state.map_height_max = 1.0

    src_h, src_w = heights.shape
    x_idx = np.linspace(0, src_w - 1, config.STANDARD_MAP_WIDTH).astype(int)
    z_idx = np.linspace(0, src_h - 1, config.STANDARD_MAP_HEIGHT).astype(int)
    normalized = heights[np.ix_(z_idx, x_idx)]

    if state.map_height_max > state.map_height_min:
        normalized = (normalized - state.map_height_min) / (state.map_height_max - state.map_height_min) * config.STANDARD_MAP_Y

    state.normalized_map_heights = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    state.cached_map_embedding = None
    state.cached_map_embedding_device = None


def height_at_normalized(nx, nz):
    """Look up the normalized height value at a given standardized (x, z) position."""
    if state.normalized_map_heights is None:
        return -1000.0

    map_x = int(max(0, min(config.STANDARD_MAP_WIDTH - 1, nx)))
    map_z = int(max(0, min(config.STANDARD_MAP_HEIGHT - 1, nz)))
    return float(state.normalized_map_heights[map_z, map_x])


def get_cached_map_embedding(agent, device):
    """Return the CNN-encoded map embedding, computing and caching it on the first call per device."""
    if state.normalized_map_heights is None:
        return torch.zeros(config.MAP_EMBED_SIZE, dtype=torch.float32, device=device)

    if state.cached_map_embedding is not None and state.cached_map_embedding_device == device:
        return state.cached_map_embedding

    map_array = np.array(state.normalized_map_heights, dtype=np.float32, copy=True)
    map_array = np.nan_to_num(map_array, nan=0.0, posinf=0.0, neginf=0.0)
    map_mean = np.mean(map_array)
    map_std = np.std(map_array)
    if np.isfinite(map_std) and map_std > 0:
        map_array = (map_array - map_mean) / map_std
    else:
        map_array = map_array - map_mean
    map_array = np.nan_to_num(map_array, nan=0.0, posinf=0.0, neginf=0.0)

    map_tensor = torch.tensor(map_array, dtype=torch.float32, device=device)
    map_tensor = map_tensor.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        map_cnn_out = agent.map_cnn(map_tensor)
        map_emb = agent.map_fc(map_cnn_out.squeeze(0))
        map_emb = torch.nan_to_num(map_emb, nan=0.0, posinf=0.0, neginf=0.0)

    state.cached_map_embedding = map_emb.detach()
    state.cached_map_embedding_device = device
    return state.cached_map_embedding

def get_vision_embedding(agent, vision_image, device):
    """Encode the vision image using the agent's vision CNN to produce a fixed-size embedding."""
    if vision_image is None:
        return torch.zeros(config.VISION_EMBED_SIZE, dtype=torch.float32, device=device)

    img_array = np.array(vision_image, dtype=np.float32)
    img_array = np.nan_to_num(img_array, nan=0.0, posinf=0.0, neginf=0.0)
    img_tensor = torch.tensor(img_array, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        vision_cnn_out = agent.vision_cnn(img_tensor)
        vision_emb = agent.vision_fc(vision_cnn_out.squeeze(0))
        vision_emb = torch.nan_to_num(vision_emb, nan=0.0, posinf=0.0, neginf=0.0)
    return vision_emb


def reconstruct_state_with_map(agent, state_no_map):
    """Reinsert the cached map embedding into a state vector that was stored without it."""
    split_idx = config.SELF_EMBED_SIZE + config.ECO_EMBED_SIZE + config.MASS_EMBED_SIZE 
    device = state_no_map.device
    map_emb = get_cached_map_embedding(agent, device)
    prefix = state_no_map[:split_idx]
    suffix = state_no_map[split_idx:]
    return torch.cat([prefix, map_emb, suffix], dim=0)


def normalize_image(img_array):
    """Min-max normalize an image array to [0, 1] for visualization."""
    if img_array.size == 0:
        return img_array
    arr = np.array(img_array, dtype=np.float32, copy=True)
    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return np.zeros_like(arr, dtype=np.float32)

    finite_vals = arr[finite_mask]
    min_val = float(np.min(finite_vals))
    max_val = float(np.max(finite_vals))
    if max_val - min_val <= 1e-8:
        out = np.zeros_like(arr, dtype=np.float32)
        out[~finite_mask] = 1.0
        return out

    arr[~finite_mask] = max_val
    return ((arr - min_val) / (max_val - min_val)).astype(np.float32)


def reward_map_to_rgb(reward_map):
    """Convert a scalar reward map [0,1] into an RGB heatmap (red=bad, green=good)."""
    reward = np.clip(np.array(reward_map, dtype=np.float32), 0.0, 1.0)
    red = 1.0 - reward
    green = reward
    blue = np.clip(0.25 * (1.0 - np.abs(2.0 * reward - 1.0)), 0.0, 0.25)
    return np.stack([red, green, blue], axis=-1).astype(np.float32)

import math

def generate_vision_image(friendly_units, map_w, map_h, map_heights):
    """Generate a vision and radar coverage map.
    Returns a normalized float32 array: 1.0 = spotted by sight, 0.5 = spotted by radar, 0.0 = fog.
    Calculates simple raycasts to simulate terrain blocking for radar and LOS.
    """
    if map_heights is None:
        return None
        
    h, w = map_heights.shape
    vis_img = np.zeros((h, w), dtype=np.float32)
    
    if not friendly_units:
        return vis_img
        
    range_scale = 1.0
    if map_w > 0 and map_h > 0:
        scale_x = float(w) / map_w
        scale_z = float(h) / map_h
        range_scale = (scale_x + scale_z) / 2.0

    height_range = float(state.map_height_max - state.map_height_min) if np.isfinite(state.map_height_max) else 1.0
    y_world_per_norm = height_range / max(float(config.STANDARD_MAP_Y), 1e-6)
    x_world_step = max(((state.map_width/8) / float(config.STANDARD_MAP_WIDTH)) if hasattr(state, 'map_width') and state.map_width > 0 else 1.0, 1e-6)
    
    # We will use the config threshold safely 
    max_block_slope = config.MAX_TRAVERSABLE_SLOPE * 2.0
    
    num_rays = 48
    angles = np.linspace(0, 2 * math.pi, num_rays, endpoint=False)
    dx_all = np.cos(angles)
    dz_all = np.sin(angles)
    
    for unit in friendly_units:
        radar_range = unit.get('radar_range', 0)
        sight_range = unit.get('sight_range', 0)
        max_range = max(radar_range, sight_range)
        if max_range <= 0:
            continue
            
        ex = int(normalize_x(unit.get('x', 0)))
        ez = int(normalize_z(unit.get('z', 0)))
        if not (0 <= ez < h and 0 <= ex < w):
            continue
            
        unit_h = map_heights[ez, ex]
        r_pixel = int(max_range * range_scale)
        if r_pixel <= 0:
            continue
            
        radar_pixel = int(radar_range * range_scale)
        sight_pixel = int(sight_range * range_scale)
        
        vis_img[ez, ex] = max(vis_img[ez, ex], 1.0 if sight_pixel > 0 else 0.5)
        
        for dx, dz in zip(dx_all, dz_all):
            max_slope = -float('inf')
            for step in range(1, r_pixel + 1):
                px = int(ex + step * dx)
                pz = int(ez + step * dz)
                
                if not (0 <= px < w and 0 <= pz < h):
                    break
                    
                target_h = map_heights[pz, px]
                dh_world = (target_h - unit_h) * y_world_per_norm
                dist_world = step * x_world_step
                
                slope = dh_world / dist_world if dist_world > 0 else 0
                
                if slope < max_slope and max_slope > max_block_slope:
                    break
                    
                max_slope = max(max_slope, slope)
                
                if step <= sight_pixel:
                    vis_img[pz, px] = max(vis_img[pz, px], 1.0)
                elif step <= radar_pixel:
                    vis_img[pz, px] = max(vis_img[pz, px], 0.5)

    return vis_img

def is_position_buildable(tx, tz, unit):
    """Check if a position is buildable based on terrain cost map (e.g. not blocked by impassable terrain)."""
    if state.terrain_cost_map is None:
        return False
    unit_size = unit_defs.get_unit_size("armrad")  # Assuming we're building an armrad, adjust if needed (use unit in the future)
    tw = unit_size.get("width", 1) * 16.0
    th = unit_size.get("height", 1) * 16.0
    map_w = state.map_width
    map_h = state.map_height
    if map_w <= 0 or map_h <= 0:
        return False
    tw_norm = (tw / map_w) * config.STANDARD_MAP_WIDTH
    th_norm = (th / map_h) * config.STANDARD_MAP_HEIGHT
    left = int(max(0, min(config.STANDARD_MAP_WIDTH - 1, tx - tw_norm / 2.0)))
    right = int(max(0, min(config.STANDARD_MAP_WIDTH - 1, tx + tw_norm / 2.0)))
    top = int(max(0, min(config.STANDARD_MAP_HEIGHT - 1, tz - th_norm / 2.0)))
    bottom = int(max(0, min(config.STANDARD_MAP_HEIGHT - 1, tz + th_norm / 2.0)))

    return np.all(np.isfinite(state.terrain_cost_map[top:bottom+1, left:right+1]))


def generate_enemy_range_image(enemy_units, map_w, map_h, map_heights_shape, enemy_range=None):
    """Generate a gradient danger image using per-unit weapon range with intensity falloff from each enemy."""
    if map_heights_shape is None:
        return None
    h, w = map_heights_shape
    img = np.zeros((h, w), dtype=np.float32)
    if map_w > 0 and map_h > 0:
        range_scale = (config.STANDARD_MAP_WIDTH / map_w + config.STANDARD_MAP_HEIGHT / map_h) / 2.0
    else:
        range_scale = 1.0
    for enemy in enemy_units:
        ex = int(normalize_x(enemy['x']))
        ez = int(normalize_z(enemy['z']))
        # Use the unit's actual weapon range if available, otherwise fall back to default
        unit_range = enemy.get('range', config.DEFAULT_ENEMY_RANGE)
        if unit_range <= 0:
            unit_range = config.DEFAULT_ENEMY_RANGE
        r = int(unit_range * range_scale * config.ENEMY_RANGE_FALLOFF_BUFFER)
        r_inner = int(unit_range * range_scale)
        x_min = max(0, ex - r)
        x_max = min(w - 1, ex + r)
        z_min = max(0, ez - r)
        z_max = min(h - 1, ez + r)
        r_sq = r * r
        r_inner_sq = r_inner * r_inner
        for z in range(z_min, z_max + 1):
            for x in range(x_min, x_max + 1):
                dist_sq = (x - ex) ** 2 + (z - ez) ** 2
                if dist_sq <= r_sq:
                    if dist_sq <= r_inner_sq:
                        # Full danger inside actual weapon range
                        intensity = 1.0
                    else:
                        # Linear falloff in the buffer zone between weapon range and outer radius
                        dist = dist_sq ** 0.5
                        intensity = max(0.0, 1.0 - (dist - r_inner) / max(1.0, r - r_inner))
                    img[z, x] = max(img[z, x], intensity)
    return img


def find_nearest_enemy(unit_nx, unit_nz, enemy_units):
    """Find the nearest enemy to a normalized position, returning (distance, enemy_nx, enemy_nz) or None if no enemies."""
    if not enemy_units:
        return None
    best_dist = float('inf')
    best_ex, best_ez = 0.0, 0.0
    for enemy in enemy_units:
        ex = normalize_x(enemy['x'])
        ez = normalize_z(enemy['z'])
        dist = ((ex - unit_nx) ** 2 + (ez - unit_nz) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_ex = ex
            best_ez = ez
    return best_dist, best_ex, best_ez


def compute_enemy_escape_direction(unit_nx, unit_nz, enemy_units):
    """Compute the best escape direction (dx, dz) as a unit vector pointing away from nearby enemies."""
    if not enemy_units:
        return 0.0, 0.0
    flee_dx = 0.0
    flee_dz = 0.0
    for enemy in enemy_units:
        ex = normalize_x(enemy['x'])
        ez = normalize_z(enemy['z'])
        dx = unit_nx - ex
        dz = unit_nz - ez
        dist = (dx ** 2 + dz ** 2) ** 0.5
        if dist < config.ENEMY_PROXIMITY_THRESHOLD and dist > 1e-6:
            # Weight contribution inversely by distance (closer enemies matter more)
            weight = 1.0 / (dist + 1e-6)
            flee_dx += dx * weight
            flee_dz += dz * weight
    mag = (flee_dx ** 2 + flee_dz ** 2) ** 0.5
    if mag > 1e-6:
        return flee_dx / mag, flee_dz / mag
    return 0.0, 0.0


def sample_path_values(grid, start_x, start_z, end_x, end_z, sample_count=None):
    """Sample values from a grid along a straight line between two points."""
    if grid is None:
        return None

    h, w = grid.shape
    samples = max(2, int(sample_count or config.PATH_SAMPLE_COUNT))

    xs = np.linspace(start_x, end_x, num=samples)
    zs = np.linspace(start_z, end_z, num=samples)

    x_idx = np.clip(xs.astype(int), 0, w - 1)
    z_idx = np.clip(zs.astype(int), 0, h - 1)
    return grid[z_idx, x_idx]


def estimate_path_terrain_penalty(start_x, start_z, end_x, end_z, sample_count=None):
    """Estimate a terrain traversal penalty along a path by sampling the cost map between two points."""
    if state.terrain_cost_map is None:
        return 0.0

    path_costs = sample_path_values(
        state.terrain_cost_map,
        start_x,
        start_z,
        end_x,
        end_z,
        sample_count=sample_count or config.PATH_SAMPLE_COUNT,
    )
    if path_costs is None or path_costs.size == 0:
        return 0.0

    # Replace inf (impassable) samples with a large but finite penalty
    # so that paths crossing impassable terrain are strongly discouraged
    # without producing inf/nan in downstream Q-value calculations.
    finite_mask = np.isfinite(path_costs)
    if not np.any(finite_mask):
        # Entire path is impassable — return a large finite penalty
        return -float(config.PATH_TERRAIN_WEIGHT * 100.0)

    path_costs = np.where(finite_mask, path_costs, np.max(path_costs[finite_mask]) * 10.0)

    excess_cost = np.maximum(path_costs - 1.0, 0.0)
    if excess_cost.size == 0:
        return 0.0

    return -float(excess_cost.mean() * config.PATH_TERRAIN_WEIGHT)


def _collect_action_candidates_for_view(unit_nx, unit_nz, enemies, mass_destination):
    """Collect candidate points using the same generation logic used during action selection."""
    candidates = []

    for dx in np.linspace(-200, 200, num=10):
        for dz in np.linspace(-200, 200, num=10):
            tx = unit_nx + dx
            tz = unit_nz + dz
            tx = max(0, min(config.STANDARD_MAP_WIDTH, tx))
            tz = max(0, min(config.STANDARD_MAP_HEIGHT, tz))
            if is_position_reachable(tx, tz):
                candidates.append((tx, tz, 'grid'))

    # Current position candidate (NOOP)
    candidates.append((unit_nx, unit_nz, 'noop'))

    # Escape candidates
    escape_dx, escape_dz = compute_enemy_escape_direction(unit_nx, unit_nz, enemies)
    if abs(escape_dx) > 1e-6 or abs(escape_dz) > 1e-6:
        import math
        for dist_mult in [0.5, 1.0, 1.5]:
            esc_dist = config.ESCAPE_CANDIDATE_DISTANCE * dist_mult
            for angle_offset in np.linspace(-0.5, 0.5, config.ESCAPE_CANDIDATE_COUNT):
                base_angle = math.atan2(escape_dz, escape_dx)
                angle = base_angle + angle_offset * math.pi
                tx = unit_nx + math.cos(angle) * esc_dist
                tz = unit_nz + math.sin(angle) * esc_dist
                tx = max(0, min(config.STANDARD_MAP_WIDTH, tx))
                tz = max(0, min(config.STANDARD_MAP_HEIGHT, tz))

                outside_enemy_range = True
                for enemy in enemies:
                    ex = normalize_x(enemy['x'])
                    ez = normalize_z(enemy['z'])
                    enemy_range = normalize_range(enemy.get('range', config.DEFAULT_ENEMY_RANGE))
                    dist_to_enemy = ((tx - ex) ** 2 + (tz - ez) ** 2) ** 0.5
                    if dist_to_enemy <= enemy_range:
                        outside_enemy_range = False
                        break

                if is_position_reachable(tx, tz) and outside_enemy_range:
                    candidates.append((tx, tz, 'escape'))

    # Lateral dodge candidates
    for enemy in enemies:
        wtype = enemy.get('weapon_type', 'projectile')
        if wtype in ('projectile', 'missile'):
            ex = normalize_x(enemy['x'])
            ez = normalize_z(enemy['z'])
            fire_dx = unit_nx - ex
            fire_dz = unit_nz - ez
            fire_mag = (fire_dx ** 2 + fire_dz ** 2) ** 0.5
            if fire_mag > 1e-6:
                perp_dx = -fire_dz / fire_mag
                perp_dz = fire_dx / fire_mag
                for sign in [1.0, -1.0]:
                    for dist_mult in [0.5, 1.0]:
                        d = config.ESCAPE_CANDIDATE_DISTANCE * dist_mult
                        tx = unit_nx + sign * perp_dx * d
                        tz = unit_nz + sign * perp_dz * d
                        tx = max(0, min(config.STANDARD_MAP_WIDTH, tx))
                        tz = max(0, min(config.STANDARD_MAP_HEIGHT, tz))

                        # Match intended dodge logic: keep points that increase distance from shooter.
                        new_fire_mag = ((tx - ex) ** 2 + (tz - ez) ** 2) ** 0.5
                        further_from_enemy = new_fire_mag >= fire_mag

                        if is_position_reachable(tx, tz) and further_from_enemy:
                            candidates.append((tx, tz, 'strafe'))

    # Short-range scatter candidates for throwing off predictive projectiles.
    # These stay close to the current position to avoid large path deviations.
    for enemy in enemies:
        wtype = enemy.get('weapon_type', 'projectile')
        if wtype in ('projectile', 'missile'):
            ex = normalize_x(enemy['x'])
            ez = normalize_z(enemy['z'])
            fire_dx = unit_nx - ex
            fire_dz = unit_nz - ez
            fire_mag = (fire_dx ** 2 + fire_dz ** 2) ** 0.5
            if fire_mag > 1e-6:
                perp_dx = -fire_dz / fire_mag
                perp_dz = fire_dx / fire_mag
                # Prefer tiny lateral jinks plus slight forward/back offsets.
                # This creates a compact scatter pattern around the unit.
                forward_dx = fire_dx / fire_mag
                forward_dz = fire_dz / fire_mag
                for lateral_sign in [1.0, -1.0]:
                    for forward_sign in [0.0, 1.0, -1.0]:
                        for dist_mult in [0.08, 0.14, 0.2]:
                            d = config.ESCAPE_CANDIDATE_DISTANCE * dist_mult
                            tx = unit_nx + (lateral_sign * perp_dx + 0.45 * forward_sign * forward_dx) * d
                            tz = unit_nz + (lateral_sign * perp_dz + 0.45 * forward_sign * forward_dz) * d
                            tx = max(0, min(config.STANDARD_MAP_WIDTH, tx))
                            tz = max(0, min(config.STANDARD_MAP_HEIGHT, tz))
                            if is_position_reachable(tx, tz):
                                candidates.append((tx, tz, 'dodge'))

    if mass_destination is not None:
        if is_position_reachable(mass_destination[0], mass_destination[1]):
            candidates.append((mass_destination[0], mass_destination[1], 'mass_destination'))

        terrain_waypoints = extract_terrain_waypoints(
            mass_destination,
            unit_nx,
            unit_nz,
            count=config.TERRAIN_WAYPOINT_COUNT,
            search_radius=config.TERRAIN_WAYPOINT_SEARCH_RADIUS,
        )
        candidates.extend((wx, wz, 'terrain_waypoint') for wx, wz in terrain_waypoints)

    return candidates


def get_local_view_image(unit_x, unit_z, map_heights_array, view_size=500, unit_id=None, enemy_units=None):
    """Extract a local height-map patch with overlaid action candidates for TensorBoard visualization."""
    if map_heights_array is None:
        return None
    h, w = map_heights_array.shape
    cx = int(normalize_x(unit_x))
    cz = int(normalize_z(unit_z))
    half = view_size // 2
    x_min = max(0, cx - half)
    x_max = min(w, cx + half)
    z_min = max(0, cz - half)
    z_max = min(h, cz + half)

    patch = map_heights_array[z_min:z_max, x_min:x_max]
    padded = np.zeros((view_size, view_size), dtype=np.float32)
    padded[0:patch.shape[0], 0:patch.shape[1]] = patch

    base_max = float(np.max(padded)) if padded.size > 0 else 0.0

    unit_local_x = min(view_size - 1, max(0, int(normalize_x(unit_x) - x_min)))
    unit_local_z = min(view_size - 1, max(0, int(normalize_z(unit_z) - z_min)))
    if 0 <= unit_local_z < view_size and 0 <= unit_local_x < view_size:
        padded[unit_local_z, unit_local_x] = base_max + 8.0

    if enemy_units is None:
        enemies = list(state.eUnits)
        for eu in state.eKUnits:
            if all(eu['id'] != existing['id'] for existing in enemies):
                enemies.append(eu)
    else:
        enemies = list(enemy_units)

    unit_nx = normalize_x(unit_x)
    unit_nz = normalize_z(unit_z)
    mass_destination = state.mass_destinations.get(unit_id) if unit_id is not None else None
    candidates = _collect_action_candidates_for_view(unit_nx, unit_nz, enemies, mass_destination)

    marker_height = {
        'grid': base_max + 1.0,
        'terrain_waypoint': base_max + 2.0,
        'dodge': base_max + 3.0,
        'escape': base_max + 4.0,
        'mass_destination': base_max + 6.0,
        'noop': base_max + 7.0,
        'strafe': base_max + 5.0,
    }

    for tx, tz, kind in candidates:
        sense_local_x = min(view_size - 1, max(0, int(tx - x_min)))
        sense_local_z = min(view_size - 1, max(0, int(tz - z_min)))
        if 0 <= sense_local_z < view_size and 0 <= sense_local_x < view_size:
            padded[sense_local_z, sense_local_x] = max(
                padded[sense_local_z, sense_local_x],
                marker_height.get(kind, base_max + 1.0),
            )

    return padded

def get_total_map_view_image(unit_x, unit_z, map_heights_array, enemy_units):
    """Generate a full-map view image with cost and enemy positions for TensorBoard visualization."""
    """This function creates a visualization of the entire map's cost data and overlays enemy positions as bright spots."""
    """Contains the units current position as a bright spot as well for reference, enemy ranges as light red zones and the terrain cost map as a base layer."""
    if map_heights_array is None:
        return None
    h, w = map_heights_array.shape
    cost_map = state.terrain_cost_map if state.terrain_cost_map is not None else np.zeros_like(map_heights_array)
    cost_norm = normalize_image(cost_map)
    enemy_range_img = generate_enemy_range_image(enemy_units, state.map_width, state.map_height, map_heights_array.shape)
    if enemy_range_img is not None:
        enemy_range_norm = normalize_image(enemy_range_img)
        combined = np.clip(cost_norm + enemy_range_norm, 0.0, 1.0)
    else:
        combined = cost_norm
    # Visualize favorability as a heatmap: red = low reward / high risk, green = high reward / low risk.
    reward_map = 1.0 - combined
    combined_rgb = reward_map_to_rgb(reward_map)
    for enemy in enemy_units:
        ex = int(normalize_x(enemy['x']))
        ez = int(normalize_z(enemy['z']))
        if 0 <= ez < h and 0 <= ex < w:
            combined_rgb[ez, ex] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    unit_ex = int(normalize_x(unit_x))
    unit_ez = int(normalize_z(unit_z))
    if 0 <= unit_ez < h and 0 <= unit_ex < w:
        combined_rgb[unit_ez, unit_ex] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return combined_rgb

def _get_map_signature():
    """Compute a SHA-256 hash of the map data and config to uniquely identify cached cost fields."""
    if state.map_heights is None:
        return None

    hasher = hashlib.sha256()

    heights = np.ascontiguousarray(state.map_heights, dtype=np.float32)
    hasher.update(heights.tobytes())

    # Include normalized mass spot coordinates and values in the signature
    mass_spots = np.ascontiguousarray(state.map_spots_norm, dtype=np.float32) if state.map_spots_norm else np.empty((0, 3), dtype=np.float32)
    hasher.update(mass_spots.tobytes())

    hasher.update(str(state.map_width).encode("utf-8"))
    hasher.update(str(state.map_height).encode("utf-8"))
    hasher.update(str(config.STANDARD_MAP_WIDTH).encode("utf-8"))
    hasher.update(str(config.STANDARD_MAP_HEIGHT).encode("utf-8"))
    hasher.update(str(config.PATH_SPIKE_THRESHOLD).encode("utf-8"))
    hasher.update(str(config.MAX_TRAVERSABLE_SLOPE).encode("utf-8"))
    hasher.update(str(config.MAP_CACHE_VERSION).encode("utf-8"))

    return hasher.hexdigest()


def _get_map_cache_path():
    """Return the filesystem path for the cached cost fields file based on the map's signature hash."""
    map_signature = _get_map_signature()
    if map_signature is None:
        return None

    cache_dir = Path(config.MAP_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{map_signature}.npz"


def load_cached_cost_fields():
    """Load previously saved terrain and mass cost fields from the cache if the map signature matches."""
    cache_path = _get_map_cache_path()
    if cache_path is None or not cache_path.exists():
        return False

    try:
        data = np.load(cache_path, allow_pickle=False)

        if int(data["cache_version"][0]) != int(config.MAP_CACHE_VERSION):
            return False

        cached_path_spike = float(data["path_spike_threshold"][0])
        if not np.isclose(cached_path_spike, config.PATH_SPIKE_THRESHOLD):
            return False

        cached_max_slope = float(data["max_traversable_slope"][0])
        if not np.isclose(cached_max_slope, config.MAX_TRAVERSABLE_SLOPE):
            return False

        cached_map_width = int(data["map_width"][0])
        cached_map_height = int(data["map_height"][0])
        if cached_map_width != int(state.map_width) or cached_map_height != int(state.map_height):
            return False

        cached_spots = data["mass_spots_norm"]
        current_spots = np.array(state.map_spots_norm, dtype=np.float32) if state.map_spots_norm else np.empty((0, 3), dtype=np.float32)

        if cached_spots.shape != current_spots.shape:
            return False
        if cached_spots.size > 0 and not np.allclose(cached_spots, current_spots, atol=1e-5):
            return False

        terrain_cost_map = np.array(data["terrain_cost_map"], dtype=np.float32, copy=False)
        mass_cost_stack = np.array(data["mass_cost_stack"], dtype=np.float32, copy=False)

        state.terrain_cost_map = terrain_cost_map
        state.mass_cost_fields = {}

        for idx, spot in enumerate(state.map_spots_norm):
            # spot is (mx, mz, value_norm)
            mx, mz = float(spot[0]), float(spot[1])
            state.mass_cost_fields[(mx, mz)] = mass_cost_stack[idx]

        print(f"Loaded cached map cost fields from {cache_path}")
        return True
    except Exception as exc:
        print(f"Failed to load map cache ({cache_path}): {exc}")
        return False


def save_cached_cost_fields():
    """Save the current terrain and mass cost fields to disk so they can be reused on the same map."""
    cache_path = _get_map_cache_path()
    if cache_path is None or state.terrain_cost_map is None:
        return False

    try:
        if state.map_spots_norm:
            mass_cost_stack = np.stack(
                [
                    np.array(
                        state.mass_cost_fields.get((float(mx), float(mz)), np.full_like(state.terrain_cost_map, np.inf, dtype=np.float32)),
                        dtype=np.float32,
                        copy=False,
                    )
                    for mx, mz, _ in state.map_spots_norm
                ],
                axis=0,
            )
            mass_spots_norm = np.array(state.map_spots_norm, dtype=np.float32)
        else:
            terrain_shape = state.terrain_cost_map.shape
            mass_cost_stack = np.empty((0, terrain_shape[0], terrain_shape[1]), dtype=np.float32)
            mass_spots_norm = np.empty((0, 2), dtype=np.float32)

        np.savez_compressed(
            cache_path,
            cache_version=np.array([config.MAP_CACHE_VERSION], dtype=np.int32),
            path_spike_threshold=np.array([config.PATH_SPIKE_THRESHOLD], dtype=np.float32),
            max_traversable_slope=np.array([config.MAX_TRAVERSABLE_SLOPE], dtype=np.float32),
            map_width=np.array([state.map_width], dtype=np.int32),
            map_height=np.array([state.map_height], dtype=np.int32),
            terrain_cost_map=np.array(state.terrain_cost_map, dtype=np.float32, copy=False),
            mass_spots_norm=mass_spots_norm,
            mass_cost_stack=mass_cost_stack,
        )

        print(f"Saved map cost fields cache to {cache_path}")
        return True
    except Exception as exc:
        print(f"Failed to save map cache ({cache_path}): {exc}")
        return False


def build_terrain_cost_map():
    """Build a cost map and per-edge slope arrays for terrain traversal.

    Slope is an *edge* property: whether a unit can move from cell A to adjacent
    cell B depends on the height difference between A and B divided by the
    real-world horizontal distance of that step.

    A cell is marked impassable (inf) only when **every** adjacent edge exceeds
    ``MAX_TRAVERSABLE_SLOPE``.  This prevents cliff bases, dips and gradual
    slopes from being falsely blocked just because one neighbouring cell is a
    cliff face.

    The per-edge slope arrays (``state.edge_slope_*``) are stored so that
    Dijkstra can reject individual steep edges without needing to mark cells.
    """
    if state.normalized_map_heights is None:
        state.terrain_cost_map = None
        return

    h, w = state.normalized_map_heights.shape
    cost_map = np.ones((h, w), dtype=np.float32)

    height_range = float(state.map_height_max - state.map_height_min)
    # if not np.isfinite(height_range) or height_range <= 0.0:
    #     state.terrain_cost_map = cost_map
    #     state.edge_slope_zp = state.edge_slope_zn = None
    #     state.edge_slope_xp = state.edge_slope_xn = None
    #     print(f"Terrain cost map built: {w}x{h}, cost range [1.00, 1.00]")
    #     print("  Impassable cells: 0 (100.0% passable)")
    #     return

    y_world_per_norm = height_range / float(config.STANDARD_MAP_Y)
    x_world_step = max(((state.map_width/8) / float(config.STANDARD_MAP_WIDTH)) if state.map_width > 0 else 1.0, 1e-6)
    z_world_step = max(((state.map_height/8) / float(config.STANDARD_MAP_HEIGHT)) if state.map_height > 0 else 1.0, 1e-6)

    for z in range(h) :
        for x in range(w) :
            neighbor_heights = []
            zPos = z * z_world_step
            xPos = x * x_world_step
            if z > 0:
                neighbor_heights.append(state.map_heights[(int)(zPos - z_world_step), (int)(xPos)])
            else : neighbor_heights.append(0.0)
            if z < h - 1:
                neighbor_heights.append(state.map_heights[(int)(zPos + z_world_step), (int)(xPos)])
            else : neighbor_heights.append(0.0)
            if x > 0:
                neighbor_heights.append(state.map_heights[(int)(zPos), (int)(xPos - x_world_step)])
            else : neighbor_heights.append(0.0)
            if x < w - 1:
                neighbor_heights.append(state.map_heights[(int)(zPos), (int)(xPos + x_world_step)])
            else : neighbor_heights.append(0.0)
            changeLeft = (neighbor_heights[0] - neighbor_heights[1]) / (z_world_step * 2)
            changeRight = (neighbor_heights[2] - neighbor_heights[3]) / (x_world_step * 2)
            averageChange = (abs(changeLeft) + abs(changeRight)) / 2
            if averageChange > config.MAX_TRAVERSABLE_SLOPE*8:
                cost_map[z, x] = np.inf
            else:
                # Base cost of 1.0 ensures Dijkstra accounts for distance on flat ground.
                # Slope adds a proportional premium scaled by PATH_SPIKE_THRESHOLD.
                # Without the 1.0 base, flat paths cost ~0 regardless of length,
                # making the agent aggressively avoid any slope even when it's shorter.
                cost_map[z, x] = 1.0 + (averageChange / config.PATH_SPIKE_THRESHOLD)
    # heights = state.normalized_map_heights

    # # --- Vectorised per-edge slope computation (world units) ---
    # # slope = abs(delta_height_world) / horizontal_world_step
    # # Each array is (h, w); boundary edges are set to inf (no neighbour).

    # slope_zn = np.full((h, w), np.inf, dtype=np.float32)  # edge toward z-1
    # slope_zp = np.full((h, w), np.inf, dtype=np.float32)  # edge toward z+1
    # slope_xn = np.full((h, w), np.inf, dtype=np.float32)  # edge toward x-1
    # slope_xp = np.full((h, w), np.inf, dtype=np.float32)  # edge toward x+1

    # slope_zn[1:, :] = np.abs(heights[1:, :] - heights[:-1, :]) * y_world_per_norm / z_world_step
    # slope_zp[:-1, :] = np.abs(heights[:-1, :] - heights[1:, :]) * y_world_per_norm / z_world_step
    # slope_xn[:, 1:] = np.abs(heights[:, 1:] - heights[:, :-1]) * y_world_per_norm / x_world_step
    # slope_xp[:, :-1] = np.abs(heights[:, :-1] - heights[:, 1:]) * y_world_per_norm / x_world_step

    # # Store for Dijkstra edge checks
    # state.edge_slope_zn = slope_zn
    # state.edge_slope_zp = slope_zp
    # state.edge_slope_xn = slope_xn
    # state.edge_slope_xp = slope_xp

    # threshold = config.MAX_TRAVERSABLE_SLOPE

    # # A cell is impassable only if ALL edges leaving it are too steep
    # all_edges_blocked = (
    #     (slope_zn > threshold) &
    #     (slope_zp > threshold) &
    #     (slope_xn > threshold) &
    #     (slope_xp > threshold)
    # )

    # # Minimum traversable edge slope per cell (used for cost scaling)
    # min_edge_slope = np.minimum(
    #     np.minimum(slope_zn, slope_zp),
    #     np.minimum(slope_xn, slope_xp),
    # )

    # impassable_count = int(np.sum(all_edges_blocked))
    # cost_map[all_edges_blocked] = np.inf

    # # For passable cells, cost scales with the gentlest available edge
    # passable = ~all_edges_blocked
    # passable_min = np.where(np.isfinite(min_edge_slope) & passable, min_edge_slope, 0.0)
    # cost_map[passable] = 1.0 + (passable_min[passable] / config.PATH_SPIKE_THRESHOLD)

    state.terrain_cost_map = cost_map
    # passable_pct = 100.0 * (1.0 - impassable_count / (h * w))
    # finite_costs = cost_map[np.isfinite(cost_map)]
    # if finite_costs.size > 0:
    #     print(f"Terrain cost map built: {w}x{h}, cost range [{np.min(finite_costs):.2f}, {np.max(finite_costs):.2f}]")
    # else:
    #     print(f"Terrain cost map built: {w}x{h}, all cells impassable")
    # print(f"  Impassable cells: {impassable_count} ({passable_pct:.1f}% passable)")


def build_mass_cost_fields():
    """Precompute cost fields from each mass point using Dijkstra.

    Edge slope is checked per move: even if both cells are passable, the
    transition is blocked when the edge slope exceeds MAX_TRAVERSABLE_SLOPE.
    """
    if state.terrain_cost_map is None or not state.map_spots_norm:
        state.mass_cost_fields = {}
        return
    
    import heapq
    
    h, w = state.terrain_cost_map.shape
    state.mass_cost_fields = {}

    threshold = config.MAX_TRAVERSABLE_SLOPE

    # Edge slope arrays (may be None if height range was degenerate)
    eslope_zn = state.edge_slope_zn
    eslope_zp = state.edge_slope_zp
    eslope_xn = state.edge_slope_xn
    eslope_xp = state.edge_slope_xp
    has_edge_slopes = eslope_zn is not None

    # Direction offsets and corresponding edge slope lookup:
    #   dz, dx  →  which edge array to check at (z, x) for that move
    # Moving z-1: we leave current cell in the z-negative direction
    # Moving z+1: z-positive direction, etc.
    
    for mass_idx, (mx, mz, mval) in enumerate(state.map_spots_norm):
        # Convert normalized coords to grid coords
        grid_x = int(np.clip(mx, 0, w - 1))
        grid_z = int(np.clip(mz, 0, h - 1))
        
        # Initialize cost field
        cost_field = np.full((h, w), np.inf, dtype=np.float32)
        cost_field[grid_z, grid_x] = 0.0
        
        # Dijkstra with priority queue
        heap = [(0.0, grid_z, grid_x)]
        visited = set()
        
        while heap:
            current_cost, z, x = heapq.heappop(heap)
            
            if (z, x) in visited:
                continue
            visited.add((z, x))
            
            # Check 4-connected neighbors with edge-slope gating
            for dz, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nz, nx = z + dz, x + dx
                if 0 <= nz < h and 0 <= nx < w and (nz, nx) not in visited:
                    # Gate: reject moves across edges steeper than threshold
                    if has_edge_slopes:
                        if dz == -1:
                            edge_slope = eslope_zn[z, x]
                        elif dz == 1:
                            edge_slope = eslope_zp[z, x]
                        elif dx == -1:
                            edge_slope = eslope_xn[z, x]
                        else:
                            edge_slope = eslope_xp[z, x]
                        if edge_slope > threshold:
                            continue
                    
                    move_cost = state.terrain_cost_map[nz, nx]
                    new_cost = current_cost + move_cost
                    if new_cost < cost_field[nz, nx]:
                        cost_field[nz, nx] = new_cost
                        heapq.heappush(heap, (new_cost, nz, nx))
        
        # Apply mass-value weighting to pull costs down near higher-value spots
        try:
            alpha = float(getattr(config, 'MASS_VALUE_ALPHA', 0.5))
        except Exception:
            alpha = 0.5
        scale = 1.0 - alpha * float(mval)
        if scale <= 0.0:
            scale = 0.01
        # Only scale finite costs (leave inf as-is)
        cost_field = np.where(np.isfinite(cost_field), cost_field * scale, cost_field)

        state.mass_cost_fields[(mx, mz)] = cost_field
    
    print(f"Built {len(state.mass_cost_fields)} mass cost fields")

    # --- Second pass: mark cells unreachable from ALL mass points as impassable ---
    # If no mass point's Dijkstra can reach a cell, that cell is isolated
    # (e.g. an island mountain top) and should be treated as impassable.
    if state.mass_cost_fields:
        reachable = np.zeros((h, w), dtype=bool)
        for cf in state.mass_cost_fields.values():
            reachable |= np.isfinite(cf)

        newly_blocked = (~reachable) & np.isfinite(state.terrain_cost_map)
        blocked_count = int(np.sum(newly_blocked))
        if blocked_count > 0:
            state.terrain_cost_map[newly_blocked] = np.inf
            print(f"  Second pass: marked {blocked_count} isolated cells as impassable")


def is_position_reachable(pos_nx, pos_nz):
    """Check if a position is reachable based on terrain cost map."""
    if state.terrain_cost_map is None:
        return True  # No terrain data, assume reachable
    
    h, w = state.terrain_cost_map.shape
    grid_x = int(np.clip(pos_nx, 0, w - 1))
    grid_z = int(np.clip(pos_nz, 0, h - 1))
    
    cost = state.terrain_cost_map[grid_z, grid_x]
    return not np.isinf(cost)


def get_mass_cost_at(mass_spot, unit_nx, unit_nz):
    """Query the precomputed cost from a mass spot to a unit position. Returns np.inf if unreachable.

    Accepts `mass_spot` as either the stored dict key (mx, mz) or a triple (mx, mz, value_norm).
    """
    # Resolve key format
    if mass_spot in state.mass_cost_fields:
        key = mass_spot
    else:
        try:
            key = (float(mass_spot[0]), float(mass_spot[1]))
            if key not in state.mass_cost_fields:
                return np.inf
        except Exception:
            return np.inf
    
    cost_field = state.mass_cost_fields[key]
    h, w = cost_field.shape
    
    grid_x = int(np.clip(unit_nx, 0, w - 1))
    grid_z = int(np.clip(unit_nz, 0, h - 1))
    
    cost = cost_field[grid_z, grid_x]
    return float(cost)  # May be inf if unreachable


def extract_terrain_waypoints(mass_spot, unit_nx, unit_nz, count=None, search_radius=None):
    """Extract waypoint candidates from cost field that follow terrain-aware gradient descent.

    Accepts `mass_spot` as either a dict key (mx, mz) or a triple (mx, mz, value_norm).
    """
    # Resolve possible mass_spot formats to a key used in state.mass_cost_fields
    key = None
    if mass_spot in state.mass_cost_fields:
        key = mass_spot
    else:
        try:
            key = (float(mass_spot[0]), float(mass_spot[1]))
            if key not in state.mass_cost_fields:
                key = None
        except Exception:
            key = None

    if key is None:
        return []
    
    cost_field = state.mass_cost_fields[key]
    h, w = cost_field.shape
    
    unit_grid_x = int(np.clip(unit_nx, 0, w - 1))
    unit_grid_z = int(np.clip(unit_nz, 0, h - 1))
    current_cost = cost_field[unit_grid_z, unit_grid_x]
    
    if np.isinf(current_cost):
        return []
    
    waypoint_count = count or config.TERRAIN_WAYPOINT_COUNT
    radius = search_radius or config.TERRAIN_WAYPOINT_SEARCH_RADIUS
    
    # Sample grid cells within search radius
    x_min = max(0, unit_grid_x - radius)
    x_max = min(w - 1, unit_grid_x + radius)
    z_min = max(0, unit_grid_z - radius)
    z_max = min(h - 1, unit_grid_z + radius)
    
    candidates = []
    sample_step = max(1, int(radius / 20))  # Sample sparsely to avoid performance hit
    
    for z in range(z_min, z_max + 1, sample_step):
        for x in range(x_min, x_max + 1, sample_step):
            cell_cost = cost_field[z, x]
            if np.isinf(cell_cost):
                continue
            
            # Only consider cells with meaningfully lower cost to destination
            cost_improvement = current_cost - cell_cost
            if cost_improvement < 5.0:  # Require at least 5 cost units of progress
                continue
            
            # Calculate Euclidean distance from unit
            dx = x - unit_grid_x
            dz = z - unit_grid_z
            dist_to_cell = (dx * dx + dz * dz) ** 0.5
            
            if dist_to_cell < 10.0:  # Too close to be useful
                continue
            
            # Score by cost improvement per distance traveled
            efficiency = cost_improvement / (dist_to_cell + 1e-6)
            candidates.append((x, z, efficiency, cell_cost))
    
    if not candidates:
        return []
    
    # Sort by efficiency and take top candidates
    candidates.sort(key=lambda c: c[2], reverse=True)
    top_candidates = candidates[:waypoint_count]
    
    # Convert grid coords back to normalized coords
    waypoints = [(float(x), float(z)) for x, z, _, _ in top_candidates]
    return waypoints
