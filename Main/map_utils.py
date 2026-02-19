import numpy as np
import torch
from pathlib import Path
import hashlib

import config
import runtime_state as state


def normalize_x(x):
    return (x / state.map_width) * config.STANDARD_MAP_WIDTH if state.map_width > 0 else x


def normalize_z(z):
    return (z / state.map_height) * config.STANDARD_MAP_HEIGHT if state.map_height > 0 else z


def denormalize_x(nx):
    return (nx / config.STANDARD_MAP_WIDTH) * state.map_width if state.map_width > 0 else nx


def denormalize_z(nz):
    return (nz / config.STANDARD_MAP_HEIGHT) * state.map_height if state.map_height > 0 else nz


def normalize_range(rng):
    if state.map_width > 0 and state.map_height > 0:
        scale = (config.STANDARD_MAP_WIDTH / state.map_width + config.STANDARD_MAP_HEIGHT / state.map_height) / 2.0
        return rng * scale
    return rng


def normalize_distance(dist):
    if state.map_width > 0 and state.map_height > 0:
        scale = (config.STANDARD_MAP_WIDTH / state.map_width + config.STANDARD_MAP_HEIGHT / state.map_height) / 2.0
        return dist * scale
    return dist


def normalize_y(y):
    if np.isfinite(state.map_height_min) and np.isfinite(state.map_height_max) and state.map_height_max > state.map_height_min:
        return (y - state.map_height_min) / (state.map_height_max - state.map_height_min) * config.STANDARD_MAP_Y
    return y


def denormalize_y(ny):
    if np.isfinite(state.map_height_min) and np.isfinite(state.map_height_max) and state.map_height_max > state.map_height_min:
        return state.map_height_min + (ny / config.STANDARD_MAP_Y) * (state.map_height_max - state.map_height_min)
    return ny


def build_normalized_height_map():
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
    if state.normalized_map_heights is None:
        return -1000.0

    map_x = int(max(0, min(config.STANDARD_MAP_WIDTH - 1, nx)))
    map_z = int(max(0, min(config.STANDARD_MAP_HEIGHT - 1, nz)))
    return float(state.normalized_map_heights[map_z, map_x])


def get_cached_map_embedding(agent, device):
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


def reconstruct_state_with_map(agent, state_no_map):
    split_idx = config.SELF_EMBED_SIZE + config.MASS_EMBED_SIZE
    device = state_no_map.device
    map_emb = get_cached_map_embedding(agent, device)
    prefix = state_no_map[:split_idx]
    suffix = state_no_map[split_idx:]
    return torch.cat([prefix, map_emb, suffix], dim=0)


def normalize_image(img_array):
    if img_array.size == 0:
        return img_array
    min_val = np.min(img_array)
    max_val = np.max(img_array)
    if max_val - min_val == 0:
        return np.zeros_like(img_array, dtype=np.float32)
    return ((img_array - min_val) / (max_val - min_val)).astype(np.float32)


def generate_enemy_range_image(enemy_units, map_w, map_h, map_heights_shape, enemy_range=300):
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
        r = int(enemy_range * range_scale)
        x_min = max(0, ex - r)
        x_max = min(w - 1, ex + r)
        z_min = max(0, ez - r)
        z_max = min(h - 1, ez + r)
        for z in range(z_min, z_max + 1):
            for x in range(x_min, x_max + 1):
                if (x - ex) ** 2 + (z - ez) ** 2 <= r ** 2:
                    img[z, x] = 1.0
    return img


def sample_path_values(grid, start_x, start_z, end_x, end_z, sample_count=None):
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

    excess_cost = np.maximum(path_costs - 1.0, 0.0)
    if excess_cost.size == 0:
        return 0.0

    return -float(excess_cost.mean() * config.PATH_TERRAIN_WEIGHT)


def get_local_view_image(unit_x, unit_z, map_heights_array, view_size=250):
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

    unit_local_x = min(half, max(0, int(normalize_x(unit_x) - x_min)))
    unit_local_z = min(half, max(0, int(normalize_z(unit_z) - z_min)))
    if 0 <= unit_local_z < view_size and 0 <= unit_local_x < view_size:
        padded[unit_local_z, unit_local_x] = np.max(padded) + 10.0

    for dx in np.linspace(-200, 200, num=10):
        for dz in np.linspace(-200, 200, num=10):
            tx = normalize_x(unit_x) + dx
            tz = normalize_z(unit_z) + dz
            tx = max(0, min(config.STANDARD_MAP_WIDTH, tx))
            tz = max(0, min(config.STANDARD_MAP_HEIGHT, tz))
            sense_local_x = min(half, max(0, int(tx - x_min)))
            sense_local_z = min(half, max(0, int(tz - z_min)))
            if 0 <= sense_local_z < view_size and 0 <= sense_local_x < view_size:
                padded[sense_local_z, sense_local_x] = np.max(padded) + 1.0

    # for radius in [100, 200]:
    #     step_count = 6 if radius == 100 else 12
    #     for angle in np.linspace(0, 2 * np.pi, num=step_count, endpoint=False):
    #         sense_x = normalize_x(unit_x) + (radius / state.map_width * config.STANDARD_MAP_WIDTH if state.map_width > 0 else radius) * np.cos(angle)
    #         sense_z = normalize_z(unit_z) + (radius / state.map_height * config.STANDARD_MAP_HEIGHT if state.map_height > 0 else radius) * np.sin(angle)
    #         sense_local_x = min(half, max(0, int(sense_x - x_min)))
    #         sense_local_z = min(half, max(0, int(sense_z - z_min)))
    #         if 0 <= sense_local_z < view_size and 0 <= sense_local_x < view_size:
    #             padded[sense_local_z, sense_local_x] = np.max(padded) + 5.0

    return padded


def _get_map_signature():
    if state.map_heights is None:
        return None

    hasher = hashlib.sha256()

    heights = np.ascontiguousarray(state.map_heights, dtype=np.float32)
    hasher.update(heights.tobytes())

    mass_spots = np.ascontiguousarray(state.mass_spots, dtype=np.float32) if state.mass_spots else np.empty((0, 2), dtype=np.float32)
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
    map_signature = _get_map_signature()
    if map_signature is None:
        return None

    cache_dir = Path(config.MAP_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{map_signature}.npz"


def load_cached_cost_fields():
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
        current_spots = np.array(state.map_spots_norm, dtype=np.float32) if state.map_spots_norm else np.empty((0, 2), dtype=np.float32)

        if cached_spots.shape != current_spots.shape:
            return False
        if cached_spots.size > 0 and not np.allclose(cached_spots, current_spots, atol=1e-5):
            return False

        terrain_cost_map = np.array(data["terrain_cost_map"], dtype=np.float32, copy=False)
        mass_cost_stack = np.array(data["mass_cost_stack"], dtype=np.float32, copy=False)

        state.terrain_cost_map = terrain_cost_map
        state.mass_cost_fields = {}

        for idx, (mx, mz) in enumerate(state.map_spots_norm):
            state.mass_cost_fields[(mx, mz)] = mass_cost_stack[idx]

        print(f"Loaded cached map cost fields from {cache_path}")
        return True
    except Exception as exc:
        print(f"Failed to load map cache ({cache_path}): {exc}")
        return False


def save_cached_cost_fields():
    cache_path = _get_map_cache_path()
    if cache_path is None or state.terrain_cost_map is None:
        return False

    try:
        if state.map_spots_norm:
            mass_cost_stack = np.stack(
                [
                    np.array(
                        state.mass_cost_fields.get((mx, mz), np.full_like(state.terrain_cost_map, np.inf, dtype=np.float32)),
                        dtype=np.float32,
                        copy=False,
                    )
                    for mx, mz in state.map_spots_norm
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
    """Build a cost map where each cell represents traversal difficulty based on height gradients."""
    if state.normalized_map_heights is None:
        state.terrain_cost_map = None
        return
    
    h, w = state.normalized_map_heights.shape
    cost_map = np.ones((h, w), dtype=np.float32)
    
    # Compute height gradients (steepness)
    impassable_count = 0
    for z in range(h):
        for x in range(w):
            neighbors = []
            if z > 0:
                neighbors.append(state.normalized_map_heights[z-1, x])
            if z < h - 1:
                neighbors.append(state.normalized_map_heights[z+1, x])
            if x > 0:
                neighbors.append(state.normalized_map_heights[z, x-1])
            if x < w - 1:
                neighbors.append(state.normalized_map_heights[z, x+1])
            
            if neighbors:
                current_h = state.normalized_map_heights[z, x]
                max_delta = max(abs(current_h - n) for n in neighbors)
                
                # Mark as impassable if slope exceeds unit's climbing capability
                if max_delta > config.MAX_TRAVERSABLE_SLOPE:
                    cost_map[z, x] = np.inf
                    impassable_count += 1
                else:
                    # Cost increases with steepness: 1 + (height_delta / threshold)
                    cost_map[z, x] = 1.0 + (max_delta / config.PATH_SPIKE_THRESHOLD)
    
    state.terrain_cost_map = cost_map
    passable_pct = 100.0 * (1.0 - impassable_count / (h * w))
    print(f"Terrain cost map built: {w}x{h}, cost range [{np.min(cost_map[np.isfinite(cost_map)]):.2f}, {np.max(cost_map[np.isfinite(cost_map)]):.2f}]")
    print(f"  Impassable cells: {impassable_count} ({passable_pct:.1f}% passable)")


def build_mass_cost_fields():
    """Precompute cost fields from each mass point using Dijkstra."""
    if state.terrain_cost_map is None or not state.map_spots_norm:
        state.mass_cost_fields = {}
        return
    
    import heapq
    
    h, w = state.terrain_cost_map.shape
    state.mass_cost_fields = {}
    
    for mass_idx, (mx, mz) in enumerate(state.map_spots_norm):
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
            
            # Check 4-connected neighbors
            for dz, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nz, nx = z + dz, x + dx
                if 0 <= nz < h and 0 <= nx < w and (nz, nx) not in visited:
                    move_cost = state.terrain_cost_map[nz, nx]
                    new_cost = current_cost + move_cost
                    if new_cost < cost_field[nz, nx]:
                        cost_field[nz, nx] = new_cost
                        heapq.heappush(heap, (new_cost, nz, nx))
        
        state.mass_cost_fields[(mx, mz)] = cost_field
    
    print(f"Built {len(state.mass_cost_fields)} mass cost fields")


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
    """Query the precomputed cost from a mass spot to a unit position. Returns np.inf if unreachable."""
    if mass_spot not in state.mass_cost_fields:
        return np.inf  # No cost field means unreachable
    
    cost_field = state.mass_cost_fields[mass_spot]
    h, w = cost_field.shape
    
    grid_x = int(np.clip(unit_nx, 0, w - 1))
    grid_z = int(np.clip(unit_nz, 0, h - 1))
    
    cost = cost_field[grid_z, grid_x]
    return float(cost)  # May be inf if unreachable


def extract_terrain_waypoints(mass_spot, unit_nx, unit_nz, count=None, search_radius=None):
    """Extract waypoint candidates from cost field that follow terrain-aware gradient descent."""
    if mass_spot not in state.mass_cost_fields:
        return []
    
    cost_field = state.mass_cost_fields[mass_spot]
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
