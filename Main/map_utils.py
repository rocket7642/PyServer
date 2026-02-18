import numpy as np
import torch

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
