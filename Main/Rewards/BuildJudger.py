import numpy as np
import config
import map_utils
import runtime_state as state

def compute_build_features(
    unit_nx, unit_nz, unit_ny,
    target_nx, target_nz,
    active_mass,
    enemy_units,
    friendly_units,
    is_noop
):
    """
    Compute features for evaluating a potential building placement location.
    
    Feature indices correspond to `config.NUM_BUILD_FEATURES`:
    0: friendly_proximity (Place near friendlies, away from enemies)
    1: is_building (noop - if we just want to build on the exact spot without moving ie finish the current building, this should be high)
    2: enemy_proximity (Don't build near enemies)
    3: mass_spot_proximity (Prefer building near mass spots)
    4: terrain_suitability (Prefer building on flatter terrain)
    """
    features = np.zeros(config.NUM_BUILD_FEATURES, dtype=np.float32)

    # Feature 0: friendly_proximity
    # The closer to friendly structures/units, the higher the score (up to a distance limit)
    min_f_dist = float('inf')
    for f_u in friendly_units:
        # Distance to the target build spot
        fnx = map_utils.normalize_x(f_u['x'])
        fnz = map_utils.normalize_z(f_u['z'])
        # Skip evaluating proximity to self exactly at the current position
        if abs(fnx - unit_nx) < 1e-4 and abs(fnz - unit_nz) < 1e-4:
            continue
            
        dist = ((fnx - target_nx)**2 + (fnz - target_nz)**2)**0.5
        if dist < min_f_dist:
            min_f_dist = dist
            
    # Max reward at 0 distance, linearly falls off to 0 at 150 units.
    if min_f_dist != float('inf'):
        features[0] = max(0.0, 1.0 - (min_f_dist / 150.0))

    # Feature 1: is_building (Noop Equivalent)
    features[1] = 1.0 if is_noop else 0.0

    # Feature 2: enemy_proximity
    # The closer to enemies, the higher the signal. The agent should learn a negative weight here.
    min_e_dist = float('inf')
    for e_u in enemy_units:
        enx = map_utils.normalize_x(e_u['x'])
        enz = map_utils.normalize_z(e_u['z'])
        dist = ((enx - target_nx)**2 + (enz - target_nz)**2)**0.5
        if dist < min_e_dist:
            min_e_dist = dist
            
    if min_e_dist != float('inf'):
        features[2] = max(0.0, 1.0 - (min_e_dist / config.ENEMY_PROXIMITY_THRESHOLD))

    # Feature 3: mass_spot_proximity
    # Prefer expanding towards mass extraction points
    min_m_dist = float('inf')
    if active_mass:
        for m in active_mass:
            if m is None:
                continue
            dist = ((m[0] - target_nx)**2 + (m[1] - target_nz)**2)**0.5
            if dist < min_m_dist:
                min_m_dist = dist
                
    if min_m_dist != float('inf'):
        # Falloff out to 200 units
        features[3] = max(0.0, 1.0 - (min_m_dist / 200.0))

    # Feature 4: terrain_suitability
    # Punish high ridges / slopes by checking the height difference between the target spot and the current location.
    # In a fully fleshed out engine, you might sample 4 corners of the building to find the strict slope.
    if state.normalized_map_heights is not None:
        target_h = map_utils.get_terrain_height_at(target_nx, target_nz, state.normalized_map_heights)
        unit_h = map_utils.get_terrain_height_at(unit_nx, unit_nz, state.normalized_map_heights)
        h_diff = abs(target_h - unit_h)
        features[4] = -min(2.0, h_diff)
    else:
        features[4] = 0.0

    return features
