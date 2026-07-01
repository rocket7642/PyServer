import numpy as np
import config
import map_utils
import runtime_state as state

import unit_defs

def compute_build_features(
    unit_nx, unit_nz, unit_ny,
    target_nx, target_nz,
    active_mass,
    enemy_units,
    friendly_units,
    is_noop,
    vision_image=None,
    target_structure_name="armrad",
    unit_id=None
):
    """
    Compute features for evaluating a potential building placement location.
    
    Feature indices correspond to `config.NUM_BUILD_FEATURES`:
    0: friendly_proximity (Place near friendlies, away from enemies)
    1: is_building (noop - if we just want to build on the exact spot without moving ie finish the current building, this should be high)
    2: enemy_proximity (Don't build near enemies)
    3: mass_spot_proximity (Prefer building near mass spots)
    4: terrain_suitability (Prefer building on flatter terrain)
    5: blocking_proximity (Don't build on top of existing units/buildings)
    6: transit_progress (If already moving towards this build location, encourage completion)
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
            
    # Max reward at 0 distance, linearly falls off to 0 at 15% of the map width (which is 153.6 units on the normalized map)
    if min_f_dist != float('inf'):
        features[0] = max(0.0, 1.0 - (min_f_dist / float(config.STANDARD_MAP_WIDTH * 0.15)))  # Normalize by map size for consistency across maps

    # Feature 1: is_continuing_commitment (Noop Equivalent)
    prev_discrete = state.previous_discrete_actions.get(unit_id)
    committed_target_world = state.build_committed_target.get(unit_id)

    if committed_target_world is not None:
        committed_nx = map_utils.normalize_x(committed_target_world[0])
        committed_nz = map_utils.normalize_z(committed_target_world[1])
    else:
        committed_nx = committed_nz = None

    is_continuing = (
        prev_discrete == config.ACTION_BUILD
        and committed_nx is not None
        and abs(target_nx - committed_nx) < 1e-3
        and abs(target_nz - committed_nz) < 1e-3
    )
    features[1] = config.BUILD_CONTINUITY_BONUS if is_continuing else 0.0

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
        # Falloff out to 20% of the map width (204.8 units on the normalized map), which is a reasonable distance for a building to be considered "near" a mass spot
        features[3] = max(0.0, 1.0 - (min_m_dist / float(config.STANDARD_MAP_WIDTH * 0.20)))

    # Feature 4: terrain_suitability
    if state.normalized_map_heights is not None and state.map_width > 0 and state.map_height > 0:
        # Hard gate first: if the terrain cost map marks this footprint as impassable, 
        # immediately assign the worst score and skip the height sampling entirely.
        if not map_utils.is_position_buildable(target_nx, target_nz, target_structure_name):
            features[4] = -1.0
        else:
            # Convert footprint tile size to normalized map units,
            # using the same formula as blocking_proximity (feature 5) for consistency.
            size = unit_defs.get_unit_size(target_structure_name)
            tw_norm = (size["width"] * 16.0 / state.map_width) * config.STANDARD_MAP_WIDTH
            th_norm = (size["height"] * 16.0 / state.map_height) * config.STANDARD_MAP_HEIGHT

            # Sample the five footprint points: center + four corners.
            sample_points = [
                (target_nx,              target_nz),
                (target_nx - tw_norm/2,  target_nz - th_norm/2),
                (target_nx + tw_norm/2,  target_nz - th_norm/2),
                (target_nx - tw_norm/2,  target_nz + th_norm/2),
                (target_nx + tw_norm/2,  target_nz + th_norm/2),
            ]

            heights = [
                map_utils.height_at_normalized(px, pz)
                for px, pz in sample_points
            ]

            height_range = max(heights) - min(heights)
            features[4] = -min(1.0, height_range / config.STANDARD_MAP_Y)
    else:
        features[4] = 0.0

    # Feature 5: blocking_proximity
    # Overlap detection to avoid placing a structure on top of an existing unit/building.
    blocking_proximity = 0.0
    if state.map_width > 0 and state.map_height > 0:
        target_size = unit_defs.get_unit_size(target_structure_name)
        tw_norm = (target_size.get("width", 1) * 16.0 / state.map_width) * config.STANDARD_MAP_WIDTH
        th_norm = (target_size.get("height", 1) * 16.0 / state.map_height) * config.STANDARD_MAP_HEIGHT
        
        target_left = target_nx - tw_norm / 2.0
        target_right = target_nx + tw_norm / 2.0
        target_top = target_nz - th_norm / 2.0
        target_bottom = target_nz + th_norm / 2.0
        
        for u in friendly_units + enemy_units:
            ux = map_utils.normalize_x(u['x'])
            uz = map_utils.normalize_z(u['z'])

            uw = u.get("width", 1) * 16.0 / state.map_width * config.STANDARD_MAP_WIDTH
            uh = u.get("height", 1) * 16.0 / state.map_height * config.STANDARD_MAP_HEIGHT
            
            u_left = ux - uw / 2.0
            u_right = ux + uw / 2.0
            u_top = uz - uh / 2.0
            u_bottom = uz + uh / 2.0
            
            if (target_left < u_right) and (target_right > u_left) and (target_top < u_bottom) and (target_bottom > u_top):
                blocking_proximity = -20.0 # VeryHeavy negative signal for collision!
                break
                
    features[5] = blocking_proximity

    # Feature 6: transit_progress
    # If the unit is already in the process of moving towards this build location, provide a positive signal to encourage completion.
    steps_since_commit = (
        state.step_counter - state.build_committed_since_step[unit_id]
        if state.build_committed_since_step.get(unit_id) is not None
        else None
    )

    if is_continuing:
        original_dist = state.build_committed_distance.get(unit_id, None)
        current_dist = ((target_nx - unit_nx)**2 + (target_nz - unit_nz)**2)**0.5
        if original_dist is not None and original_dist > 1e-3:
            transit_progress = 1.0 - min(1.0, current_dist / original_dist)
        else:
            transit_progress = 0.0

        # If committed too long with no construction starting, apply a stall penalty
        if steps_since_commit is not None and steps_since_commit > config.BUILD_COMMIT_TIMEOUT_STEPS:
            transit_progress = config.BUILD_TRANSIT_STALL_PENALTY
    else:
        transit_progress = 0.0
        
    features[6] = transit_progress

    # Feature 7: prospective_vision_gain
    # Fraction of cells within this structure's radar radius that are currently
    # unknown (vision_image == 0). High value means building here reveals a lot.
    # This is the primary signal for "build in fog boundary before advancing."
    prospective_vision_gain = 0.0
    if vision_image is not None:
        radar_range = unit_defs.get_unit_ranges(target_structure_name).get('radar_range', 0)
        if radar_range > 0 and state.map_width > 0 and state.map_height > 0:
            range_scale = (config.STANDARD_MAP_WIDTH / state.map_width +
                        config.STANDARD_MAP_HEIGHT / state.map_height) / 2.0
            r_pixels = int(radar_range * range_scale)
            h, w = vision_image.shape
            cx, cz = int(np.clip(target_nx, 0, w - 1)), int(np.clip(target_nz, 0, h - 1))
            x_min = max(0, cx - r_pixels)
            x_max = min(w - 1, cx + r_pixels)
            z_min = max(0, cz - r_pixels)
            z_max = min(h - 1, cz + r_pixels)
            patch = vision_image[z_min:z_max+1, x_min:x_max+1]
            if patch.size > 0:
                # Fraction of cells in radar radius that are fully unknown
                prospective_vision_gain = float(np.mean(patch == 0))
    features[7] = prospective_vision_gain

    return features
