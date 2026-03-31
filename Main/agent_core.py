import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config
import map_utils
import runtime_state as state
from Rewards import MoveJudger
from agent_model import RTSAgent

ENCODER_OUTPUT_SIZE = (
    config.SELF_EMBED_SIZE
    + config.MASS_EMBED_SIZE
    + config.MAP_EMBED_SIZE
    + config.FRIENDLY_EMBED_SIZE
    + config.ENEMY_EMBED_SIZE
)

agent = RTSAgent(input_size=ENCODER_OUTPUT_SIZE, num_features=config.NUM_ACTION_FEATURES)
# Lowered from 0.001 to 0.0001 for testing
# Added in weight decay for better generalization and to help prevent overfitting to the training data, which can be especially important given the complexity of the environment and the potential for noisy rewards
optimizer = optim.Adam(agent.parameters(), lr=0.0001, weight_decay=0.05) 
# criterion = nn.MSELoss()
criterion = nn.SmoothL1Loss() # swapped from MSELoss to SmoothL1Loss for better stability with TD targets

def init_lstm_hidden(batch_size=1):
    """Create zero-initialized LSTM hidden and cell states for the given batch size."""
    h0 = torch.zeros(config.LSTM_NUM_LAYERS, batch_size, config.LSTM_HIDDEN_SIZE)
    c0 = torch.zeros(config.LSTM_NUM_LAYERS, batch_size, config.LSTM_HIDDEN_SIZE)
    return (h0, c0)


def get_state(agent_unit, friendly_units, enemy_units):
    """Encode the full game state (including map embedding) into a feature vector for the given unit."""
    with torch.no_grad():
        return agent.encode_state(agent_unit, friendly_units, enemy_units).detach()


def get_state_no_map(agent_unit, friendly_units, enemy_units):
    """Encode the game state without the map embedding, used for lightweight storage in replay buffers."""
    with torch.no_grad():
        return agent.encode_state_no_map(agent_unit, friendly_units, enemy_units).detach()


def select_mass_destination(unit_id, unit_nx, unit_nz, unvisited_mass):
    """Select or maintain a target mass spot for a unit, swapping only if a significantly better option appears."""
    if not unvisited_mass:
        state.mass_destinations.pop(unit_id, None)
        state.mass_destination_distances.pop(unit_id, None)
        return None

    current_dest = state.mass_destinations.get(unit_id)
    if current_dest in unvisited_mass:
        current_dist_to_dest = MoveJudger.compute_mass_spot_score(unit_nx, unit_nz, current_dest)
        prev_dist_to_dest = state.mass_destination_distances.get(unit_id, current_dist_to_dest)
        
        best_spot, best_score = MoveJudger.select_best_mass_spot(
            unit_nx,
            unit_nz,
            unvisited_mass,
            return_score=True
        )
        
        if best_spot is not None and best_spot != current_dest:
            # best_dist = ((best_spot[0] - unit_nx) ** 2 + (best_spot[1] - unit_nz) ** 2) ** 0.5
            best_dist = MoveJudger.compute_mass_spot_score(unit_nx, unit_nz, best_spot)
            swap_threshold = prev_dist_to_dest * (1.0 - config.MASS_DESTINATION_SWAP_THRESHOLD)
            
            if best_dist < swap_threshold:
                state.mass_destinations[unit_id] = best_spot
                state.mass_destination_distances[unit_id] = best_dist
                state.writer.add_scalar('Mass_Destination/swapped', 1.0, state.step_counter)
                print(
                    f"Unit {unit_id} swapped destination: "
                    f"old_dist={prev_dist_to_dest:.2f}, "
                    f"new_dist={best_dist:.2f} (threshold {swap_threshold:.2f})"
                )
                return best_spot
        
        state.mass_destination_distances[unit_id] = current_dist_to_dest
        state.writer.add_scalar('Mass_Destination/swapped', 0.0, state.step_counter)
        return current_dest

    best_spot, best_score = MoveJudger.select_best_mass_spot(
        unit_nx,
        unit_nz,
        unvisited_mass,
        return_score=True
    )
    if best_spot is None:
        state.mass_destinations.pop(unit_id, None)
        state.mass_destination_distances.pop(unit_id, None)
        return None

    dest_dist = MoveJudger.compute_mass_spot_score(unit_nx, unit_nz, best_spot)
    state.mass_destinations[unit_id] = best_spot
    state.mass_destination_distances[unit_id] = dest_dist
    state.writer.add_scalar('Mass_Destination/terrain_score', best_score, state.step_counter)
    state.writer.add_scalar('Mass_Destination/x', best_spot[0], state.step_counter)
    state.writer.add_scalar('Mass_Destination/z', best_spot[1], state.step_counter)
    state.writer.add_scalar('Mass_Destination/swapped', 0.0, state.step_counter)
    top_candidates = MoveJudger.get_top_mass_spots(unit_nx, unit_nz, unvisited_mass, limit=4)
    for rank, (spot, score) in enumerate(top_candidates, start=1):
        dist = MoveJudger.compute_mass_spot_score(unit_nx, unit_nz, spot)
        state.writer.add_scalar(
            f"Mass_Destination/top_{rank}/score",
            score,
            state.step_counter
        )
        state.writer.add_scalar(
            f"Mass_Destination/top_{rank}/distance",
            dist,
            state.step_counter
        )
        # state.writer.add_scalar(
        #     f"Mass_Destination/top_{rank}/x",
        #     spot[0],
        #     state.step_counter
        # )
        # state.writer.add_scalar(
        #     f"Mass_Destination/top_{rank}/z",
        #     spot[1],
        #     state.step_counter
        # )
    print(
        f"Unit {unit_id} mass destination -> ({best_spot[0]:.2f}, {best_spot[1]:.2f}) "
        f"score={best_score:.2f}"
    )
    return best_spot


def _get_unit_speed_norm(unit_id):
    """Return the unit speed in normalized map units per second with safe fallbacks."""
    speed_world = None
    for fu in state.units:
        if fu.get('id') == unit_id:
            speed_world = fu.get('speed', None)
            break

    if speed_world is None:
        speed_world = config.DEFAULT_UNIT_SPEED

    speed_norm = map_utils.normalize_distance(float(speed_world))
    return max(speed_norm, config.MIN_EFFECTIVE_SPEED_NORM)


def _get_unit_health_context(unit_id):
    """Return (current_hp, max_hp) for a unit with safe fallbacks and running max tracking."""
    current_hp = None
    for fu in state.units:
        if fu.get('id') == unit_id:
            raw_health = fu.get('health', None)
            if raw_health is not None:
                current_hp = max(0.0, float(raw_health))
            break

    if current_hp is None:
        return None, None

    previous_max = state.unit_max_healths.get(unit_id, 0.0)
    max_hp = max(previous_max, current_hp)
    state.unit_max_healths[unit_id] = max_hp
    return current_hp, max_hp


def _compute_direct_approach_penalty(
    unit_nx,
    unit_nz,
    target_x,
    target_z,
    enemy_units,
    enemy_range_image,
    unit_speed_norm,
    unit_hp=None,
    unit_max_hp=None,
    bias=1.0,
):
    """Estimate expected incoming damage pressure along a path and convert it into a score penalty."""
    if not enemy_units:
        return 0.0

    move_distance = ((target_x - unit_nx) ** 2 + (target_z - unit_nz) ** 2) ** 0.5
    if move_distance <= 1e-6:
        return 0.0

    sample_count = max(
        config.PATH_SAMPLE_COUNT,
        int(move_distance / max(config.DIRECT_APPROACH_SAMPLE_SPACING, 1.0)) + 1,
    )
    xs = np.linspace(unit_nx, target_x, num=sample_count)
    zs = np.linspace(unit_nz, target_z, num=sample_count)

    total_dps_seconds = 0.0
    for enemy in enemy_units:
        ex = map_utils.normalize_x(enemy['x'])
        ez = map_utils.normalize_z(enemy['z'])
        enemy_range = map_utils.normalize_range(enemy.get('range', config.DEFAULT_ENEMY_RANGE))
        if enemy_range <= 0.0:
            continue

        dps = max(0.0, float(enemy.get('dps', 50.0)))
        if dps <= 0.0:
            continue

        dist = np.sqrt((xs - ex) ** 2 + (zs - ez) ** 2)
        inside_ratio = float(np.mean(dist <= enemy_range))
        if inside_ratio <= 0.0:
            continue

        inside_distance = move_distance * inside_ratio
        inside_seconds = inside_distance / max(unit_speed_norm, 1e-6)
        total_dps_seconds += dps * inside_seconds

    if total_dps_seconds <= 0.0:
        return 0.0

    danger_weight = 0.0
    if enemy_range_image is not None:
        path_values = map_utils.sample_path_values(
            enemy_range_image,
            unit_nx,
            unit_nz,
            target_x,
            target_z,
            sample_count=sample_count,
        )
        if path_values is not None and path_values.size > 0:
            danger_weight = float(np.mean(path_values))

    base_penalty = total_dps_seconds * config.DIRECT_APPROACH_DPS_SECONDS_SCALE
    penalty = base_penalty * (1.0 + danger_weight * config.DIRECT_APPROACH_DANGER_WEIGHT_SCALE)

    # Weight by survivability: paths that consume a large share of max/current HP
    # become much less attractive, with extra emphasis on potentially lethal paths.
    if unit_hp is not None and unit_max_hp is not None and unit_hp > 0.0 and unit_max_hp > 0.0:
        hp_fraction_of_max = total_dps_seconds / max(unit_max_hp, 1e-6)
        hp_fraction_of_current = total_dps_seconds / max(unit_hp, 1e-6)

        percent_scale = float(getattr(config, 'DIRECT_APPROACH_HP_PERCENT_SCALE', 1.5))
        lethal_bonus = float(getattr(config, 'DIRECT_APPROACH_LETHAL_BONUS', 1.0))
        overkill_scale = float(getattr(config, 'DIRECT_APPROACH_OVERKILL_SCALE', 0.75))

        hp_threat_multiplier = 1.0 + hp_fraction_of_current * percent_scale
        if hp_fraction_of_current >= 1.0:  # lethal
            hp_threat_multiplier += lethal_bonus
            hp_threat_multiplier += (hp_fraction_of_current - 1.0) * overkill_scale

        penalty *= hp_threat_multiplier
        penalty *= bias # This is used for certain moves to be less affected by danger such as dodges, escapes, strafes, etc

    return penalty


def _filter_mass_spots_by_threat(
    unit_id,
    unit_nx,
    unit_nz,
    unvisited_mass,
    enemy_units,
    enemy_range_image,
    unit_speed_norm,
    unit_hp=None,
    unit_max_hp=None,
):
    """Temporarily block dangerous mass spots using hysteresis and a cooldown window."""
    if not unvisited_mass:
        return []

    blocked_until_by_spot = state.mass_spot_blocked_until.setdefault(unit_id, {})
    unvisited_set = set(unvisited_mass)

    # Remove stale entries for already-visited or removed spots.
    for spot in list(blocked_until_by_spot.keys()):
        if spot not in unvisited_set:
            blocked_until_by_spot.pop(spot, None)

    allowed_spots = []
    blocked_count = 0

    for spot in unvisited_mass:
        risk_score = _compute_direct_approach_penalty(
            unit_nx,
            unit_nz,
            spot[0],
            spot[1],
            enemy_units,
            enemy_range_image,
            unit_speed_norm,
            unit_hp=unit_hp,
            unit_max_hp=unit_max_hp,
            bias=1.0
        )

        unblock_step = blocked_until_by_spot.get(spot, -1)
        currently_blocked = state.step_counter < unblock_step

        if currently_blocked:
            if risk_score <= config.MASS_SPOT_UNBLOCK_RISK_THRESHOLD:
                blocked_until_by_spot.pop(spot, None)
                allowed_spots.append(spot)
            else:
                blocked_count += 1
        else:
            if risk_score >= config.MASS_SPOT_BLOCK_RISK_THRESHOLD:
                blocked_until_by_spot[spot] = state.step_counter + config.MASS_SPOT_BLOCK_COOLDOWN_STEPS
                blocked_count += 1
            else:
                allowed_spots.append(spot)

    if unvisited_mass:
        blocked_ratio = blocked_count / float(len(unvisited_mass))
        state.writer.add_scalar('Mass_Destination/blocked_spots', blocked_count, state.step_counter)
        state.writer.add_scalar('Mass_Destination/blocked_ratio', blocked_ratio, state.step_counter)

    return allowed_spots


def get_action(state_vec, unit_x, unit_z, unit_y, unit_id):
    """Run the agent's policy to select the best action and move target for a unit given its encoded state."""
    with torch.no_grad():
        hidden = state.lstm_hidden_states.get(unit_id, init_lstm_hidden())
        state.previous_lstm_hidden_states[unit_id] = (hidden[0].detach(), hidden[1].detach())

        input_seq = state_vec.unsqueeze(0).unsqueeze(0)
        feature_weights, new_hidden = agent(input_seq, hidden)
        feature_weights = feature_weights.squeeze(0)

        state.lstm_hidden_states[unit_id] = (new_hidden[0].detach(), new_hidden[1].detach())

        print(f"\nFeature weights for unit {unit_id}:")
        for name, weight in zip(config.FEATURE_NAMES, feature_weights):
            print(f"  {name}: {weight.item():.4f}")
            state.writer.add_scalar(f"Feature_Weights/{name}", weight.item(), state.step_counter)

        unit_nx = map_utils.normalize_x(unit_x)
        unit_nz = map_utils.normalize_z(unit_z)
        unit_ny = map_utils.normalize_y(unit_y)

        # Combine all known enemies for feature computation
        all_enemies = list(state.eUnits)
        for u in state.eKUnits:
            if all(u['id'] != eu['id'] for eu in all_enemies):
                all_enemies.append(u)

        enemy_range_image = map_utils.generate_enemy_range_image(
            all_enemies,
            state.map_width,
            state.map_height,
            state.normalized_map_heights.shape if state.normalized_map_heights is not None else None
        )

        unit_speed_norm = _get_unit_speed_norm(unit_id)
        unit_hp, unit_max_hp = _get_unit_health_context(unit_id)
        unvisited_mass = [p for p in state.map_spots_norm if p not in state.visited_mass_spots_norm]
        available_mass = _filter_mass_spots_by_threat(
            unit_id,
            unit_nx,
            unit_nz,
            unvisited_mass,
            all_enemies,
            enemy_range_image,
            unit_speed_norm,
            unit_hp=unit_hp,
            unit_max_hp=unit_max_hp,
        )

        mass_destination = select_mass_destination(unit_id, unit_nx, unit_nz, available_mass)
        active_mass = [mass_destination] if mass_destination is not None else available_mass

        candidates = []

        for dx in np.linspace(-200, 200, num=10):
            for dz in np.linspace(-200, 200, num=10):
                tx = unit_nx + dx
                tz = unit_nz + dz
                tx = max(0, min(config.STANDARD_MAP_WIDTH, tx))
                tz = max(0, min(config.STANDARD_MAP_HEIGHT, tz))
                # Only add reachable candidates
                if map_utils.is_position_reachable(tx, tz):
                    candidates.append((tx, tz, 'grid'))

        # Current position is always valid
        candidates.append((unit_nx, unit_nz, 'noop'))

        # Generate escape candidates pointing away from nearby enemies
        # Also add lateral dodge candidates for projectile/missile enemies
        escape_dx, escape_dz = map_utils.compute_enemy_escape_direction(unit_nx, unit_nz, all_enemies)
        if abs(escape_dx) > 1e-6 or abs(escape_dz) > 1e-6:
            for dist_mult in [0.5, 1.0, 1.5]:
                esc_dist = config.ESCAPE_CANDIDATE_DISTANCE * dist_mult
                for angle_offset in np.linspace(-0.5, 0.5, config.ESCAPE_CANDIDATE_COUNT):
                    import math
                    base_angle = math.atan2(escape_dz, escape_dx)
                    angle = base_angle + angle_offset * math.pi
                    tx = unit_nx + math.cos(angle) * esc_dist
                    tz = unit_nz + math.sin(angle) * esc_dist
                    tx = max(0, min(config.STANDARD_MAP_WIDTH, tx))
                    tz = max(0, min(config.STANDARD_MAP_HEIGHT, tz))

                    outsideEnemyRange = True
                    for enemy in all_enemies:
                        ex = map_utils.normalize_x(enemy['x'])
                        ez = map_utils.normalize_z(enemy['z'])
                        enemy_range = map_utils.normalize_range(enemy.get('range', config.DEFAULT_ENEMY_RANGE))
                        dist_to_enemy = ((tx - ex) ** 2 + (tz - ez) ** 2) ** 0.5
                        if dist_to_enemy <= enemy_range:
                            outsideEnemyRange = False
                            break

                    # Verify if the canidate is reachable and outside the enemy range before adding
                    if map_utils.is_position_reachable(tx, tz) and outsideEnemyRange:
                        candidates.append((tx, tz, 'escape'))

        # Lateral dodge candidates perpendicular to incoming fire from projectile/missile enemies
        import math
        for eu in all_enemies:
            wtype = eu.get('weapon_type', 'projectile')
            if wtype in ('projectile', 'missile'):
                eu_nx = map_utils.normalize_x(eu['x'])
                eu_nz = map_utils.normalize_z(eu['z'])
                fire_dx = unit_nx - eu_nx
                fire_dz = unit_nz - eu_nz
                fire_mag = (fire_dx ** 2 + fire_dz ** 2) ** 0.5
                if fire_mag > 1e-6:
                    # Perpendicular directions (90 degrees to fire line)
                    perp_dx = -fire_dz / fire_mag
                    perp_dz = fire_dx / fire_mag
                    for sign in [1.0, -1.0]:
                        for dist_mult in [0.5, 1.0]:
                            d = config.ESCAPE_CANDIDATE_DISTANCE * dist_mult
                            tx = unit_nx + sign * perp_dx * d
                            tz = unit_nz + sign * perp_dz * d
                            tx = max(0, min(config.STANDARD_MAP_WIDTH, tx))
                            tz = max(0, min(config.STANDARD_MAP_HEIGHT, tz))

                            further_from_enemy = True
                            if fire_mag < (tx - eu_nx ** 2 + tz - eu_nz ** 2) ** 0.5:
                                further_from_enemy = False

                            if map_utils.is_position_reachable(tx, tz) and further_from_enemy:
                                candidates.append((tx, tz, 'strafe'))

        # Short-range scatter candidates for throwing off predictive projectiles.
        # These stay close to the current position to avoid large path deviations.
        for enemy in all_enemies:
            wtype = enemy.get('weapon_type', 'projectile')
            if wtype in ('projectile', 'missile'):
                ex = map_utils.normalize_x(enemy['x'])
                ez = map_utils.normalize_z(enemy['z'])
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
                                if map_utils.is_position_reachable(tx, tz):
                                    candidates.append((tx, tz, 'dodge'))

        if mass_destination is not None:
            dest_world_x = map_utils.denormalize_x(mass_destination[0])
            dest_world_z = map_utils.denormalize_z(mass_destination[1])
            dist_to_dest = ((dest_world_x - unit_x) ** 2 + (dest_world_z - unit_z) ** 2) ** 0.5
            # Only add mass destination if reachable and within approach radius
            if dist_to_dest <= config.MASS_FINAL_APPROACH_RADIUS and map_utils.is_position_reachable(mass_destination[0], mass_destination[1]):
                candidates.append((mass_destination[0], mass_destination[1], 'mass_destination'))
            
            # Extract terrain-guided waypoints from cost field
            terrain_waypoints = map_utils.extract_terrain_waypoints(
                mass_destination,
                unit_nx,
                unit_nz,
                count=config.TERRAIN_WAYPOINT_COUNT,
                search_radius=config.TERRAIN_WAYPOINT_SEARCH_RADIUS
            )
            candidates.extend((wx, wz, 'terrain_waypoint') for wx, wz in terrain_waypoints)
            if terrain_waypoints:
                state.writer.add_scalar(
                    'Action_Selection/terrain_waypoints_generated',
                    len(terrain_waypoints),
                    state.step_counter
                )

        action_scores = []
        best_score = -float('inf')
        best_target = (unit_nx, unit_nz)
        chosen_action_features = [0.0] * config.NUM_ACTION_FEATURES

        noop_score = None
        move_scores = []
        direct_approach_penalties = []

        for (tx, tz, candidate_kind) in candidates:
            try:
                if abs(tx - unit_nx) < 1e-3 and abs(tz - unit_nz) < 1e-3:
                    features = MoveJudger.compute_action_features(
                        config.NOOP_ACTION,
                        unit_nx,
                        unit_nz,
                        unit_ny,
                        active_mass,
                        enemy_range_image=enemy_range_image,
                        enemy_units=all_enemies,
                    )
                else:
                    features = MoveJudger.compute_action_features(
                        "MOVE",
                        unit_nx,
                        unit_nz,
                        unit_ny,
                        active_mass,
                        tx,
                        tz,
                        enemy_range_image=enemy_range_image,
                        enemy_units=all_enemies,
                    )

                if features is None:
                    features = [0.0] * config.NUM_ACTION_FEATURES

                features_tensor = torch.tensor(features, dtype=torch.float32)
                score = torch.dot(feature_weights, features_tensor).item()

                if candidate_kind not in ('noop', 'escape', 'dodge', 'strafe'):
                    direct_penalty = _compute_direct_approach_penalty(
                        unit_nx,
                        unit_nz,
                        tx,
                        tz,
                        all_enemies,
                        enemy_range_image,
                        unit_speed_norm,
                        unit_hp=unit_hp,
                        unit_max_hp=unit_max_hp,
                        bias=1.0,  
                    )
                    score -= direct_penalty
                    direct_approach_penalties.append(direct_penalty)
                elif candidate_kind in ('dodge', 'strafe'):
                    # Apply a different bias for dodge and strafe actions
                    direct_penalty = _compute_direct_approach_penalty(
                        unit_nx,
                        unit_nz,
                        tx,
                        tz,
                        all_enemies,
                        enemy_range_image,
                        unit_speed_norm,
                        unit_hp=unit_hp,
                        unit_max_hp=unit_max_hp,
                        bias=0.5,  # Reduce the impact of direct approach penalties for these actions (safer and required, but still should be avoided if it means death)
                    )
                    score -= direct_penalty
                    direct_approach_penalties.append(direct_penalty)


                action_scores.append(score)

                if abs(tx - unit_nx) < 1e-3 and abs(tz - unit_nz) < 1e-3:
                    noop_score = score
                else:
                    move_scores.append(score)

                if score > best_score:
                    best_score = score
                    best_target = (tx, tz)
                    chosen_action_features = features
            except Exception as exc:
                print(f"Error computing action features: {exc}")
                continue

        if move_scores:
            print(
                f"Unit {unit_id}: NOOP score={noop_score:.3f}, "
                f"MOVE scores: min={min(move_scores):.3f}, max={max(move_scores):.3f}, "
                f"mean={np.mean(move_scores):.3f}"
            )

        if direct_approach_penalties:
            state.writer.add_scalar(
                'Action_Selection/direct_approach_penalty_mean',
                float(np.mean(direct_approach_penalties)),
                state.step_counter,
            )
            state.writer.add_scalar(
                'Action_Selection/direct_approach_penalty_max',
                float(np.max(direct_approach_penalties)),
                state.step_counter,
            )

        is_noop = (abs(best_target[0] - unit_nx) < 1e-3 and abs(best_target[1] - unit_nz) < 1e-3)
        best_action = config.NOOP_ACTION if is_noop else "MOVE"
        if is_noop:
            noop_features = MoveJudger.compute_action_features(
                config.NOOP_ACTION,
                unit_nx,
                unit_nz,
                unit_ny,
                unvisited_mass,
                enemy_range_image=enemy_range_image,
                enemy_units=all_enemies,
            )
            if noop_features is not None:
                chosen_action_features = noop_features

        state.writer.add_scalar('Action_Selection/best_score', best_score, state.step_counter)
        state.writer.add_scalar('Action_Selection/mean_score', np.mean(action_scores), state.step_counter)
        state.writer.add_scalar('Action_Selection/std_score', np.std(action_scores), state.step_counter)
        state.writer.add_scalar('Action_Selection/chosen_action', 1 if best_action == "MOVE" else 0, state.step_counter)
        state.writer.add_scalar('Action_Selection/is_noop', 1.0 if best_action == config.NOOP_ACTION else 0.0, state.step_counter)

        for name, feature_val in zip(config.FEATURE_NAMES, chosen_action_features):
            state.writer.add_scalar(f"Chosen_Action_Features/{name}", feature_val, state.step_counter)

        if state.step_counter % 100 == 0:
            state.writer.add_histogram('Action_Scores/distribution', np.array(action_scores), state.step_counter)

        return best_action, best_target, best_score


def train_agent(
    state_vec,
    action,
    reward,
    next_state,
    done,
    unit_x,
    unit_z,
    unit_y,
    next_unit_x,
    next_unit_z,
    next_unit_y,
    target_x=None,
    target_z=None,
    mass_destination=None,
    unit_id=None,
):
    """Perform a single TD (temporal difference) training step using the transition data and clamped Q-targets."""
    if unit_id is not None:
        hidden = state.previous_lstm_hidden_states.get(unit_id, init_lstm_hidden())
    else:
        hidden = init_lstm_hidden()
    input_seq = state_vec.unsqueeze(0).unsqueeze(0)
    current_weights, _ = agent(input_seq, hidden)
    current_weights = current_weights.squeeze(0)

    unvisited_mass = [p for p in state.map_spots_norm if p not in state.visited_mass_spots_norm]
    active_mass = [mass_destination] if mass_destination is not None else unvisited_mass
    unit_nx = map_utils.normalize_x(unit_x)
    unit_nz = map_utils.normalize_z(unit_z)
    unit_ny = map_utils.normalize_y(unit_y)
    target_nx = map_utils.normalize_x(target_x) if target_x is not None else unit_nx
    target_nz = map_utils.normalize_z(target_z) if target_z is not None else unit_nz

    # Combine all known enemies for feature computation
    all_enemies = list(state.eUnits)
    for u in state.eKUnits:
        if all(u['id'] != eu['id'] for eu in all_enemies):
            all_enemies.append(u)

    enemy_range_image = map_utils.generate_enemy_range_image(
        state.eUnits,
        state.map_width,
        state.map_height,
        state.normalized_map_heights.shape if state.normalized_map_heights is not None else None
    )

    action_features = MoveJudger.compute_action_features(
        action,
        unit_nx,
        unit_nz,
        unit_ny,
        active_mass,
        target_nx,
        target_nz,
        enemy_range_image=enemy_range_image,
        enemy_units=all_enemies,
    )
    
    # Validate features don't contain inf/nan
    action_features = [np.clip(f, -1e6, 1e6) if not (np.isinf(f) or np.isnan(f)) else 0.0 for f in action_features]
    
    features_tensor = torch.tensor(action_features, dtype=torch.float32)

    current_q = torch.dot(current_weights, features_tensor)

    if not done:
        with torch.no_grad():
            next_hidden = state.lstm_hidden_states.get(unit_id, init_lstm_hidden())
            next_input_seq = next_state.unsqueeze(0).unsqueeze(0)
            next_weights, _ = agent(next_input_seq, next_hidden)
            next_weights = next_weights.squeeze(0)
            max_next_q = -float('inf')
            candidates = []
            for dx in np.linspace(-200, 200, num=10):
                for dz in np.linspace(-200, 200, num=10):
                    tx = unit_nx + dx
                    tz = unit_nz + dz
                    tx = max(0, min(config.STANDARD_MAP_WIDTH, tx))
                    tz = max(0, min(config.STANDARD_MAP_HEIGHT, tz))
                    # Only add reachable candidates
                    if map_utils.is_position_reachable(tx, tz):
                        candidates.append((tx, tz))
            # Current next position is always valid
            next_nx_pos = map_utils.normalize_x(next_unit_x)
            next_nz_pos = map_utils.normalize_z(next_unit_z)
            candidates.append((next_nx_pos, next_nz_pos))

            # Generate escape candidates for TD target calculation
            esc_dx, esc_dz = map_utils.compute_enemy_escape_direction(next_nx_pos, next_nz_pos, all_enemies)
            if abs(esc_dx) > 1e-6 or abs(esc_dz) > 1e-6:
                import math
                for dist_mult in [0.5, 1.0, 1.5]:
                    esc_dist = config.ESCAPE_CANDIDATE_DISTANCE * dist_mult
                    for angle_offset in np.linspace(-0.5, 0.5, config.ESCAPE_CANDIDATE_COUNT):
                        base_angle = math.atan2(esc_dz, esc_dx)
                        angle = base_angle + angle_offset * math.pi
                        tx = next_nx_pos + math.cos(angle) * esc_dist
                        tz = next_nz_pos + math.sin(angle) * esc_dist
                        tx = max(0, min(config.STANDARD_MAP_WIDTH, tx))
                        tz = max(0, min(config.STANDARD_MAP_HEIGHT, tz))
                        if map_utils.is_position_reachable(tx, tz):
                            candidates.append((tx, tz))

            # Lateral dodge candidates for TD target (projectile/missile enemies)
            for eu in all_enemies:
                wtype = eu.get('weapon_type', 'projectile')
                if wtype in ('projectile', 'missile'):
                    eu_nx = map_utils.normalize_x(eu['x'])
                    eu_nz = map_utils.normalize_z(eu['z'])
                    fire_dx = next_nx_pos - eu_nx
                    fire_dz = next_nz_pos - eu_nz
                    fire_mag = (fire_dx ** 2 + fire_dz ** 2) ** 0.5
                    if fire_mag > 1e-6:
                        perp_dx = -fire_dz / fire_mag
                        perp_dz = fire_dx / fire_mag
                        for sign in [1.0, -1.0]:
                            for dist_mult in [0.5, 1.0]:
                                d = config.ESCAPE_CANDIDATE_DISTANCE * dist_mult
                                tx = next_nx_pos + sign * perp_dx * d
                                tz = next_nz_pos + sign * perp_dz * d
                                tx = max(0, min(config.STANDARD_MAP_WIDTH, tx))
                                tz = max(0, min(config.STANDARD_MAP_HEIGHT, tz))
                                if map_utils.is_position_reachable(tx, tz):
                                    candidates.append((tx, tz))

            if mass_destination is not None:
                dest_world_x = map_utils.denormalize_x(mass_destination[0])
                dest_world_z = map_utils.denormalize_z(mass_destination[1])
                dist_to_dest = ((dest_world_x - next_unit_x) ** 2 + (dest_world_z - next_unit_z) ** 2) ** 0.5
                # Only add mass destination if reachable and within approach radius
                if dist_to_dest <= config.MASS_FINAL_APPROACH_RADIUS and map_utils.is_position_reachable(mass_destination[0], mass_destination[1]):
                    candidates.append(mass_destination)
                
                # Add terrain waypoints for TD target calculation
                next_nx_norm = map_utils.normalize_x(next_unit_x)
                next_nz_norm = map_utils.normalize_z(next_unit_z)
                terrain_waypoints = map_utils.extract_terrain_waypoints(
                    mass_destination,
                    next_nx_norm,
                    next_nz_norm,
                    count=config.TERRAIN_WAYPOINT_COUNT,
                    search_radius=config.TERRAIN_WAYPOINT_SEARCH_RADIUS
                )
                candidates.extend(terrain_waypoints)

            next_unvisited = [p for p in state.map_spots_norm if p not in state.visited_mass_spots_norm]
            next_active_mass = [mass_destination] if mass_destination is not None else next_unvisited
            next_enemy_range_image = map_utils.generate_enemy_range_image(
                state.eUnits,
                state.map_width,
                state.map_height,
                state.normalized_map_heights.shape if state.normalized_map_heights is not None else None
            )

            for (tx, tz) in candidates:
                next_nx = map_utils.normalize_x(next_unit_x)
                next_nz = map_utils.normalize_z(next_unit_z)
                next_ny = map_utils.normalize_y(next_unit_y)
                next_features = MoveJudger.compute_action_features(
                    "MOVE",
                    next_nx,
                    next_nz,
                    next_ny,
                    next_active_mass,
                    tx,
                    tz,
                    enemy_range_image=next_enemy_range_image,
                    enemy_units=all_enemies,
                )
                next_features_tensor = torch.tensor(next_features, dtype=torch.float32)
                next_q = torch.dot(next_weights, next_features_tensor)
                max_next_q = max(max_next_q, next_q.item())
    else:
        max_next_q = 0

    current_q_val = current_q.detach().item()
    target_raw = reward + 0.99 * max_next_q
    
    # Validate reward doesn't contain inf/nan
    if np.isinf(target_raw) or np.isnan(target_raw):
        print(f"[WARNING] Invalid target_raw: {target_raw} (reward={reward}, max_next_q={max_next_q})")
        target_raw = np.clip(target_raw, -1e6, 1e6)
    
    td_raw = np.clip(target_raw - current_q_val, -config.td_cap, config.td_cap)
    target_value = current_q_val + td_raw
    target_tensor = torch.tensor(target_value, dtype=torch.float32, device=current_q.device)

    td_error = abs(target_value - current_q_val)

    state.writer.add_scalar('Training/current_q_value', current_q.item(), state.step_counter)
    state.writer.add_scalar('Training/target_q_value', target_value, state.step_counter)
    state.writer.add_scalar('Training/max_next_q', max_next_q, state.step_counter)
    state.writer.add_scalar('Training/td_error', td_error, state.step_counter)

    loss = criterion(current_q, target_tensor)
    state.writer.add_scalar('Training/loss', loss.item(), state.step_counter)

    optimizer.zero_grad()
    torch.autograd.set_detect_anomaly(True)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)

    total_grad_norm = 0.0
    for param in agent.parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.norm().item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    state.writer.add_scalar('Training/gradient_norm', total_grad_norm, state.step_counter)

    optimizer.step()

    # Might be worth doing the standardization here to prevent certain negatives


def save_agent():
    """Save the agent's neural network weights to disk."""
    torch.save(agent.state_dict(), 'agent_weights_feature_based.pth')
    print("Feature-based agent weights saved.")


def _reset_agent_parameters(model: nn.Module):
    """Reinitialize all learnable parameters of the model, including CNN-specific weight init."""
    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    if hasattr(model, "_initialize_cnn_weights"):
        model._initialize_cnn_weights()


def load_agent():
    """Load saved agent weights from disk, reinitializing if weights are missing, incompatible, or contain NaN."""
    try:
        agent.load_state_dict(torch.load('agent_weights_feature_based.pth'))
        print("Feature-based agent weights loaded.")
        has_nan = any(torch.isnan(p).any().item() for p in agent.parameters())
        if has_nan:
            print("Loaded weights contain NaN. Reinitializing model weights.")
            _reset_agent_parameters(agent)
    except FileNotFoundError:
        print("No saved feature-based weights found, starting fresh.")
    except RuntimeError as exc:
        print("Saved weights are incompatible with the new LSTM architecture.")
        print(f"Details: {exc}")
        print("Starting with fresh weights.")
        _reset_agent_parameters(agent)
    log_model_graph_once()

def log_model_graph_once():
    """Log the model's computation graph to TensorBoard once for visualization."""
    if not getattr(config, 'ENABLE_MODEL_GRAPH_LOG', True):
        return
    if state.model_graph_logged:
        return

    try:
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, ENCODER_OUTPUT_SIZE)
            dummy_hidden = init_lstm_hidden(batch_size=1)
            state.writer.add_graph(agent, (dummy_input, dummy_hidden))
        state.model_graph_logged = True
        print("Model graph logged to TensorBoard (Graphs tab).")
    except Exception as exc:
        print(f"Unable to log model graph: {exc}")
