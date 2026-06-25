import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

import config
import map_utils
import runtime_state as state
from Rewards import MoveJudger
from Rewards import BuildJudger
from agent_model import RTSAgent

ENCODER_OUTPUT_SIZE = (
    config.SELF_EMBED_SIZE
    + config.ECO_EMBED_SIZE
    + config.MASS_EMBED_SIZE
    + config.MAP_EMBED_SIZE
    + config.FRIENDLY_EMBED_SIZE
    + config.ENEMY_EMBED_SIZE 
    + config.VISION_EMBED_SIZE
)

agent = RTSAgent(input_size=ENCODER_OUTPUT_SIZE)
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
                # state.writer.add_scalar('Mass_Destination/swapped', 1.0, state.step_counter)
                print(
                    f"Unit {unit_id} swapped destination: "
                    f"old_dist={prev_dist_to_dest:.2f}, "
                    f"new_dist={best_dist:.2f} (threshold {swap_threshold:.2f})"
                )
                return best_spot
        
        state.mass_destination_distances[unit_id] = current_dist_to_dest
        # state.writer.add_scalar('Mass_Destination/swapped', 0.0, state.step_counter)
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
    # state.writer.add_scalar('Mass_Destination/terrain_score', best_score, state.step_counter)
    # state.writer.add_scalar('Mass_Destination/x', best_spot[0], state.step_counter)
    # state.writer.add_scalar('Mass_Destination/z', best_spot[1], state.step_counter)
    # state.writer.add_scalar('Mass_Destination/swapped', 0.0, state.step_counter)
    top_candidates = MoveJudger.get_top_mass_spots(unit_nx, unit_nz, unvisited_mass, limit=4)
    for rank, (spot, score) in enumerate(top_candidates, start=1):
        dist = MoveJudger.compute_mass_spot_score(unit_nx, unit_nz, spot)
        # state.writer.add_scalar(
        #     f"Mass_Destination/top_{rank}/score",
        #     score,
        #     state.step_counter
        # )
        # state.writer.add_scalar(
        #     f"Mass_Destination/top_{rank}/distance",
        #     dist,
        #     state.step_counter
        # )
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


def _unit_name_for_id(unit_id):
    """Return the stable `name` for a given runtime `unit_id`, or None if unknown."""
    for fu in state.units:
        if fu.get('id') == unit_id:
            return fu.get('name')
    return None


def _resolve_template_key_for_unit(unit_id):
    """Return the key to use in `state.adaptive_candidate_templates` for this unit.

    Preference order:
    - If a template set already exists for the unit `name`, use that (stable).
    - If templates exist keyed by numeric `unit_id`, migrate them to `name` and return `name`.
    - Otherwise return the original `unit_id` (fallback).
    """
    name = _unit_name_for_id(unit_id)
    if name:
        if getattr(state, 'evalRun', False):
            if name in state.adaptive_candidate_templates:
                return name
            if unit_id in state.adaptive_candidate_templates:
                return unit_id
            sid = str(unit_id)
            if sid in state.adaptive_candidate_templates:
                return sid
            return name

        # If already present under name, prefer it
        if name in state.adaptive_candidate_templates:
            return name

        # Migrate templates stored under numeric id (or stringified id) to the stable name
        if unit_id in state.adaptive_candidate_templates:
            state.adaptive_candidate_templates[name] = state.adaptive_candidate_templates.pop(unit_id)
            return name
        sid = str(unit_id)
        if sid in state.adaptive_candidate_templates:
            state.adaptive_candidate_templates[name] = state.adaptive_candidate_templates.pop(sid)
            return name

        # No existing templates found; still use name as the preferred key for any new templates
        return name

    # No stable name available; fall back to numeric id key
    return unit_id


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
        # state.writer.add_scalar('Mass_Destination/blocked_spots', blocked_count, state.step_counter)
        state.writer.add_scalar('Mass_Destination/blocked_ratio', blocked_ratio, state.step_counter)

    return allowed_spots


def _inject_hazard_prediction_feature(features, hazard_penalty):
    """Write the hazard prediction value into the final feature slot as a bounded penalty."""
    if features is None:
        features = [0.0] * config.NUM_ACTION_FEATURES
    if len(features) < config.NUM_ACTION_FEATURES:
        features = features + [0.0] * (config.NUM_ACTION_FEATURES - len(features))
    # Store hazard as a normalized penalty so extreme threat estimates cannot turn into a reward spike.
    hazard_penalty = max(0.0, float(hazard_penalty))
    features[-1] = -hazard_penalty / (1.0 + hazard_penalty)
    return features


def _hazard_bias_for_candidate_kind(candidate_kind):
    """Return hazard penalty bias by candidate kind to keep scoring/training consistent."""
    if candidate_kind in ('noop', 'escape', None):
        return 0.0
    if candidate_kind in ('dodge', 'strafe'):
        return 0.5
    return 1.0


def _wrap_angle(angle):
    """Wrap angle to [-pi, pi] for stable polar-offset template storage."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _next_adaptive_template_id():
    """Allocate a unique integer ID for adaptive candidate templates."""
    next_id = state.adaptive_template_next_id
    state.adaptive_template_next_id += 1
    return next_id


def _nearest_enemy_anchor(unit_nx, unit_nz, enemy_units):
    """Return nearest enemy anchor as (x, z) in normalized space, or None."""
    nearest = map_utils.find_nearest_enemy(unit_nx, unit_nz, enemy_units) if enemy_units else None
    if nearest is None:
        return None
    return nearest[1], nearest[2]


def _build_base_candidate_lookup(candidates):
    """Group candidate positions by kind to support kind-anchored templates."""
    grouped = {}
    for tx, tz, kind in candidates:
        grouped.setdefault(kind, []).append((tx, tz))
    return grouped


def _resolve_template_anchor(template, unit_nx, unit_nz, mass_destination, enemy_units, base_by_kind):
    """Resolve template anchor point and base heading from current context."""
    anchor_type = template.get('anchor_type', 'unit')
    anchor_x, anchor_z = unit_nx, unit_nz
    heading = 0.0

    if anchor_type == 'enemy':
        enemy_anchor = _nearest_enemy_anchor(unit_nx, unit_nz, enemy_units)
        if enemy_anchor is None:
            return None
        anchor_x, anchor_z = enemy_anchor
        # Enemy-anchored heading points from enemy toward unit (escape frame).
        heading = np.arctan2(unit_nz - anchor_z, unit_nx - anchor_x)
    elif anchor_type == 'mass':
        if mass_destination is None:
            return None
        anchor_x, anchor_z = mass_destination
        heading = np.arctan2(anchor_z - unit_nz, anchor_x - unit_nx)
    elif anchor_type == 'kind':
        kind = template.get('anchor_kind')
        kind_points = base_by_kind.get(kind, []) if kind is not None else []
        if not kind_points:
            return None
        anchor_x = float(np.mean([p[0] for p in kind_points]))
        anchor_z = float(np.mean([p[1] for p in kind_points]))
        heading = np.arctan2(anchor_z - unit_nz, anchor_x - unit_nx)

    return anchor_x, anchor_z, heading


def _candidate_kind_to_anchor_type(candidate_kind):
    """Pick default anchor type for template promotion based on chosen candidate kind."""
    if candidate_kind in ('escape', 'strafe', 'dodge'):
        return 'enemy'
    if candidate_kind in ('mass_destination', 'terrain_waypoint'):
        return 'mass'
    return 'unit'


def _promote_candidate_template(
    unit_id,
    candidate_kind,
    tx,
    tz,
    score,
    unit_nx,
    unit_nz,
    mass_destination,
    enemy_units,
    base_by_kind,
):
    """Promote a useful selected candidate into a persistent context-relative template."""
    key = _resolve_template_key_for_unit(unit_id)
    templates = state.adaptive_candidate_templates.setdefault(key, [])
    anchor_type = _candidate_kind_to_anchor_type(candidate_kind)

    if anchor_type == 'enemy':
        enemy_anchor = _nearest_enemy_anchor(unit_nx, unit_nz, enemy_units)
        if enemy_anchor is None:
            anchor_type = 'unit'
    if anchor_type == 'mass' and mass_destination is None:
        anchor_type = 'unit'

    anchor_kind = candidate_kind if anchor_type == 'kind' else None
    temp_template = {
        'anchor_type': anchor_type,
        'anchor_kind': anchor_kind,
    }
    resolved = _resolve_template_anchor(
        temp_template,
        unit_nx,
        unit_nz,
        mass_destination,
        enemy_units,
        base_by_kind,
    )
    if resolved is None:
        return

    anchor_x, anchor_z, heading = resolved
    rel_dx = tx - anchor_x
    rel_dz = tz - anchor_z
    radius = float((rel_dx ** 2 + rel_dz ** 2) ** 0.5)
    if radius < config.ADAPTIVE_MIN_CANDIDATE_DISTANCE:
        return

    target_angle = np.arctan2(rel_dz, rel_dx)
    offset_angle = _wrap_angle(target_angle - heading)

    templates.append({
        'id': _next_adaptive_template_id(),
        'anchor_type': anchor_type,
        'anchor_kind': anchor_kind,
        'kind_hint': candidate_kind,
        'radius': float(np.clip(radius, config.ADAPTIVE_MIN_CANDIDATE_DISTANCE, config.ADAPTIVE_MAX_CANDIDATE_DISTANCE)),
        'offset_angle': float(offset_angle),
        'score_ema': float(score),
        'visits': 1,
    })


def _prune_adaptive_templates(unit_id):
    """Decay and prune stale/underperforming templates, then cap per-unit count."""
    key = _resolve_template_key_for_unit(unit_id)
    templates = state.adaptive_candidate_templates.get(key, [])
    if not templates:
        return

    kept = []
    for t in templates:
        t['score_ema'] = float(t.get('score_ema', 0.0)) * config.ADAPTIVE_SCORE_DECAY
        visits = int(t.get('visits', 0))
        if visits >= config.ADAPTIVE_PRUNE_MIN_VISITS and t['score_ema'] < config.ADAPTIVE_PRUNE_SCORE:
            continue
        kept.append(t)

    kept.sort(key=lambda x: x.get('score_ema', 0.0), reverse=True)
    state.adaptive_candidate_templates[key] = kept[: config.ADAPTIVE_MAX_TEMPLATES_PER_UNIT]


def _get_adaptive_mutation_schedule_values():
    """Return decayed adaptive mutation parameters based on step or epoch progress."""
    progress_source = str(getattr(config, 'ADAPTIVE_MUTATION_DECAY_SOURCE', 'step')).lower()
    if progress_source == 'epoch':
        progress = float(max(0, int(getattr(state, 'mass_cycle_completions', 0))))
    else:
        progress = float(max(0, int(getattr(state, 'step_counter', 0))))

    decay_rate = float(max(0.0, getattr(config, 'ADAPTIVE_MUTATION_DECAY_RATE', 0.0)))
    decay = float(np.exp(-decay_rate * progress)) if decay_rate > 0.0 else 1.0

    start_mutations = int(max(1, config.ADAPTIVE_MUTATIONS_PER_STEP))
    min_mutations = int(max(1, getattr(config, 'ADAPTIVE_MUTATIONS_PER_STEP_MIN', 1)))
    high_mutations = max(start_mutations, min_mutations)
    low_mutations = min(start_mutations, min_mutations)
    mutations_per_step = int(round(low_mutations + (high_mutations - low_mutations) * decay))

    start_distance_std = float(max(0.0, config.ADAPTIVE_MUTATION_DISTANCE_STD))
    min_distance_std = float(max(0.0, getattr(config, 'ADAPTIVE_MUTATION_DISTANCE_STD_MIN', 0.0)))
    high_distance_std = max(start_distance_std, min_distance_std)
    low_distance_std = min(start_distance_std, min_distance_std)
    mutation_distance_std = float(low_distance_std + (high_distance_std - low_distance_std) * decay)

    start_angle_std = float(max(0.0, config.ADAPTIVE_MUTATION_ANGLE_STD))
    min_angle_std = float(max(0.0, getattr(config, 'ADAPTIVE_MUTATION_ANGLE_STD_MIN', 0.0)))
    high_angle_std = max(start_angle_std, min_angle_std)
    low_angle_std = min(start_angle_std, min_angle_std)
    mutation_angle_std = float(low_angle_std + (high_angle_std - low_angle_std) * decay)

    return mutations_per_step, mutation_distance_std, mutation_angle_std


def _generate_adaptive_candidates(
    unit_id,
    unit_nx,
    unit_nz,
    mass_destination,
    enemy_units,
    base_candidates,
):
    """Generate mutated context-anchored candidates from persistent templates."""
    if not config.ADAPTIVE_CANDIDATES_ENABLED:
        return []
    if getattr(state, 'evalRun', False):
        return []

    key = _resolve_template_key_for_unit(unit_id)
    templates = state.adaptive_candidate_templates.get(key, [])
    if not templates:
        return []

    adaptive_candidates = []
    base_by_kind = _build_base_candidate_lookup(base_candidates)
    mutations_per_step, mutation_distance_std, mutation_angle_std = _get_adaptive_mutation_schedule_values()

    # Focus mutations on templates with stronger historical utility.
    order = sorted(templates, key=lambda t: t.get('score_ema', 0.0), reverse=True)
    for template in order[:mutations_per_step]:
        resolved = _resolve_template_anchor(
            template,
            unit_nx,
            unit_nz,
            mass_destination,
            enemy_units,
            base_by_kind,
        )
        if resolved is None:
            continue

        anchor_x, anchor_z, heading = resolved
        radius = float(template.get('radius', config.ESCAPE_CANDIDATE_DISTANCE))
        offset_angle = float(template.get('offset_angle', 0.0))

        mut_radius = radius + float(np.random.normal(0.0, mutation_distance_std))
        mut_radius = float(np.clip(mut_radius, config.ADAPTIVE_MIN_CANDIDATE_DISTANCE, config.ADAPTIVE_MAX_CANDIDATE_DISTANCE))
        mut_offset = offset_angle + float(np.random.normal(0.0, mutation_angle_std))
        world_angle = heading + mut_offset

        tx = anchor_x + np.cos(world_angle) * mut_radius
        tz = anchor_z + np.sin(world_angle) * mut_radius
        tx = float(np.clip(tx, 0.0, config.STANDARD_MAP_WIDTH))
        tz = float(np.clip(tz, 0.0, config.STANDARD_MAP_HEIGHT))
        if not map_utils.is_position_reachable(tx, tz):
            continue

        adaptive_candidates.append({
            'tx': tx,
            'tz': tz,
            'kind': 'adaptive',
            'template_id': template.get('id'),
            'template_kind_hint': template.get('kind_hint', 'adaptive'),
        })

    return adaptive_candidates


def _update_adaptive_template_feedback(
    unit_id,
    selected_kind,
    selected_tx,
    selected_tz,
    selected_score,
    selected_meta,
    unit_nx,
    unit_nz,
    mass_destination,
    enemy_units,
    base_candidates,
):
    """Update template statistics for selected candidate and optionally promote new templates."""
    if not config.ADAPTIVE_CANDIDATES_ENABLED:
        return
    if getattr(state, 'evalRun', False):
        return

    key = _resolve_template_key_for_unit(unit_id)
    templates = state.adaptive_candidate_templates.setdefault(key, [])
    template_id = selected_meta.get('template_id') if selected_meta else None
    if template_id is not None:
        for template in templates:
            if template.get('id') == template_id:
                visits = int(template.get('visits', 0)) + 1
                old_ema = float(template.get('score_ema', 0.0))
                alpha = 0.15
                template['score_ema'] = (1.0 - alpha) * old_ema + alpha * float(selected_score)
                template['visits'] = visits
                break
    else:
        if selected_kind != config.NOOP_ACTION and selected_score >= config.ADAPTIVE_PROMOTION_SCORE:
            base_by_kind = _build_base_candidate_lookup(base_candidates)
            _promote_candidate_template(
                unit_id,
                selected_kind,
                selected_tx,
                selected_tz,
                selected_score,
                unit_nx,
                unit_nz,
                mass_destination,
                enemy_units,
                base_by_kind,
            )

    _prune_adaptive_templates(unit_id)


def get_action(state_vec, unit_x, unit_z, unit_y, unit_id):
    """Run the agent's policy to select the best action and move target for a unit given its encoded state."""
    with torch.no_grad():
        hidden = state.lstm_hidden_states.get(unit_id, init_lstm_hidden())
        state.previous_lstm_hidden_states[unit_id] = (hidden[0].detach(), hidden[1].detach())

        input_seq = state_vec.unsqueeze(0).unsqueeze(0)
        action_logits, move_features, build_features, new_hidden = agent(input_seq, hidden)
        discrete_action = None

        steps_since_build = state.step_counter - state.last_build_step.get(unit_id, 0)
        if not getattr(state, 'evalRun', False) and steps_since_build >= config.FORCE_BUILD_EVERY_N_STEPS:
            discrete_action = config.ACTION_BUILD
            # state.last_build_step[unit_id] = state.step_counter

        # Testing statement, enforce build action
        # discrete_action = config.ACTION_BUILD
        
        # If it was the forced build step, skip choosing a new action to allow the build to go through, otherwise choose action as normal
        if not discrete_action:
            # Determine the action type (Move vs Build)
            # discrete_action = torch.argmax(action_logits, dim=-1).item()
            if not getattr(state, 'evalRun', False):
                epsilon = max(config.DISCRETE_EPSILON_MIN,
                            config.DISCRETE_EPSILON_START * (config.DISCRETE_EPSILON_DECAY ** state.step_counter))
                if random.random() < epsilon:
                    discrete_action = random.randint(0, config.NUM_DISCRETE_ACTIONS - 1)
                else:
                    discrete_action = torch.argmax(action_logits, dim=-1).item()

                state.writer.add_scalar('Action_Selection/epsilon',
                                epsilon if not getattr(state, 'evalRun', False) else 0.0,
                                state.step_counter)
            else:
                discrete_action = torch.argmax(action_logits, dim=-1).item()
        
        # Currently, all logic below is based on movement. We will output the move_features 
        # and handle the build features if the discrete_action is ACTION_BUILD.
        feature_weights = move_features.squeeze(0)
        build_weights = build_features.squeeze(0)

        state.lstm_hidden_states[unit_id] = (new_hidden[0].detach(), new_hidden[1].detach())

        print(f"\nMove feature weights for unit {unit_id}:")
        for name, weight in zip(config.FEATURE_NAMES, feature_weights):
            print(f"  {name}: {weight.item():.4f}")
            state.writer.add_scalar(f"Feature_Weights/{name}", weight.item(), state.step_counter)
        print(f"\nBuild feature weights for unit {unit_id}:")
        for name, weight in zip(config.BUILD_FEATURE_NAMES, build_weights):
            print(f"  {name}: {weight.item():.4f}")
            state.writer.add_scalar(f"Build_Weights/{name}", weight.item(), state.step_counter)

        # # Log hazard_prediction weight separately for easy monitoring
        # if len(feature_weights) > 0:
        #     hazard_weight = feature_weights[-1].item()
        #     state.writer.add_scalar("Feature_Weights/hazard_prediction_importance", hazard_weight, state.step_counter)

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

        vision_image = map_utils.generate_vision_image(
            state.units,
            state.map_width,
            state.map_height,
            state.normalized_map_heights
        )

        unit_speed_norm = _get_unit_speed_norm(unit_id)
        unit_hp, unit_max_hp = _get_unit_health_context(unit_id)
        unvisited_mass = [p for p in state.map_spots_norm if (p[0], p[1]) not in state.visited_mass_spots_norm]
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
        candidate_meta = []  

        # regardless of what the next action is, maintain the previously chosen candidate assuming there is one 
        if unit_id is not None:
            prev_candidate = state.previous_chosen_targets.get(unit_id)
            if prev_candidate is not None:
                prev_kind = prev_candidate[2] if len(prev_candidate) > 2 else 'previous'
                is_build_kind = prev_kind in ('build', 'vision_edge_build')
                if (discrete_action == config.ACTION_BUILD) == is_build_kind:
                    candidates.append(prev_candidate)
                    candidate_meta.append({'kind': 'previous'})

        if discrete_action == config.ACTION_MOVE:

            # Swapped from 10 candidates in both directions to 3 x 3 at 200 x 200
            for dx in np.linspace(-40, 40, num=3):
                for dz in np.linspace(-40, 40, num=3):
                    tx = unit_nx + dx
                    tz = unit_nz + dz
                    tx = max(0, min(config.STANDARD_MAP_WIDTH, tx))
                    tz = max(0, min(config.STANDARD_MAP_HEIGHT, tz))
                    # Only add reachable candidates
                    if map_utils.is_position_reachable(tx, tz):
                        candidates.append((tx, tz, 'grid'))
                        candidate_meta.append(None)

            # Current position is always valid
            candidates.append((unit_nx, unit_nz, 'noop'))
            candidate_meta.append(None)

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

                        # outsideEnemyRange = True
                        # for enemy in all_enemies:
                        #     ex = map_utils.normalize_x(enemy['x'])
                        #     ez = map_utils.normalize_z(enemy['z'])
                        #     enemy_range = map_utils.normalize_range(enemy.get('range', config.DEFAULT_ENEMY_RANGE))
                        #     dist_to_enemy = ((tx - ex) ** 2 + (tz - ez) ** 2) ** 0.5
                        #     if dist_to_enemy <= enemy_range:
                        #         outsideEnemyRange = False
                        #         break

                        # Verify if the canidate is reachable and outside the enemy range before adding
                        # TEMPORARY CHANGE, no longer has to be outside enemy range, just has to be reachable. The threat of being in range of an enemy is now handled by the hazard prediction feature and the model's learned weighting of it, allowing for more nuanced decisions about when to risk being in range for better positioning or mass gathering.
                        if map_utils.is_position_reachable(tx, tz): #and outsideEnemyRange:
                            candidates.append((tx, tz, 'escape'))
                            candidate_meta.append(None)

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

                                # further_from_enemy = True
                                # candidate_enemy_dist = ((tx - eu_nx) ** 2 + (tz - eu_nz) ** 2) ** 0.5
                                # if candidate_enemy_dist <= fire_mag:
                                #     further_from_enemy = False

                                # Same as above, no longer requiring the dodge candidate to be further from the enemy, just reachable, since the model can learn to weigh the hazard prediction feature to understand the risk of being in range and make more nuanced decisions.
                                if map_utils.is_position_reachable(tx, tz): #and further_from_enemy:
                                    candidates.append((tx, tz, 'strafe'))
                                    candidate_meta.append(None)

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
                                        candidate_meta.append(None)

            if mass_destination is not None:
                dest_world_x = map_utils.denormalize_x(mass_destination[0])
                dest_world_z = map_utils.denormalize_z(mass_destination[1])
                dist_to_dest = ((dest_world_x - unit_x) ** 2 + (dest_world_z - unit_z) ** 2) ** 0.5
                # Only add mass destination if reachable and within approach radius
                if dist_to_dest <= config.MASS_FINAL_APPROACH_RADIUS and map_utils.is_position_reachable(mass_destination[0], mass_destination[1]):
                    candidates.append((mass_destination[0], mass_destination[1], 'mass_destination'))
                    candidate_meta.append(None)
                
                # Extract terrain-guided waypoints from cost field
                terrain_waypoints = map_utils.extract_terrain_waypoints(
                    mass_destination,
                    unit_nx,
                    unit_nz,
                    count=config.TERRAIN_WAYPOINT_COUNT,
                    search_radius=config.TERRAIN_WAYPOINT_SEARCH_RADIUS
                )
                for wx, wz in terrain_waypoints:
                    candidates.append((wx, wz, 'terrain_waypoint'))
                    candidate_meta.append(None)
                # if terrain_waypoints:
                #     state.writer.add_scalar(
                #         'Action_Selection/terrain_waypoints_generated',
                #         len(terrain_waypoints),
                #         state.step_counter
                #     )

            adaptive_generated = _generate_adaptive_candidates(
                unit_id,
                unit_nx,
                unit_nz,
                mass_destination,
                all_enemies,
                candidates,
            )
            for adaptive in adaptive_generated:
                candidates.append((adaptive['tx'], adaptive['tz'], adaptive['kind']))
                candidate_meta.append(adaptive)

            state.writer.add_scalar('Action_Selection/adaptive_candidate_count', float(len(adaptive_generated)), state.step_counter)

        if discrete_action == config.ACTION_BUILD:
            # For building, we can consider a different set of candidates, such as nearby buildable locations or specific strategic points.
            # For simplicity, let's consider a small grid around the unit for potential build locations.
            for dx in np.linspace(-20, 20, num=3):
                for dz in np.linspace(-20, 20, num=3):
                    tx = unit_nx + dx
                    tz = unit_nz + dz
                    tx = max(0, min(config.STANDARD_MAP_WIDTH, tx))
                    tz = max(0, min(config.STANDARD_MAP_HEIGHT, tz))
                    dist = ((tx - unit_nx) ** 2 + (tz - unit_nz) ** 2) ** 0.5
                    if map_utils.is_position_buildable(tx, tz, "armrad") and dist <= config.BUILD_CANDIDATE_RADIUS:
                        candidates.append((tx, tz, 'build'))
                        candidate_meta.append(None)

            # Also generate a small circle of build candidates around mass points if they are nearby, as building near mass can be a common strategy.
            for mass in active_mass:
                for dx in np.linspace(-20, 20, num=3):
                    for dz in np.linspace(-20, 20, num=3):
                        tx = mass[0] + dx
                        tz = mass[1] + dz
                        tx = max(0, min(config.STANDARD_MAP_WIDTH, tx))
                        tz = max(0, min(config.STANDARD_MAP_HEIGHT, tz))
                        dist = ((tx - unit_nx) ** 2 + (tz - unit_nz) ** 2) ** 0.5
                        if map_utils.is_position_buildable(tx, tz, "armrad") and dist <= config.BUILD_CANDIDATE_RADIUS:
                            candidates.append((tx, tz, 'build'))
                            candidate_meta.append(None)

            # Maybe include some relating to flat terrain but generic flat terrain points might not be too useful
            # Include at edge of current radar vision as well, as expanding vision can be a key reason to build. (randomly choose like 10)
            # Gonna need to scan the vision image for this (1 LOS, 0.5 Radar, 0 unknown), look for points that are currently unknown but adjacent to known, as those are the ones that building could reveal. Could also weight them by how many unknown cells they would reveal in the vision image.
            # Go out in 24 directions around the unit until you hit a tile that is listed as unknown in the vision image, then add that as a candidate
            h_vis, w_vis = vision_image.shape
            max_ray_steps = max(h_vis, w_vis)
            num_rays = 24
            angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

            for angle in angles:
                # Generate all steps along this ray at once
                steps = np.arange(2, max_ray_steps)
                txs = (unit_nx + steps * np.cos(angle)).astype(int)
                tzs = (unit_nz + steps * np.sin(angle)).astype(int)

                # Clip and find valid (in-bounds) indices
                in_bounds = (txs >= 0) & (txs < w_vis) & (tzs >= 0) & (tzs < h_vis)
                if not np.any(in_bounds):
                    continue

                txs_valid = txs[in_bounds]
                tzs_valid = tzs[in_bounds]

                # Sample the vision image along the ray
                ray_values = vision_image[tzs_valid, txs_valid]

                # Find the first position that is NOT radar-covered (< 0.4 threshold)
                unknown_mask = ray_values < 0.4
                if not np.any(unknown_mask):
                    continue

                first_unknown = np.argmax(unknown_mask)
                tx = float(txs_valid[first_unknown])
                tz = float(tzs_valid[first_unknown])

                if map_utils.is_position_buildable(tx, tz, "armrad"):
                    candidates.append((tx, tz, 'vision_edge_build'))
                    candidate_meta.append(None)


        action_scores = []
        best_score = -float('inf')
        best_target = (unit_nx, unit_nz)
        best_candidate_kind = 'noop'
        best_candidate_meta = None
        chosen_action_features = [0.0] * config.NUM_ACTION_FEATURES

        noop_score = None
        move_scores = []
        build_scores = []
        direct_approach_penalties = []

        
        
        for idx, (tx, tz, candidate_kind) in enumerate(candidates):
            try:
                this_meta = candidate_meta[idx] if idx < len(candidate_meta) else None
                is_noop = (abs(tx - unit_nx) < 1e-3 and abs(tz - unit_nz) < 1e-3)
                
                if discrete_action == config.ACTION_MOVE:
                    # EVALUATE MOVEMENT
                    if is_noop:
                        features = MoveJudger.compute_action_features(
                            config.NOOP_ACTION,
                            unit_nx,
                            unit_nz,
                            unit_ny,
                            active_mass,
                            enemy_range_image=enemy_range_image,
                            enemy_units=all_enemies,
                            vision_image=vision_image
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
                            vision_image=vision_image
                        )

                    if features is None:
                        features = [0.0] * config.NUM_ACTION_FEATURES

                    direct_penalty = 0.0
                    hazard_bias = _hazard_bias_for_candidate_kind(candidate_kind)
                    if hazard_bias > 0.0 and not is_noop:
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
                            bias=hazard_bias,
                        )
                        direct_approach_penalties.append(direct_penalty)

                    features = _inject_hazard_prediction_feature(features, direct_penalty)
                    features_tensor = torch.tensor(features, dtype=torch.float32)
                    score = torch.dot(feature_weights, features_tensor).item()
                else:
                    # EVALUATE BUILDING
                    features = BuildJudger.compute_build_features(
                        unit_nx, unit_nz, unit_ny,
                        tx, tz,
                        active_mass,
                        all_enemies,
                        state.units, # friendly units
                        is_noop,
                        vision_image=vision_image,
                        target_structure_name="armrad",
                        unit_id=unit_id
                    )
                    
                    if features is None:
                        features = [0.0] * config.NUM_BUILD_FEATURES
                        
                    features_tensor = torch.tensor(features, dtype=torch.float32)
                    score = torch.dot(build_weights, features_tensor).item()
                
                # Need to calculate what type of enemy it is as retreat from a proj/missile will likely still hit if its a consistent movement.
                # enemy_type = None
                # for enemy in all_enemies:
                #     enemy_type = enemy.weapon_type

                #     # Once the type is determined, process existing candidates based on this
                #     break


                action_scores.append(score)
                if discrete_action == config.ACTION_BUILD:
                    build_scores.append(score)
                elif not is_noop:
                    move_scores.append(score)
                else:
                    noop_score = score

                if score > best_score:
                    best_score = score
                    best_target = (tx, tz)
                    best_candidate_kind = candidate_kind
                    best_candidate_meta = this_meta
                    chosen_action_features = features
            except Exception as exc:
                print(f"Error computing action features: {exc}")
                continue

        state.previous_chosen_targets[unit_id] = (best_target[0], best_target[1], best_candidate_kind)

        if discrete_action == config.ACTION_BUILD and build_scores:
            print(
                f"Unit {unit_id}: [BUILD] Candidates={len(build_scores)}, "
                f"BUILD scores: min={min(build_scores):.3f}, max={max(build_scores):.3f}, "
                f"mean={np.mean(build_scores):.3f}"
            )
        elif discrete_action == config.ACTION_MOVE and move_scores:
            print(
                f"Unit {unit_id}: [MOVE] NOOP={noop_score:.3f}, "
                f"Move scores: min={min(move_scores):.3f}, max={max(move_scores):.3f}, "
                f"mean={np.mean(move_scores):.3f}"
            )

        if move_scores:
            state.writer.add_scalar('Action_Selection/move_score_min', min(move_scores), state.step_counter)
            state.writer.add_scalar('Action_Selection/move_score_max', max(move_scores), state.step_counter)
            state.writer.add_scalar('Action_Selection/move_score_mean', np.mean(move_scores), state.step_counter)
        if build_scores:
            state.writer.add_scalar('Action_Selection/build_score_min', min(build_scores), state.step_counter)
            state.writer.add_scalar('Action_Selection/build_score_max', max(build_scores), state.step_counter)
            state.writer.add_scalar('Action_Selection/build_score_mean', np.mean(build_scores), state.step_counter)

        if noop_score is not None:
            state.writer.add_scalar('Action_Selection/noop_score', noop_score, state.step_counter)
        else:
            state.writer.add_scalar('Action_Selection/noop_score', 0.0, state.step_counter)

        state.writer.add_scalar('Action_Selection/discrete_action_chosen', 
                         float(discrete_action), state.step_counter)
        

        action_probs = torch.softmax(action_logits.squeeze(0), dim=-1).cpu().numpy()
        state.writer.add_scalar('Action_Selection/prob_move', action_probs[config.ACTION_MOVE], state.step_counter)
        state.writer.add_scalar('Action_Selection/prob_build', action_probs[config.ACTION_BUILD], state.step_counter)

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
                vision_image=vision_image
            )
            if noop_features is not None:
                chosen_action_features = noop_features

        # state.writer.add_scalar('Action_Selection/best_score', best_score, state.step_counter)
        # state.writer.add_scalar('Action_Selection/mean_score', np.mean(action_scores), state.step_counter)
        # state.writer.add_scalar('Action_Selection/std_score', np.std(action_scores), state.step_counter)

        _update_adaptive_template_feedback(
            unit_id,
            best_candidate_kind,
            best_target[0],
            best_target[1],
            best_score,
            best_candidate_meta,
            unit_nx,
            unit_nz,
            mass_destination,
            all_enemies,
            candidates,
        )
        template_key = _resolve_template_key_for_unit(unit_id)
        template_count = len(state.adaptive_candidate_templates.get(template_key, []))
        state.writer.add_scalar('Action_Selection/adaptive_template_count', float(template_count), state.step_counter)

        chosenActionVar = 0
        match best_candidate_kind:
            case 'adaptive':
                chosenActionVar = 0
            case 'escape':
                chosenActionVar = 1
            case 'strafe':
                chosenActionVar = 2
            case 'dodge':
                chosenActionVar = 3
            case 'mass_destination':
                chosenActionVar = 4
            case 'terrain_waypoint':
                chosenActionVar = 5
            case 'grid':
                chosenActionVar = 6
            case 'noop':
                chosenActionVar = 7
            case _:
                chosenActionVar = 8
        state.writer.add_scalar('Action_Selection/chosen_action', 
                                chosenActionVar,
                                  state.step_counter)

        # state.writer.add_scalar('Action_Selection/is_noop', 1.0 if best_action == config.NOOP_ACTION else 0.0, state.step_counter)

        if discrete_action == config.ACTION_BUILD:
            best_action = "BUILD"
            
            print(f"Building chosen by unit {unit_id}! Target picked: {best_target}")
            for name, weight in zip(config.BUILD_FEATURE_NAMES, build_weights):
                 state.writer.add_scalar(f"Feature_Weights/Build_{name}", weight.item(), state.step_counter)

            # We also want to log chosen_build_features for analytics, similar to move features
            for name, feature_val in zip(config.BUILD_FEATURE_NAMES, chosen_action_features):
                state.writer.add_scalar(f"Chosen_Action_Features/Build_{name}", feature_val, state.step_counter)
        else:
            for name, feature_val in zip(config.FEATURE_NAMES, chosen_action_features):
                state.writer.add_scalar(f"Chosen_Action_Features/{name}", feature_val, state.step_counter)

        if state.step_counter % 100 == 0:
            state.writer.add_histogram('Action_Scores/distribution', np.array(action_scores), state.step_counter)

        return best_action, best_target, best_score, best_candidate_kind, discrete_action


def train_agent(
    state_vec,
    discrete_action,
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
    action_kind=None,
    unit_id=None,
):
    """Perform a single TD (temporal difference) training step using the transition data and clamped Q-targets."""
    if getattr(state, 'evalRun', False):
        return

    if unit_id is not None:
        hidden = state.previous_lstm_hidden_states.get(unit_id, init_lstm_hidden())
    else:
        hidden = init_lstm_hidden()
    input_seq = state_vec.unsqueeze(0).unsqueeze(0)
    action_logits, move_weights, build_weights, _ = agent(input_seq, hidden)
    
    if discrete_action == config.ACTION_BUILD:
        current_weights = build_weights.squeeze(0)
    else:
        current_weights = move_weights.squeeze(0)

    unvisited_mass = [p for p in state.map_spots_norm if (p[0], p[1]) not in state.visited_mass_spots_norm]
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

    vision_image = map_utils.generate_vision_image(
        state.units,
        state.map_width,
        state.map_height,
        state.normalized_map_heights
    )

    if discrete_action == config.ACTION_BUILD:
        action_features = BuildJudger.compute_build_features(
            unit_nx, unit_nz, unit_ny,
            target_nx, target_nz,
            active_mass,
            all_enemies,
            state.units, # friendly units
            False, # is_noop doesn't make as much sense for build actions but we'll set it to false for consistency in training since we want the features to reflect the actual action taken
            vision_image=vision_image,
            target_structure_name="armrad",
            unit_id=unit_id
        )
    else:
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
            vision_image=vision_image
        )

    unit_speed_norm = _get_unit_speed_norm(unit_id) if unit_id is not None else config.MIN_EFFECTIVE_SPEED_NORM
    unit_hp, unit_max_hp = _get_unit_health_context(unit_id) if unit_id is not None else (None, None)

    is_noop_action = action == config.NOOP_ACTION or (
        abs(target_nx - unit_nx) < 1e-3 and abs(target_nz - unit_nz) < 1e-3
    )
    if discrete_action == config.ACTION_MOVE:
        current_direct_penalty = 0.0
        
        if not is_noop_action:
            current_kind = action_kind if action_kind is not None else 'grid'
            current_bias = _hazard_bias_for_candidate_kind(current_kind)
            if current_bias > 0.0:
                current_direct_penalty = _compute_direct_approach_penalty(
                    unit_nx,
                    unit_nz,
                    target_nx,
                    target_nz,
                    all_enemies,
                    enemy_range_image,
                    unit_speed_norm,
                    unit_hp=unit_hp,
                    unit_max_hp=unit_max_hp,
                    bias=current_bias,
                )
        action_features = _inject_hazard_prediction_feature(action_features, current_direct_penalty)
    
    # Validate features don't contain inf/nan
    action_features = [np.clip(f, -1e6, 1e6) if not (np.isinf(f) or np.isnan(f)) else 0.0 for f in action_features]
    
    features_tensor = torch.tensor(action_features, dtype=torch.float32)

    current_q = torch.dot(current_weights, features_tensor)

    if not done:
        with torch.no_grad():
            next_hidden = state.lstm_hidden_states.get(unit_id, init_lstm_hidden())
            next_input_seq = next_state.unsqueeze(0).unsqueeze(0)
            next_action_logits, next_move_weights, next_build_weights, _ = agent(next_input_seq, next_hidden)
            
            # Predict best action for Next State
            next_discrete_action = torch.argmax(next_action_logits, dim=-1).item()
            
            if next_discrete_action == config.ACTION_BUILD:
                next_weights = next_build_weights.squeeze(0)
            else:
                next_weights = next_move_weights.squeeze(0)
            
            max_next_q = -float('inf')
            candidates = []

            # Re-inject the previously chosen target into next-state candidates
            # to mirror get_action's continuity behaviour
            if target_x is not None and target_z is not None:
                target_nx_prev = map_utils.normalize_x(target_x)
                target_nz_prev = map_utils.normalize_z(target_z)
                if map_utils.is_position_reachable(target_nx_prev, target_nz_prev):
                    candidates.append((target_nx_prev, target_nz_prev, 'previous'))

            if next_discrete_action == config.ACTION_MOVE:
                for dx in np.linspace(-200, 200, num=10):
                    for dz in np.linspace(-200, 200, num=10):
                        tx = unit_nx + dx
                        tz = unit_nz + dz
                        tx = max(0, min(config.STANDARD_MAP_WIDTH, tx))
                        tz = max(0, min(config.STANDARD_MAP_HEIGHT, tz))
                        # Only add reachable candidates
                        if map_utils.is_position_reachable(tx, tz):
                            candidates.append((tx, tz, 'grid'))
                # Current next position is always valid
                next_nx_pos = map_utils.normalize_x(next_unit_x)
                next_nz_pos = map_utils.normalize_z(next_unit_z)
                candidates.append((next_nx_pos, next_nz_pos, 'noop'))

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
                                candidates.append((tx, tz, 'escape'))

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
                                        candidates.append((tx, tz, 'strafe'))

                if mass_destination is not None:
                    dest_world_x = map_utils.denormalize_x(mass_destination[0])
                    dest_world_z = map_utils.denormalize_z(mass_destination[1])
                    dist_to_dest = ((dest_world_x - next_unit_x) ** 2 + (dest_world_z - next_unit_z) ** 2) ** 0.5
                    # Only add mass destination if reachable and within approach radius
                    if dist_to_dest <= config.MASS_FINAL_APPROACH_RADIUS and map_utils.is_position_reachable(mass_destination[0], mass_destination[1]):
                        candidates.append((mass_destination[0], mass_destination[1], 'mass_destination'))
                    
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
                    for wx, wz in terrain_waypoints:
                        candidates.append((wx, wz, 'terrain_waypoint'))

            if next_discrete_action == config.ACTION_BUILD:

                # For building, we can consider a different set of candidates, such as nearby buildable locations or specific strategic points.
                # For simplicity, let's consider a small grid around the unit for potential build locations.
                for dx in np.linspace(-20, 20, num=3):
                    for dz in np.linspace(-20, 20, num=3):
                        tx = unit_nx + dx
                        tz = unit_nz + dz
                        tx = max(0, min(config.STANDARD_MAP_WIDTH, tx))
                        tz = max(0, min(config.STANDARD_MAP_HEIGHT, tz))
                        dist = ((tx - unit_nx) ** 2 + (tz - unit_nz) ** 2) ** 0.5
                        if map_utils.is_position_buildable(tx, tz) and dist <= config.BUILD_CANDIDATE_RADIUS:
                            candidates.append((tx, tz, 'build'))

                # Also generate a small circle of build candidates around mass points if they are nearby, as building near mass can be a common strategy.
                for mass in active_mass:
                    for dx in np.linspace(-20, 20, num=3):
                        for dz in np.linspace(-20, 20, num=3):
                            tx = mass[0] + dx
                            tz = mass[1] + dz
                            tx = max(0, min(config.STANDARD_MAP_WIDTH, tx))
                            tz = max(0, min(config.STANDARD_MAP_HEIGHT, tz))
                            dist = ((tx - unit_nx) ** 2 + (tz - unit_nz) ** 2) ** 0.5
                            if map_utils.is_position_buildable(tx, tz) and dist <= config.BUILD_CANDIDATE_RADIUS:
                                candidates.append((tx, tz, 'build'))

                # Maybe include some relating to flat terrain but generic flat terrain points might not be too useful
                # Include at edge of current radar vision as well, as expanding vision can be a key reason to build. (randomly choose like 10)
                # Gonna need to scan the vision image for this (1 LOS, 0.5 Radar, 0 unknown), look for points that are currently unknown but adjacent to known, as those are the ones that building could reveal. Could also weight them by how many unknown cells they would reveal in the vision image.
                # Go out in 24 directions around the unit until you hit a tile that is listed as unknown in the vision image, then add that as a candidate
                h_vis, w_vis = vision_image.shape
                max_ray_steps = max(h_vis, w_vis)
                num_rays = 24
                angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

                for angle in angles:
                    # Generate all steps along this ray at once
                    steps = np.arange(2, max_ray_steps)
                    txs = (unit_nx + steps * np.cos(angle)).astype(int)
                    tzs = (unit_nz + steps * np.sin(angle)).astype(int)

                    # Clip and find valid (in-bounds) indices
                    in_bounds = (txs >= 0) & (txs < w_vis) & (tzs >= 0) & (tzs < h_vis)
                    if not np.any(in_bounds):
                        continue

                    txs_valid = txs[in_bounds]
                    tzs_valid = tzs[in_bounds]

                    # Sample the vision image along the ray
                    ray_values = vision_image[tzs_valid, txs_valid]

                    # Find the first position that is NOT radar-covered (< 0.4 threshold)
                    unknown_mask = ray_values < 0.4
                    if not np.any(unknown_mask):
                        continue

                    first_unknown = np.argmax(unknown_mask)
                    tx = float(txs_valid[first_unknown])
                    tz = float(tzs_valid[first_unknown])

                    if map_utils.is_position_buildable(tx, tz, "armrad"):
                        candidates.append((tx, tz, 'vision_edge_build'))



            next_unvisited = [p for p in state.map_spots_norm if (p[0], p[1]) not in state.visited_mass_spots_norm]
            next_active_mass = [mass_destination] if mass_destination is not None else next_unvisited
            next_enemy_range_image = map_utils.generate_enemy_range_image(
                state.eUnits,
                state.map_width,
                state.map_height,
                state.normalized_map_heights.shape if state.normalized_map_heights is not None else None
            )

            next_vision_image = map_utils.generate_vision_image(
                state.units,
                state.map_width,
                state.map_height,
                state.normalized_map_heights
            )

            next_speed_norm = _get_unit_speed_norm(unit_id) if unit_id is not None else config.MIN_EFFECTIVE_SPEED_NORM
            next_hp, next_max_hp = _get_unit_health_context(unit_id) if unit_id is not None else (None, None)

            for (tx, tz, next_kind) in candidates:
                next_nx = map_utils.normalize_x(next_unit_x)
                next_nz = map_utils.normalize_z(next_unit_z)
                next_ny = map_utils.normalize_y(next_unit_y)
                is_next_noop = abs(tx - next_nx) < 1e-3 and abs(tz - next_nz) < 1e-3

                if next_discrete_action == config.ACTION_MOVE:
                    next_action = config.NOOP_ACTION if next_kind == 'noop' else "MOVE"
                    next_features = MoveJudger.compute_action_features(
                        next_action,
                        next_nx,
                        next_nz,
                        next_ny,
                        next_active_mass,
                        tx,
                        tz,
                        enemy_range_image=next_enemy_range_image,
                        enemy_units=all_enemies,
                        vision_image=next_vision_image
                    )
                    next_direct_penalty = 0.0
                    if not is_next_noop:
                        next_bias = _hazard_bias_for_candidate_kind(next_kind)
                        if next_bias > 0.0:
                            next_direct_penalty = _compute_direct_approach_penalty(
                                next_nx,
                                next_nz,
                                tx,
                                tz,
                                all_enemies,
                                next_enemy_range_image,
                                next_speed_norm,
                                unit_hp=next_hp,
                                unit_max_hp=next_max_hp,
                                bias=next_bias,
                            )
                    next_features = _inject_hazard_prediction_feature(next_features, next_direct_penalty)
                else:
                    next_features = BuildJudger.compute_build_features(
                        next_nx, next_nz, next_ny,
                        tx, tz,
                        next_active_mass,
                        all_enemies,
                        state.units, # friendly units
                        is_next_noop,
                        vision_image=next_vision_image,
                        target_structure_name="armrad",
                        unit_id=unit_id
                    )
                    
                if next_features is None:
                    next_features = [0.0] * (config.NUM_BUILD_FEATURES if next_discrete_action == config.ACTION_BUILD else config.NUM_ACTION_FEATURES)

                next_features = [np.clip(f, -1e6, 1e6) if not (np.isinf(f) or np.isnan(f)) else 0.0 for f in next_features]
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

    loss_continuous = criterion(current_q, target_tensor)
    
    # Calculate discrete classification loss (we want to encourage the agent to pick the action that led to this TD value
    # However, RL classification is tricky because we're just matching Q targets. 
    # For now, let's treat the discrete action chosen as the 'label', and we weight it by the TD advantage.
    # A simple but effective method: encourage actions that had positive advantage, discourage negative.
    adv = target_value - current_q_val
    action_log_probs = torch.nn.functional.log_softmax(action_logits.squeeze(0), dim=-1)
    # Simple advantage-weighted policy gradient for the discrete head
    loss_discrete = -action_log_probs[discrete_action] * adv
    
    loss = loss_continuous + loss_discrete
    
    state.writer.add_scalar('Training/loss', loss.item(), state.step_counter)
    state.writer.add_scalar('Training/loss_continuous', loss_continuous.item(), state.step_counter)
    state.writer.add_scalar('Training/loss_discrete', loss_discrete.item(), state.step_counter)

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
    """Save the agent's neural network weights and adaptive templates as a unified checkpoint."""
    checkpoint = {
        'checkpoint_version': 1,
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'adaptive_candidate_templates': state.adaptive_candidate_templates,
        'adaptive_template_next_id': state.adaptive_template_next_id,
    }
    base_dir = Path(__file__).resolve().parent
    standard_path = base_dir / 'agent_weights_feature_based.pth'
    torch.save(checkpoint, standard_path)

    if getattr(state, 'evalRun', False):
        checkpoint_dir = base_dir / 'Checkpoint'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        run_id = getattr(state, 'run_counter', 0)
        snapshot_name = f"{state.run_name}_run_{run_id:04d}_eval.pth"
        torch.save(checkpoint, checkpoint_dir / snapshot_name)

    print(f"Checkpoint saved: {len(state.adaptive_candidate_templates)} units with adaptive templates.")


def _reset_agent_parameters(model: nn.Module):
    """Reinitialize all learnable parameters of the model, including CNN-specific weight init."""
    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    if hasattr(model, "_initialize_cnn_weights"):
        model._initialize_cnn_weights()


def load_agent():
    """Load checkpoint with model weights and adaptive templates, with backward compatibility for old weight files."""
    try:
        checkpoint_data = torch.load('agent_weights_feature_based.pth')
        
        # Detect checkpoint format: new format is dict with 'checkpoint_version', old format is direct state dict
        if isinstance(checkpoint_data, dict) and 'checkpoint_version' in checkpoint_data:
            # New unified checkpoint format
            print(f"Loading unified checkpoint (v{checkpoint_data.get('checkpoint_version', 1)})...")
            agent.load_state_dict(checkpoint_data['model_state_dict'])
            
            if 'optimizer_state_dict' in checkpoint_data:
                try:
                    optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                except Exception as e:
                    print(f"Warning: could not restore optimizer state: {e}")
            
            if 'adaptive_candidate_templates' in checkpoint_data:
                state.adaptive_candidate_templates = checkpoint_data['adaptive_candidate_templates']
                print(f"Restored {len(state.adaptive_candidate_templates)} units with adaptive templates.")
            
            if 'adaptive_template_next_id' in checkpoint_data:
                state.adaptive_template_next_id = checkpoint_data['adaptive_template_next_id']
            
            print("Unified checkpoint loaded successfully.")
        else:
            # Old format: direct state_dict, treat as model-only
            print("Loading legacy weight format (model only, no templates)...")
            agent.load_state_dict(checkpoint_data)
            print("Legacy weights loaded. Adaptive templates will be built fresh.")
        
        has_nan = any(torch.isnan(p).any().item() for p in agent.parameters())
        if has_nan:
            print("Loaded weights contain NaN. Reinitializing model weights.")
            _reset_agent_parameters(agent)
    except FileNotFoundError:
        print("No saved checkpoint found, starting fresh.")
    except RuntimeError as exc:
        print("Saved checkpoint is incompatible with the current architecture.")
        print(f"Details: {exc}")
        print("Starting with fresh weights and templates.")
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
