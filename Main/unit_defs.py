"""
Loader and lookup helpers for unit definition data (weapon types, projectile speeds, etc.).
Reads from data/unit_defs.json at startup and provides per-unit enrichment for the neural net.
"""

import json
import os

import config

# Module-level cache populated by load_unit_defs()
_unit_defs: dict = {}

# Weapon-type string → one-hot index mapping
WEAPON_TYPE_INDEX = {
    "hitscan": config.WEAPON_HITSCAN,
    "projectile": config.WEAPON_PROJECTILE,
    "missile": config.WEAPON_MISSILE,
    "beam": config.WEAPON_BEAM,
}


def load_unit_defs(path: str | None = None) -> dict:
    """Load unit definitions from the JSON file into the module-level cache.

    Args:
        path: Optional override path. Defaults to config.UNIT_DEFS_PATH.

    Returns:
        The loaded dictionary keyed by unit name.
    """
    global _unit_defs
    if path is None:
        path = config.UNIT_DEFS_PATH

    if not os.path.isfile(path):
        print(f"[unit_defs] WARNING: {path} not found – using empty defs.")
        _unit_defs = {}
        return _unit_defs

    with open(path, "r", encoding="utf-8") as f:
        _unit_defs = json.load(f)

    # Strip metadata keys (start with '_' except '_default')
    keys_to_remove = [k for k in _unit_defs if k.startswith("_") and k != "_default"]
    for k in keys_to_remove:
        del _unit_defs[k]

    print(f"[unit_defs] Loaded {len(_unit_defs)} unit definitions from {path}")
    return _unit_defs


def get_unit_def(unit_name: str) -> dict:
    """Return the full definition dict for a unit name, falling back to _default.

    Args:
        unit_name: The internal BAR unit name (e.g. 'armpw').

    Returns:
        A dict with keys: weapon_type, projectile_speed, aoe_radius, reload_time, dps, range.
    """
    name_lower = unit_name.lower()
    if name_lower in _unit_defs:
        return _unit_defs[name_lower]
    return _unit_defs.get("_default", {
        "weapon_type": "projectile",
        "projectile_speed": 200,
        "aoe_radius": 16,
        "reload_time": 2.0,
        "dps": 50,
        "range": 300,
    })


def get_weapon_one_hot(weapon_type_str: str) -> list[float]:
    """Convert a weapon type string to a one-hot vector of length NUM_WEAPON_TYPES.

    Args:
        weapon_type_str: One of 'hitscan', 'projectile', 'missile', 'beam'.

    Returns:
        A list of floats with 1.0 at the weapon-type index and 0.0 elsewhere.
    """
    vec = [0.0] * config.NUM_WEAPON_TYPES
    idx = WEAPON_TYPE_INDEX.get(weapon_type_str, config.WEAPON_PROJECTILE)
    vec[idx] = 1.0
    return vec


def get_weapon_info(unit_name: str) -> dict:
    """Return weapon enrichment fields ready to merge into a unit dict.

    Args:
        unit_name: The internal BAR unit name.

    Returns:
        Dict with keys: weapon_type, weapon_one_hot, projectile_speed, aoe_radius, dps, range.
    """
    defn = get_unit_def(unit_name)
    weapon_type = defn.get("weapon_type", "projectile")
    return {
        "weapon_type": weapon_type,
        "weapon_one_hot": get_weapon_one_hot(weapon_type),
        "projectile_speed": defn.get("projectile_speed", 200),
        "aoe_radius": defn.get("aoe_radius", 16),
        "dps": defn.get("dps", 50),
        "range": defn.get("range", 300),
    }


def normalize_projectile_speed(speed: float) -> float:
    """Normalize a projectile speed to [0, 1] using the configured max speed.

    Args:
        speed: Raw projectile speed value.

    Returns:
        Clamped value in [0, 1].
    """
    return min(speed / config.MAX_PROJECTILE_SPEED, 1.0)


def normalize_aoe(radius: float) -> float:
    """Normalize an AoE radius to [0, 1] using the configured max AoE.

    Args:
        radius: Raw area-of-effect radius.

    Returns:
        Clamped value in [0, 1].
    """
    return min(radius / config.MAX_AOE_RADIUS, 1.0)


def normalize_dps(dps: float) -> float:
    """Normalize a DPS value to [0, 1] using the configured max DPS.

    Args:
        dps: Raw damage-per-second value.

    Returns:
        Clamped value in [0, 1].
    """
    return min(dps / config.MAX_DPS, 1.0)
