"""Offline imitation-learning pretrainer, rebuilt against the CURRENT live agent.

Design goal: zero duplicated encoder logic. Instead of maintaining a second,
parallel copy of RTSAgent (which is what silently rotted last time), this
script imports the real `agent_model.RTSAgent` and the real `Rewards`
judgers, and only adds the data-loading / contrastive-imitation-loss glue
that is specific to offline training.

Two live-agent quirks that offline data must accommodate:
  1. Map files changed format. Socket ML.py now writes/reads:
       - mapHeightInfo.txt : a bare numeric CSV grid (no header)
       - massInfo.txt      : "x,y,z,value" per line (no header)
     Legacy Hooked recordings (via hooked_converter.py) still describe maps
     with the OLD header-based format. Both are auto-detected below so a
     combined dataset spanning both eras can still be trained on in one run.
  2. The agent now has two action heads (MOVE vs BUILD) instead of one.
     Samples are routed to MoveJudger/move_head or BuildJudger/build_head
     based on their recorded action type, and the discrete action_head is
     trained with a cross-entropy loss against that same label.

KNOWN SIMPLIFICATIONS (intentional, to keep this tractable — see summary
notes delivered alongside this file for the full list):
  - No LSTM temporal context: each sample is encoded with a fresh zero
    hidden state, matching the original offline script's stateless design.
    This is a real gap vs. the live agent's per-unit persistent hidden
    state, and is the most valuable thing to fix if this pipeline is
    revived (would require reconstructing per-unit sample ORDER within a
    match, which today's exported JSON does preserve via 'step'/'timestamp').
  - hazard_prediction feature is left at 0.0 for MOVE samples (matching how
    it's injected live) rather than recomputing compute_direct_approach_penalty
    per candidate, since that requires per-sample HP/speed context that
    isn't always present in older exports.
  - state.fEnergy / state.fMass are not present in exported samples (they
    weren't part of the recorded schema), so the economy embedding sees 0
    for every sample. Cheap future fix: add fEnergy/fMass to
    PeriodicRewards.record_match_sample() so future exports carry it.
  - Legacy Hooked-converted samples have generic unit dicts (name='unit',
    no weapon/type info), so they'll encode through unit_defs' "_default"
    fallback rather than real per-unit-type stats. This is a property of
    what Hooked recorded, not something this script can recover.
"""

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# Add parent directory to path to import the live agent's modules.
# NOTE: run this script from the "Python Agent" root (e.g.
#   python Training/offline_train.py some_dataset.json
# ) so config's relative paths (data/unit_defs.json, cache/map_fields, ...)
# resolve the same way they do for Socket ML.py.
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
import map_utils
import runtime_state as state
import unit_defs
from Rewards import MoveJudger, BuildJudger
from agent_model import RTSAgent

# === TRAINING SETTINGS ===
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05          # matches agent_core.py's live optimizer
NEGATIVE_SAMPLES = 8
DISCRETE_LOSS_WEIGHT = 0.5
OFFLINE_CHECKPOINT_PATH = str(Path(__file__).resolve().parent.parent / "agent_weights_offline.pth")

# === ENCODER SIZE (mirrors agent_core.py exactly; do not hand-roll this) ===
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
optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Local mirrors of map globals (kept alongside runtime_state's copies so the
# rest of this file reads naturally; state.* is what map_utils/agent_model
# actually consult).
map_width = 0
map_height = 0
map_spots_norm: List[Tuple[float, float, float]] = []


# ---------------------------------------------------------------------------
# Map loading (format auto-detected: legacy Hooked headers vs. current raw grid)
# ---------------------------------------------------------------------------

def _looks_legacy_heights(filepath: Path) -> bool:
    with filepath.open("r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()
    return first_line.startswith("MapSizeX")


def _load_map_heights_legacy(filepath: Path) -> None:
    """Old Hooked-recording format: header lines + sparse 'x,z,height' rows."""
    global map_width, map_height
    heights: Dict[Tuple[int, int], float] = {}
    local_w = local_h = 0
    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("MapSizeX"):
                local_w = int(line.split(",")[1])
                continue
            if line.startswith("MapSizeZ"):
                local_h = int(line.split(",")[1])
                continue
            if line.startswith("WaterLevel") or line.startswith("x,z,height"):
                continue
            parts = line.split(",")
            if len(parts) >= 3:
                try:
                    x = int(float(parts[0]))
                    z = int(float(parts[1]))
                    h = float(parts[2])
                    heights[(x, z)] = h
                except ValueError:
                    continue

    map_width, map_height = local_w, local_h
    if not heights or map_width <= 0 or map_height <= 0:
        state.map_heights = None
        return

    grid = np.zeros((map_height, map_width), dtype=np.float32)
    for (x, z), h in heights.items():
        if 0 <= z < map_height and 0 <= x < map_width:
            grid[z, x] = h
    state.map_heights = grid


def _load_map_heights_raw_grid(filepath: Path, meta_width: int, meta_height: int) -> None:
    """Current live format: bare numeric CSV grid, no header (see Socket ML.py)."""
    global map_width, map_height
    grid = pd.read_csv(filepath, header=None).values.astype(np.float32)
    # Metadata (recorded at export time) already carries world-unit dims;
    # fall back to the grid's own shape if metadata is missing.
    map_width = int(meta_width) if meta_width else grid.shape[1]
    map_height = int(meta_height) if meta_height else grid.shape[0]
    state.map_heights = grid


def load_map_heights(filepath: str, meta_width: int = 0, meta_height: int = 0) -> None:
    path = Path(filepath)
    if not path.exists():
        print(f"[WARNING] Map heights file not found: {path}")
        state.map_heights = None
        return

    if _looks_legacy_heights(path):
        _load_map_heights_legacy(path)
    else:
        _load_map_heights_raw_grid(path, meta_width, meta_height)

    state.map_width = map_width
    state.map_height = map_height
    # Delegates to the REAL map_utils implementation (array-based), so
    # normalization/height stats stay identical to the live agent's.
    map_utils.build_normalized_height_map()


def _looks_legacy_spots(filepath: Path) -> bool:
    with filepath.open("r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip().lower()
    return first_line.startswith("type") or "index" in first_line


def _load_map_spots_legacy(filepath: Path) -> List[Tuple[float, float, float]]:
    """Old format: CSV with a header, columns include x/z, no value weighting."""
    import csv
    spots = []
    with filepath.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                spots.append((float(row["x"]), float(row["z"]), 1.0))
            except (ValueError, KeyError):
                continue
    return spots


def _load_map_spots_raw(filepath: Path) -> List[Tuple[float, float, float]]:
    """Current live format: 'x,y,z,value' per line, no header (see Socket ML.py)."""
    spots = []
    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                x, _y, z, value = map(float, line.split(","))
                spots.append((x, z, value))
            except ValueError:
                continue
    return spots


def load_map_spots(filepath: str) -> None:
    global map_spots_norm
    path = Path(filepath)
    if not path.exists():
        print(f"[WARNING] Map spots file not found: {path}")
        state.mass_spots = []
        map_spots_norm = []
        return

    if _looks_legacy_spots(path):
        raw_spots = _load_map_spots_legacy(path)
    else:
        raw_spots = _load_map_spots_raw(path)

    max_val = max((v for (_, _, v) in raw_spots), default=1.0) or 1.0
    state.mass_spots = raw_spots

    map_spots_norm = [
        (map_utils.normalize_x(x), map_utils.normalize_z(z), float(v) / float(max_val))
        for x, z, v in raw_spots
    ]
    state.map_spots_norm = map_spots_norm


def _apply_map_context(signature: Tuple[str, str, int, int]) -> None:
    """Load map assets + rebuild terrain/mass cost fields for one map group."""
    map_heights_file, map_spots_file, meta_width, meta_height = signature

    if map_heights_file:
        load_map_heights(map_heights_file, meta_width, meta_height)
    if map_spots_file:
        load_map_spots(map_spots_file)

    state.visited_mass_spots_norm = set()  # imitation samples are stateless snapshots

    # Reuse the live cache so repeated offline runs on the same map don't
    # re-pay the Dijkstra cost-field build every time.
    if state.normalized_map_heights is not None:
        if not map_utils.load_cached_cost_fields():
            map_utils.build_terrain_cost_map()
            map_utils.build_mass_cost_fields()
            map_utils.save_cached_cost_fields()


# ---------------------------------------------------------------------------
# Dataset loading / grouping (unchanged contract with combine_recordings.py)
# ---------------------------------------------------------------------------

def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _map_signature_from_metadata(meta: Dict[str, Any]) -> Tuple[str, str, int, int]:
    return (
        str(meta.get("map_heights_file", "") or ""),
        str(meta.get("map_spots_file", "") or ""),
        _safe_int(meta.get("map_width", 0), 0),
        _safe_int(meta.get("map_height", 0), 0),
    )


def _map_signature_from_sample(sample: Dict[str, Any], fallback):
    raw = sample.get("_map_signature")
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        return (str(raw[0] or ""), str(raw[1] or ""), _safe_int(raw[2], 0), _safe_int(raw[3], 0))
    return fallback


def load_imitation_data(filepath: str) -> Dict:
    if not Path(filepath).exists():
        print(f"File {filepath} not found.")
        return {}
    with open(filepath, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data.get('samples', []))} samples from {filepath}")
    return data


# ---------------------------------------------------------------------------
# Per-sample feature computation (delegates entirely to the real judgers)
# ---------------------------------------------------------------------------

def _build_range_and_vision_images(friendly_units, enemy_units):
    if state.normalized_map_heights is None:
        return None, None
    enemy_range_image = map_utils.generate_enemy_range_image(
        enemy_units, state.map_width, state.map_height, state.normalized_map_heights.shape
    )
    vision_image = map_utils.generate_vision_image(
        friendly_units, state.map_width, state.map_height, state.normalized_map_heights
    )
    return enemy_range_image, vision_image


def compute_move_features(unit_nx, unit_nz, unit_ny, target_nx, target_nz,
                           enemy_range_image, enemy_units, vision_image, is_noop):
    action_type = config.NOOP_ACTION if is_noop else "MOVE"
    unvisited_mass = list(map_spots_norm)  # imitation: treat every spot as still active
    features = MoveJudger.compute_action_features(
        action_type, unit_nx, unit_nz, unit_ny, unvisited_mass,
        target_nx, target_nz,
        enemy_range_image=enemy_range_image,
        enemy_units=enemy_units,
        vision_image=vision_image,
    )
    # hazard_prediction left at its default 0.0 placeholder — see module
    # docstring "KNOWN SIMPLIFICATIONS".
    return features


def compute_build_features_for_sample(unit_nx, unit_nz, unit_ny, target_nx, target_nz,
                                       enemy_units, friendly_units, is_noop,
                                       vision_image, unit_id):
    active_mass = list(map_spots_norm)
    return BuildJudger.compute_build_features(
        unit_nx, unit_nz, unit_ny,
        target_nx, target_nz,
        active_mass,
        enemy_units,
        friendly_units,
        is_noop,
        vision_image=vision_image,
        target_structure_name="armrad",
        unit_id=unit_id,
        structure_vision_image=vision_image,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_imitation(dataset_file: str, epochs: int = 10, shuffle: bool = True):
    data = load_imitation_data(dataset_file)
    samples = data.get("samples", [])
    if not samples:
        print("No samples to train on.")
        return

    meta = data.get("metadata", {})
    default_signature = _map_signature_from_metadata(meta)

    grouped: Dict[Tuple[str, str, int, int], List[Dict[str, Any]]] = {}
    for sample in samples:
        sig = _map_signature_from_sample(sample, default_signature)
        grouped.setdefault(sig, []).append(sample)

    if not grouped:
        print("No valid grouped samples to train on.")
        return

    print(f"Detected {len(grouped)} map group(s) in dataset.")
    for idx, (sig, group) in enumerate(grouped.items(), start=1):
        print(f"  Group {idx}: samples={len(group)}, map_heights='{sig[0]}', map_spots='{sig[1]}'")

    ce_loss = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        if shuffle:
            for group in grouped.values():
                random.shuffle(group)

        epoch_loss = 0.0
        sample_count = 0
        skipped = 0

        for group_idx, (sig, group_samples) in enumerate(grouped.items(), start=1):
            print(f"Epoch {epoch + 1}: loading map context for group {group_idx}/{len(grouped)}")
            _apply_map_context(sig)

            for sample in group_samples:
                unit_id = sample["unit_id"]
                friendly_units = sample["friendly_units"]
                enemy_units = sample["enemy_units"]
                agent_unit = next((u for u in friendly_units if u["id"] == unit_id), None)
                if agent_unit is None:
                    skipped += 1
                    continue

                action = sample["action"]
                action_type = action.get("type", config.NOOP_ACTION)
                target_x = action.get("x", agent_unit["x"])
                target_z = action.get("z", agent_unit["z"])
                is_build = action_type == "BUILD"
                discrete_label = config.ACTION_BUILD if is_build else config.ACTION_MOVE

                # Vision/enemy-range images depend on this sample's live positions,
                # so they're rebuilt per sample (this is the main cost of this loop).
                state.fEnergy = 0.0  # not present in exported schema — see docstring
                state.fMass = 0.0
                enemy_range_image, vision_image = _build_range_and_vision_images(friendly_units, enemy_units)
                state.vision_image = vision_image

                state_vec = agent.encode_state(agent_unit, friendly_units, enemy_units)
                input_seq = state_vec.unsqueeze(0).unsqueeze(0)
                action_logits, move_features, build_features, _ = agent(input_seq)

                unit_nx = map_utils.normalize_x(agent_unit["x"])
                unit_nz = map_utils.normalize_z(agent_unit["z"])
                unit_ny = map_utils.normalize_y(agent_unit["y"])
                target_nx = map_utils.normalize_x(target_x)
                target_nz = map_utils.normalize_z(target_z)
                is_noop = (abs(target_nx - unit_nx) < 1e-3 and abs(target_nz - unit_nz) < 1e-3)

                if is_build:
                    weights = build_features.squeeze(0)
                    pos_feats = compute_build_features_for_sample(
                        unit_nx, unit_nz, unit_ny, target_nx, target_nz,
                        enemy_units, friendly_units, is_noop, vision_image, unit_id,
                    )
                else:
                    weights = move_features.squeeze(0)
                    pos_feats = compute_move_features(
                        unit_nx, unit_nz, unit_ny, target_nx, target_nz,
                        enemy_range_image, enemy_units, vision_image, is_noop,
                    )

                pos_score = torch.dot(weights, torch.tensor(pos_feats, dtype=torch.float32))
                scores = [pos_score]

                for _ in range(NEGATIVE_SAMPLES):
                    rx = random.uniform(0, config.STANDARD_MAP_WIDTH)
                    rz = random.uniform(0, config.STANDARD_MAP_HEIGHT)
                    if is_build:
                        neg_feats = compute_build_features_for_sample(
                            unit_nx, unit_nz, unit_ny, rx, rz,
                            enemy_units, friendly_units, False, vision_image, unit_id,
                        )
                    else:
                        neg_feats = compute_move_features(
                            unit_nx, unit_nz, unit_ny, rx, rz,
                            enemy_range_image, enemy_units, vision_image, False,
                        )
                    neg_score = torch.dot(weights, torch.tensor(neg_feats, dtype=torch.float32))
                    scores.append(neg_score)

                scores_tensor = torch.stack(scores)
                contrastive_loss = -torch.log_softmax(scores_tensor, dim=0)[0]

                discrete_target = torch.tensor([discrete_label], dtype=torch.long)
                discrete_loss = ce_loss(action_logits, discrete_target)

                loss = contrastive_loss + DISCRETE_LOSS_WEIGHT * discrete_loss

                # Fixed vs. original: step every sample instead of only at epoch end
                # (previously the optimizer.step() at the end of the epoch loop only
                # ever applied the LAST sample's gradient — every other sample's
                # zero_grad()+backward() pair was discarded before a step ran).
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                sample_count += 1

        avg_loss = epoch_loss / max(1, sample_count)
        print(f"Epoch {epoch + 1}/{epochs} - Avg Loss: {avg_loss:.4f} (skipped {skipped} samples)")

    print("Imitation training complete!")


# ---------------------------------------------------------------------------
# Checkpointing — deliberately a SEPARATE file from the live agent's, using
# the SAME unified format, so it can be inspected/promoted intentionally
# rather than silently overwriting agent_weights_feature_based.pth.
# ---------------------------------------------------------------------------

def save_agent(filepath: str = OFFLINE_CHECKPOINT_PATH):
    has_nan = any(torch.isnan(p).any().item() for p in agent.parameters())
    if has_nan:
        print("Warning: NaN detected in weights. Skipping save to avoid corrupting checkpoint.")
        return
    checkpoint = {
        "checkpoint_version": 1,
        "model_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "adaptive_candidate_templates": {},
        "adaptive_template_next_id": 1,
    }
    torch.save(checkpoint, filepath)
    print(f"Agent weights saved to {filepath}")


def load_agent(filepath: str = OFFLINE_CHECKPOINT_PATH):
    """Loads either this script's own unified checkpoints, or the LIVE
    agent's unified checkpoint (agent_weights_feature_based.pth) if you want
    to fine-tune offline starting from the online agent's current weights."""
    try:
        checkpoint_data = torch.load(filepath)
        if isinstance(checkpoint_data, dict) and "checkpoint_version" in checkpoint_data:
            agent.load_state_dict(checkpoint_data["model_state_dict"])
            if "optimizer_state_dict" in checkpoint_data:
                try:
                    optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
                except Exception as e:
                    print(f"Warning: could not restore optimizer state: {e}")
            print("Unified checkpoint loaded successfully.")
        else:
            # Raw state_dict (e.g. hand-exported weights only).
            agent.load_state_dict(checkpoint_data)
            print("Legacy/raw weights loaded (model only).")

        has_nan = any(torch.isnan(p).any().item() for p in agent.parameters())
        if has_nan:
            print("Warning: NaN detected in loaded weights. Reinitializing model.")
            agent.initialize_cnn_weights()
    except FileNotFoundError:
        print(f"No saved weights found at {filepath}; starting fresh.")
    except RuntimeError as exc:
        print(f"Checkpoint at {filepath} is incompatible with the current architecture: {exc}")
        print("Starting with fresh weights.")


if __name__ == "__main__":
    dataset_file = sys.argv[1] if len(sys.argv) > 1 else "replay.json"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    unit_defs.load_unit_defs(str(Path(__file__).resolve().parent.parent / config.UNIT_DEFS_PATH))
    state.unit_defs_loaded = True

    load_agent(OFFLINE_CHECKPOINT_PATH)
    train_imitation(dataset_file, epochs=epochs)
    save_agent(OFFLINE_CHECKPOINT_PATH)

    print(f"\nUsage: python offline_train.py [imitation_dataset.json] [epochs]")
    print(f"Checkpoint path: {OFFLINE_CHECKPOINT_PATH}")