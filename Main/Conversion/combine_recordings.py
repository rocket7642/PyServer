"""Combine training datasets from legacy Hooked recordings and JSON replay exports.

Supports three input modes:
1) Hooked prefixes (legacy runtime/map files) that are converted on the fly.
2) Existing replay JSON files (including online top-match exports).
3) Auto-discovery via flags.

Examples:
  python combine_recordings.py combined.json --all
  python combine_recordings.py combined.json --all-agent
  python combine_recordings.py combined.json records/Hooked_1_20260202 replay_a.json replay_b.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hooked_converter import HookedDataConverter


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _discover_hooked_prefixes(records_dir: Path) -> List[str]:
    runtime_files = sorted(records_dir.glob('*_runtime_data.txt'))
    prefixes: List[str] = []
    for runtime_file in runtime_files:
        prefix = runtime_file.stem.replace('_runtime_data', '')
        prefixes.append(str(records_dir / prefix))
    return prefixes


def _discover_agent_json(agent_dir: Path) -> List[str]:
    if not agent_dir.exists():
        return []

    json_paths = sorted(
        p for p in agent_dir.glob('*.json')
        if p.name != 'top_matches_index.json'
    )
    return [str(p) for p in json_paths]


def _load_json_dataset(dataset_path: Path) -> Dict[str, Any]:
    with dataset_path.open('r', encoding='utf-8') as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Dataset is not a JSON object: {dataset_path}")

    samples = payload.get('samples', [])
    if not isinstance(samples, list):
        raise ValueError(f"Dataset 'samples' must be a list: {dataset_path}")

    metadata = payload.get('metadata', {})
    if not isinstance(metadata, dict):
        metadata = {}

    return {
        'metadata': metadata,
        'samples': samples,
    }


def _dataset_map_key(metadata: Dict[str, Any]) -> Tuple[str, str, int, int]:
    return (
        str(metadata.get('map_heights_file', '') or ''),
        str(metadata.get('map_spots_file', '') or ''),
        int(_safe_float(metadata.get('map_width', 0), 0.0)),
        int(_safe_float(metadata.get('map_height', 0), 0.0)),
    )


def _map_key_as_list(map_key: Tuple[str, str, int, int]) -> List[Any]:
    """Serialize map key into a JSON-friendly list for per-sample annotations."""
    return [map_key[0], map_key[1], map_key[2], map_key[3]]


def _samples_match_schema(sample: Dict[str, Any]) -> bool:
    return (
        isinstance(sample, dict)
        and 'unit_id' in sample
        and 'friendly_units' in sample
        and 'enemy_units' in sample
        and isinstance(sample.get('action', {}), dict)
    )


def _normalize_dataset_samples(dataset: Dict[str, Any], source_name: str) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for sample in dataset.get('samples', []):
        if _samples_match_schema(sample):
            normalized.append(sample)
        else:
            print(f"  ! Skipping malformed sample in {source_name}")
    return normalized


def _load_source_dataset(source: str, noop_keep_ratio: float) -> Dict[str, Any]:
    source_path = Path(source)

    # Existing JSON replay dataset.
    if source_path.suffix.lower() == '.json' and source_path.exists():
        dataset = _load_json_dataset(source_path)
        dataset['source_type'] = 'json'
        dataset['source_name'] = source
        return dataset

    # Hooked recording prefix converted on demand.
    converter = HookedDataConverter(source, noop_keep_ratio=noop_keep_ratio)
    dataset = converter.convert_to_replay(output_file=None)
    dataset['source_type'] = 'hooked'
    dataset['source_name'] = source
    return dataset


def combine_recordings(
    output_path: str,
    sources: List[str],
    noop_keep_ratio: float = 0.2,
    allow_mixed_maps: bool = False,
) -> Dict[str, Any]:
    """Combine multiple source datasets into one replay JSON file.

    When allow_mixed_maps=False, only samples from the first map signature are kept.
    This is safer for the current offline trainer, which loads one map metadata block.
    """
    combined_samples: List[Dict[str, Any]] = []
    source_entries: List[Dict[str, Any]] = []

    primary_map_key: Optional[Tuple[str, str, int, int]] = None
    primary_map_metadata: Dict[str, Any] = {}

    skipped_sources = 0
    skipped_samples_map_mismatch = 0

    for source in sources:
        print(f"\nProcessing source: {source}")
        try:
            dataset = _load_source_dataset(source, noop_keep_ratio=noop_keep_ratio)
        except Exception as exc:
            print(f"  x Failed to load source {source}: {exc}")
            skipped_sources += 1
            continue

        metadata = dataset.get('metadata', {})
        samples = _normalize_dataset_samples(dataset, source)
        map_key = _dataset_map_key(metadata)

        if primary_map_key is None:
            primary_map_key = map_key
            primary_map_metadata = dict(metadata)

        source_kept = 0
        if allow_mixed_maps or map_key == primary_map_key:
            map_key_payload = _map_key_as_list(map_key)
            # Tag each sample with its map signature so mixed-map offline training can switch context.
            for sample in samples:
                if isinstance(sample, dict):
                    sample['_map_signature'] = map_key_payload
            combined_samples.extend(samples)
            source_kept = len(samples)
        else:
            skipped_samples_map_mismatch += len(samples)
            print("  ! Map mismatch with primary dataset; skipping samples from this source.")

        source_entries.append({
            'source': source,
            'source_type': dataset.get('source_type', 'unknown'),
            'map_key': {
                'map_heights_file': map_key[0],
                'map_spots_file': map_key[1],
                'map_width': map_key[2],
                'map_height': map_key[3],
            },
            'samples_loaded': len(samples),
            'samples_kept': source_kept,
        })
        print(f"  + Loaded {len(samples)} samples, kept {source_kept}")

    output_metadata: Dict[str, Any] = {
        'total_samples': len(combined_samples),
        'combined_from': source_entries,
        'skipped_sources': skipped_sources,
        'skipped_samples_map_mismatch': skipped_samples_map_mismatch,
        'allow_mixed_maps': bool(allow_mixed_maps),
    }

    # Preserve map metadata expected by offline_train.py
    output_metadata['map_heights_file'] = primary_map_metadata.get('map_heights_file', '')
    output_metadata['map_spots_file'] = primary_map_metadata.get('map_spots_file', '')
    output_metadata['map_width'] = int(_safe_float(primary_map_metadata.get('map_width', 0), 0.0))
    output_metadata['map_height'] = int(_safe_float(primary_map_metadata.get('map_height', 0), 0.0))

    combined_dataset = {
        'metadata': output_metadata,
        'samples': combined_samples,
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('w', encoding='utf-8') as f:
        json.dump(combined_dataset, f, indent=2)

    print(f"\n{'=' * 64}")
    print(f"Combined dataset saved to: {output_path}")
    print(f"Sources processed: {len(source_entries)} (failed: {skipped_sources})")
    print(f"Total samples kept: {len(combined_samples)}")
    if skipped_samples_map_mismatch > 0:
        print(f"Skipped map-mismatched samples: {skipped_samples_map_mismatch}")
    print(f"{'=' * 64}")

    return combined_dataset


def _build_source_list(args: argparse.Namespace) -> List[str]:
    sources: List[str] = []

    # Positional mixed sources: can be Hooked prefixes or JSON files.
    sources.extend(args.sources)

    if args.all:
        hooked = _discover_hooked_prefixes(Path(args.records_dir))
        sources.extend(hooked)

    if args.all_agent:
        agent_json = _discover_agent_json(Path(args.agent_dir))
        sources.extend(agent_json)

    # Deduplicate while preserving first occurrence order.
    unique_sources: List[str] = []
    seen = set()
    for src in sources:
        if src in seen:
            continue
        seen.add(src)
        unique_sources.append(src)

    return unique_sources


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Combine Hooked recordings and replay JSON datasets into one training file.'
    )
    parser.add_argument('output', help='Output JSON dataset path')
    parser.add_argument(
        'sources',
        nargs='*',
        help='Mixed input sources: Hooked recording prefixes and/or replay JSON files',
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Auto-discover all Hooked recordings from records_dir',
    )
    parser.add_argument(
        '--all-agent',
        action='store_true',
        help='Auto-discover all agent replay JSON files from agent_dir',
    )
    parser.add_argument(
        '--records-dir',
        default='records',
        help='Directory containing *_runtime_data.txt files (default: records)',
    )
    parser.add_argument(
        '--agent-dir',
        default='../Recordings/AgentReplayTop',
        help='Directory containing online agent replay JSON files (default: ../Recordings/AgentReplayTop)',
    )
    parser.add_argument(
        '--noop-keep-ratio',
        type=float,
        default=0.2,
        help='NOOP sampling ratio for Hooked conversion inputs (default: 0.2)',
    )
    parser.add_argument(
        '--allow-mixed-maps',
        action='store_true',
        help='Allow mixing datasets from different maps (offline_train currently expects one map metadata block)',
    )
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = _parse_args(sys.argv[1:])

    sources = _build_source_list(args)
    if not sources:
        print('No input sources found.')
        print('Provide sources directly and/or use --all and/or --all-agent.')
        sys.exit(1)

    print(f"Discovered {len(sources)} source(s).")
    for src in sources:
        print(f"  - {src}")

    combine_recordings(
        output_path=args.output,
        sources=sources,
        noop_keep_ratio=max(0.0, min(1.0, args.noop_keep_ratio)),
        allow_mixed_maps=args.allow_mixed_maps,
    )
