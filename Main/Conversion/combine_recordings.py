"""
Combine multiple Hooked recordings into a single training dataset.
Usage: python combine_recordings.py output.json recording1_prefix recording2_prefix ...
Example: python combine_recordings.py combined_dataset.json "Hooked 1_20260202_215323" "Hooked 2_20260203_120000"
"""

import json
import sys
from pathlib import Path
from hooked_converter import HookedDataConverter


def combine_recordings(output_path: str, recording_prefixes: list):
    """Combine multiple Hooked recordings into one JSON dataset."""
    
    combined_samples = []
    combined_metadata = {
        "total_samples": 0,
        "recordings": []
    }
    
    for prefix in recording_prefixes:
        print(f"\nProcessing recording: {prefix}")
        
        try:
            converter = HookedDataConverter(prefix, noop_keep_ratio=0.2)
            dataset = converter.convert_to_replay(output_file=None)
            
            if dataset and "samples" in dataset:
                num_samples = len(dataset["samples"])
                combined_samples.extend(dataset["samples"])
                
                # Track metadata
                recording_info = {
                    "prefix": prefix,
                    "samples": num_samples,
                    "map_info": dataset.get("metadata", {})
                }
                combined_metadata["recordings"].append(recording_info)
                
                print(f"  ✓ Added {num_samples} samples from {prefix}")
            else:
                print(f"  ✗ Failed to convert {prefix}")
                
        except Exception as e:
            print(f"  ✗ Error processing {prefix}: {e}")
            continue
    
    # Build final combined dataset
    combined_metadata["total_samples"] = len(combined_samples)
    
    combined_dataset = {
        "metadata": combined_metadata,
        "samples": combined_samples
    }
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(combined_dataset, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Combined dataset saved to: {output_path}")
    print(f"Total recordings: {len(combined_metadata['recordings'])}")
    print(f"Total samples: {combined_metadata['total_samples']}")
    print(f"{'='*60}")
    
    # Print per-recording breakdown
    print("\nBreakdown by recording:")
    for rec in combined_metadata["recordings"]:
        print(f"  {rec['prefix']}: {rec['samples']} samples")
    
    return combined_dataset


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python combine_recordings.py output.json [recording1_prefix recording2_prefix ...]")
        print('       python combine_recordings.py output.json --all  (combines all Hooked_*.txt files)')
        print('Example: python combine_recordings.py combined.json "Hooked 1_20260202_215323" "Hooked 2_20260203_120000"')
        sys.exit(1)
    
    output_file = sys.argv[1]
    
    # If --all flag, auto-discover all Hooked recordings
    if len(sys.argv) == 3 and sys.argv[2] == "--all":
        # Find all unique Hooked recording prefixes
        runtime_files = list(Path('records/').glob('*_runtime_data.txt'))
        recording_prefixes = []
        
        for runtime_file in runtime_files:
            # Extract prefix: "Hooked 1_20260202_215323_runtime_data.txt" -> "Hooked 1_20260202_215323"
            prefix = runtime_file.stem.replace('_runtime_data', '')
            recording_prefixes.append(f"records/{prefix}")
        
        if not recording_prefixes:
            print("No recordings found in 'records' directory!")
            sys.exit(1)
        
        print(f"Auto-discovered {len(recording_prefixes)} recordings:")
        for prefix in recording_prefixes:
            print(f"  - {prefix}")
        print()
    else:
        if len(sys.argv) < 3:
            print("Error: Please provide recording prefixes or use --all flag")
            sys.exit(1)
        recording_prefixes = sys.argv[2:]
    
    print(f"Combining {len(recording_prefixes)} recordings...")
    combine_recordings(output_file, recording_prefixes)
