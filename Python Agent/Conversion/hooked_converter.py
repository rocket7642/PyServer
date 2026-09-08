"""Converts Hooked game recording files into training replay format.

UPDATE (offline-pipeline rebuild): this converter still only distinguishes
MOVE vs NOOP. It does NOT emit BUILD samples, and it does NOT enrich units
with name/weapon/type data.

Why these weren't fixed here (rather than silently guessed):
  1. Build detection: BAR's engine command ID for a build order needs to be
     confirmed against whatever CmdID Hooked actually logged for build
     orders in your recordings (the live socket protocol's own build command
     is a custom string "C: BU I ..." — a different scheme entirely — so it
     can't be inferred from GameMessager.py). Set BUILD_CMD_ID below once
     you've confirmed it against a sample recording, and the branch in
     _parse_commands()/convert_to_replay() will start emitting BUILD
     samples. Left as None (disabled) so this never mislabels data.
  2. Unit enrichment: Hooked's runtime_data.txt lines (see _parse_units)
     only ever contained id/range/health/position — there is no unit name,
     so there's no reliable way to recover weapon_type/is_constructing/etc.
     for legacy recordings. offline_train.py's live encoder falls back to
     unit_defs' "_default" entry for these, which is a known fidelity loss
     specific to Hooked-sourced data, not a bug to fix here.

Everything else (loading, MOVE/NOOP sample emission) is unchanged from the
original and still works, since it doesn't depend on any of the live
agent's newer per-unit fields.
"""

import json
import csv
import re
import random
from pathlib import Path
from typing import List, Dict, Tuple

# Set this to BAR's real build-order CmdID once confirmed (see module
# docstring). Leaving it None disables BUILD emission entirely — samples
# that were actually build orders will keep showing up as MOVE, which is
# the previous (safe, if incomplete) behavior.
BUILD_CMD_ID = None


class HookedDataConverter:
    """Converts Hooked game recording files into training replay format"""
    
    def __init__(self, base_filename: str, noop_keep_ratio: float = 0.2):
        """
        Initialize converter with base filename (without extension).
        Expects three files:
        - {base_filename}_map_heights.csv
        - {base_filename}_map_spots.csv
        - {base_filename}_runtime_data.txt
        """
        self.base_path = Path(base_filename)
        self.heights_file = Path(f"{base_filename}_map_heights.csv")
        self.spots_file = Path(f"{base_filename}_map_spots.csv")
        self.runtime_file = Path(f"{base_filename}_runtime_data.txt")
        self.noop_keep_ratio = max(0.0, min(1.0, noop_keep_ratio))
        
        self.map_heights = None
        self.map_spots = []
        self.runtime_data = []
        self.map_width = 0
        self.map_height = 0
        
        self.load_data()
    
    def load_data(self):
        """Load all three data files"""
        print(f"Loading Hooked data from {self.base_path}...")
        
        if not self.heights_file.exists():
            raise FileNotFoundError(f"Map heights file not found: {self.heights_file}")
        if not self.spots_file.exists():
            raise FileNotFoundError(f"Map spots file not found: {self.spots_file}")
        if not self.runtime_file.exists():
            raise FileNotFoundError(f"Runtime data file not found: {self.runtime_file}")
        
        self._load_heights()
        self._load_spots()
        self._load_runtime()
        
        print(f"Loaded {len(self.map_spots)} mass spots")
        print(f"Loaded {len(self.runtime_data)} updates from runtime data")
    
    def _load_heights(self):
        """Load map height data from CSV"""
        print("Loading map heights...")
        with open(self.heights_file, 'r') as f:
            self.map_heights = {}
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("MapSizeX"):
                    try:
                        self.map_width = int(line.split(',')[1])
                    except ValueError:
                        pass
                    continue
                if line.startswith("MapSizeZ"):
                    try:
                        self.map_height = int(line.split(',')[1])
                    except ValueError:
                        pass
                    continue
                if line.startswith("WaterLevel") or line.startswith("x,z,height"):
                    continue

                parts = line.split(',')
                if len(parts) >= 3:
                    try:
                        x = int(float(parts[0]))
                        z = int(float(parts[1]))
                        h = float(parts[2])
                        self.map_heights[(x, z)] = h
                    except ValueError:
                        continue
    
    def _load_spots(self):
        """Load metal/mass spot data from CSV"""
        print("Loading mass spots...")
        with open(self.spots_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    spot = {
                        'type': row['type'],
                        'index': int(row['index']),
                        'x': float(row['x']),
                        'z': float(row['z'])
                    }
                    self.map_spots.append((spot['x'], spot['z']))
                except (ValueError, KeyError):
                    continue
    
    def _load_runtime(self):
        """Load runtime updates from text file"""
        print("Loading runtime data...")
        with open(self.runtime_file, 'r') as f:
            content = f.read()
        
        # Split by UPDATE markers
        updates = re.split(r'UPDATE:\s*([\d.]+)', content)
        
        for i in range(1, len(updates), 2):
            try:
                timestamp = float(updates[i])
                data_block = updates[i + 1] if i + 1 < len(updates) else ""
                
                update = {
                    'timestamp': timestamp,
                    'commands': self._parse_commands(data_block, "Commands:"),
                    'enemy_units': self._parse_units(data_block, "Enemy Units:"),
                    'friendly_units': self._parse_units(data_block, "Friendly Units:"),
                    'known_enemy_units': self._parse_units(data_block, "Known Enemy Units:")
                }
                self.runtime_data.append(update)
            except (ValueError, IndexError):
                continue

    def _extract_section(self, data_block: str, section_header: str) -> str:
        if section_header not in data_block:
            return ""

        start = data_block.find(section_header) + len(section_header)
        end = len(data_block)

        for next_header in ["Commands:", "Enemy Units:", "Friendly Units:", "Known Enemy Units:", "END"]:
            pos = data_block.find(next_header, start)
            if pos > start:
                end = min(end, pos)

        return data_block[start:end].strip()

    def _parse_units(self, data_block: str, section_header: str) -> List[Dict]:
        """Parse unit lines like: Friendly 1: 18720, Range: 300.00, Health: 3700.00, Position: (125.00, 7.91, 1544.00)"""
        units = []
        section_text = self._extract_section(data_block, section_header)
        if not section_text:
            return units

        unit_re = re.compile(
            r"\s*\w+\s*\d*:\s*(\d+),\s*Range:\s*([-\d.]+),\s*Health:\s*([-\d.]+),\s*Position:\s*\(([^)]+)\)"
        )

        for line in section_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            match = unit_re.search(line)
            if not match:
                continue

            try:
                unit_id = int(match.group(1))
                unit_range = float(match.group(2))
                unit_health = float(match.group(3))
                pos_parts = [p.strip() for p in match.group(4).split(',')]
                if len(pos_parts) >= 3:
                    x = float(pos_parts[0])
                    y = float(pos_parts[1])
                    z = float(pos_parts[2])
                else:
                    continue

                # NOTE: 'name' stays generic — Hooked's log format never
                # captured unit type, so live-encoder enrichment (unit_defs)
                # will resolve these through the "_default" fallback entry.
                units.append({
                    'id': unit_id,
                    'name': 'unit',
                    'x': x,
                    'y': y,
                    'z': z,
                    'range': unit_range,
                    'health': unit_health
                })
            except ValueError:
                continue

        return units

    def _parse_commands(self, data_block: str, section_header: str) -> List[Dict]:
        """Parse command lines like: Command 1: UnitID: 18720, CmdID: 10, CmdParams: 55.7,4.1,1946.7, CmdOptions:"""
        commands = []
        section_text = self._extract_section(data_block, section_header)
        if not section_text:
            return commands

        for line in section_text.split('\n'):
            line = line.strip()
            if not line:
                continue

            if "UnitID:" not in line or "CmdID:" not in line:
                continue

            try:
                unit_id_match = re.search(r"UnitID:\s*(\d+)", line)
                cmd_id_match = re.search(r"CmdID:\s*([-\d]+)", line)
                params_match = re.search(r"CmdParams:\s*([^,]*?(?:,[^,]*?)*)\s*,\s*CmdOptions", line)

                if not unit_id_match or not cmd_id_match:
                    continue

                unit_id = int(unit_id_match.group(1))
                cmd_id = int(cmd_id_match.group(1))

                params = []
                if params_match:
                    params_str = params_match.group(1)
                    for p in params_str.split(','):
                        p = p.strip()
                        if p:
                            params.append(float(p))

                commands.append({
                    'unit_id': unit_id,
                    'cmd_id': cmd_id,
                    'params': params
                })
            except ValueError:
                continue

        return commands
    
    def convert_to_replay(self, output_file: str = None) -> Dict:
        """
        Convert Hooked data to imitation dataset format.

        Produces samples with full unit context + the human command target.
        Offline training can use these as expert demonstrations.
        
        Args:
            output_file: Optional path to save JSON file. If None, only returns data.
        
        Returns:
            Dictionary with 'metadata' and 'samples' keys
        """
        print("Converting to replay format...")

        samples = []
        build_samples_emitted = 0

        for update in self.runtime_data:
            friendly_units = update['friendly_units']
            enemy_units = update['enemy_units']
            known_enemy_units = update['known_enemy_units']

            if not friendly_units:
                continue

            # Merge seen + known enemies, preferring seen when IDs overlap
            seen_ids = {u['id'] for u in enemy_units}
            merged_enemy_units = list(enemy_units)
            for known in known_enemy_units:
                if known['id'] not in seen_ids:
                    merged_enemy_units.append(known)

            commands_by_unit = {}
            for cmd in update['commands']:
                commands_by_unit[cmd['unit_id']] = cmd

            for unit in friendly_units:
                unit_id = unit['id']

                if unit_id in commands_by_unit:
                    cmd = commands_by_unit[unit_id]
                    if len(cmd.get('params', [])) < 3:
                        continue

                    target_x = cmd['params'][0]
                    target_y = cmd['params'][1]
                    target_z = cmd['params'][2]

                    # See BUILD_CMD_ID docstring at top of file: emits a
                    # BUILD sample only once that CmdID has been confirmed.
                    if BUILD_CMD_ID is not None and cmd.get('cmd_id') == BUILD_CMD_ID:
                        action_type = 'BUILD'
                        build_samples_emitted += 1
                    else:
                        action_type = 'MOVE'

                    samples.append({
                        'timestamp': update['timestamp'],
                        'unit_id': unit_id,
                        'friendly_units': friendly_units,
                        'enemy_units': merged_enemy_units,
                        'action': {
                            'type': action_type,
                            'x': target_x,
                            'y': target_y,
                            'z': target_z,
                            'cmd_id': cmd.get('cmd_id')
                        }
                    })
                else:
                    # NOOP sample when no command for this unit
                    if random.random() <= self.noop_keep_ratio:
                        samples.append({
                            'timestamp': update['timestamp'],
                            'unit_id': unit_id,
                            'friendly_units': friendly_units,
                            'enemy_units': merged_enemy_units,
                            'action': {
                                'type': 'NOOP',
                                'x': unit['x'],
                                'y': unit['y'],
                                'z': unit['z']
                            }
                        })

        if BUILD_CMD_ID is None:
            print("[NOTE] BUILD_CMD_ID is unset — all commands emitted as MOVE. See module docstring.")
        else:
            print(f"Emitted {build_samples_emitted} BUILD samples (BUILD_CMD_ID={BUILD_CMD_ID}).")

        output = {
            'metadata': {
                'map_heights_file': str(self.heights_file),
                'map_spots_file': str(self.spots_file),
                'runtime_file': str(self.runtime_file),
                'map_width': self.map_width,
                'map_height': self.map_height
            },
            'samples': samples
        }

        # Save to file if output path is provided
        if output_file is not None:
            output_path = Path(output_file)
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"Saved {len(samples)} samples to {output_file}")
        else:
            print(f"Converted {len(samples)} samples")
        
        return output


def convert_hooked_files(base_filename: str, output_file: str = 'replay.json', noop_keep_ratio: float = 0.2) -> Dict:
    """
    Convenience function to convert Hooked files.
    
    Args:
        base_filename: Base path without extension (e.g., "Hooked 1_20260128_191454")
        output_file: Output JSON file for training (saves to file)
        noop_keep_ratio: Fraction of NOOP samples to keep (0.0 to 1.0)
    
    Returns:
        Dataset dictionary with 'metadata' and 'samples' keys
    """
    converter = HookedDataConverter(base_filename, noop_keep_ratio=noop_keep_ratio)
    return converter.convert_to_replay(output_file)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python hooked_converter.py <base_filename> [output_file] [noop_keep_ratio]")
        print("Example: python hooked_converter.py 'Hooked 1_20260128_191454' replay.json 0.2")
        sys.exit(1)
    
    base_filename = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'replay.json'
    noop_keep_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
    
    try:
        convert_hooked_files(base_filename, output_file, noop_keep_ratio=noop_keep_ratio)
        print(f"\nConversion complete! Use with: python offline_train.py {output_file}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)