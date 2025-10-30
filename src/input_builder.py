# /src/input_builder.py

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'data', 'testing-input-output'))

ENRICHED_METADATA_FILE = 'enriched_metadata.json'
FLOW_DATA_FILE = 'flow_data.json'
BOUNDARY_CONDITIONS_FILE = 'boundary_conditions_gmsh.json'
GEOMETRY_MASKING_FILE = 'geometry_masking_gmsh.json'
OUTPUT_FILE = 'fluid_simulation_input.json'


def load_json_file(path):
    """Load JSON from a file, raising a clear error if missing or invalid."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {path}: {e}") from e


def align_domain_keys(domain: dict) -> dict:
    """
    Aligns legacy domain keys (min_x, max_x, ...) to canonical format (x_min, x_max, ...).
    """
    key_map = {
        "min_x": "x_min", "max_x": "x_max",
        "min_y": "y_min", "max_y": "y_max",
        "min_z": "z_min", "max_z": "z_max"
    }
    aligned = {}
    for old_key, new_key in key_map.items():
        if old_key in domain:
            aligned[new_key] = domain[old_key]
        elif new_key in domain:
            aligned[new_key] = domain[new_key]
        else:
            raise KeyError(f"Missing domain key: {old_key} or {new_key}")
    # Preserve resolution keys
    for key in ["nx", "ny", "nz"]:
        if key not in domain:
            raise KeyError(f"Missing domain resolution key: {key}")
        aligned[key] = domain[key]
    return aligned


def build_fluid_simulation_input():
    """Merge the four input JSON files into the final schema."""
    enriched_metadata = load_json_file(os.path.join(BASE_DIR, ENRICHED_METADATA_FILE))
    flow_data = load_json_file(os.path.join(BASE_DIR, FLOW_DATA_FILE))
    boundary_conditions = load_json_file(os.path.join(BASE_DIR, BOUNDARY_CONDITIONS_FILE))
    geometry_masking = load_json_file(os.path.join(BASE_DIR, GEOMETRY_MASKING_FILE))

    required_keys = [
        (ENRICHED_METADATA_FILE, enriched_metadata, ['domain_definition']),
        (FLOW_DATA_FILE, flow_data, ['fluid_properties', 'initial_conditions', 'simulation_parameters']),
        (GEOMETRY_MASKING_FILE, geometry_masking, ['geometry_mask_flat', 'geometry_mask_shape']),
    ]
    for name, data, keys in required_keys:
        for key in keys:
            if key not in data:
                raise KeyError(f"Missing expected key '{key}' in {name}")

    if "flattening_order" not in geometry_masking:
        geometry_masking["flattening_order"] = "x-major"

    # ✅ Align domain keys before merging
    aligned_domain = align_domain_keys(enriched_metadata["domain_definition"])

    merged = {
        "domain_definition": aligned_domain,
        "fluid_properties": flow_data["fluid_properties"],
        "initial_conditions": flow_data["initial_conditions"],
        "simulation_parameters": flow_data["simulation_parameters"],
        "boundary_conditions": boundary_conditions,
        "geometry_definition": geometry_masking
    }

    output_path = os.path.join(BASE_DIR, OUTPUT_FILE)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2)
    print(f"✅ Merged file written to: {output_path}")


if __name__ == '__main__':
    try:
        build_fluid_simulation_input()
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
