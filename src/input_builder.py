# /src/input_builder.py

import json
import os
import sys

# Base folder where Dropbox downloads are stored (resolved relative to this script's location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'data', 'testing-input-output'))

# Filenames for the four input JSONs
ENRICHED_METADATA_FILE = 'enriched_metadata.json'
FLOW_DATA_FILE = 'flow_data.json'
BOUNDARY_CONDITIONS_FILE = 'boundary_conditions_gmsh.json'
GEOMETRY_MASKING_FILE = 'geometry_masking_gmsh.json'

# Output filename
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


def build_fluid_simulation_input():
    """Merge the four input JSON files into the final schema."""
    # Load each source file
    enriched_metadata = load_json_file(os.path.join(BASE_DIR, ENRICHED_METADATA_FILE))
    flow_data = load_json_file(os.path.join(BASE_DIR, FLOW_DATA_FILE))
    boundary_conditions = load_json_file(os.path.join(BASE_DIR, BOUNDARY_CONDITIONS_FILE))
    geometry_masking = load_json_file(os.path.join(BASE_DIR, GEOMETRY_MASKING_FILE))

    # Validate expected keys in each file
    required_keys = [
        (ENRICHED_METADATA_FILE, enriched_metadata, ['domain_definition']),
        (FLOW_DATA_FILE, flow_data, ['fluid_properties', 'initial_conditions', 'simulation_parameters']),
        # boundary_conditions can be free-form depending on mesh/BC setup
        (GEOMETRY_MASKING_FILE, geometry_masking, ['geometry_mask_flat', 'geometry_mask_shape']),
    ]
    for name, data, keys in required_keys:
        for key in keys:
            if key not in data:
                raise KeyError(f"Missing expected key '{key}' in {name}")

    # Adapt geometry_masking_gmsh.json into a geometry_definition block
    geometry_definition = {
        "mask": {
            "values_flat": geometry_masking["geometry_mask_flat"],
            "shape": geometry_masking["geometry_mask_shape"],
            "encoding": geometry_masking.get("mask_encoding"),
            "flattening_order": geometry_masking.get("flattening_order", "x-major"),
        },
        "source": "gmsh_mask_v1"
    }

    # Merge into final structure
    merged = {
        "domain_definition": enriched_metadata["domain_definition"],
        "fluid_properties": flow_data["fluid_properties"],
        "initial_conditions": flow_data["initial_conditions"],
        "simulation_parameters": flow_data["simulation_parameters"],
        "boundary_conditions": boundary_conditions,
        "geometry_definition": geometry_definition
    }

    # Write output file (overwrite if exists)
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



