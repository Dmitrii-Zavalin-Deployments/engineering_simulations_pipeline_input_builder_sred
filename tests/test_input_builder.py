# /tests/test_input_builder.py

import json
import os
import tempfile
import shutil
import pytest

# Import the module under test
import src.input_builder as ib


@pytest.fixture
def temp_base_dir(monkeypatch):
    """Create a temporary BASE_DIR with all required files."""
    tmpdir = tempfile.mkdtemp()
    monkeypatch.setattr(ib, "BASE_DIR", tmpdir)

    # ✅ Legacy domain keys to test alignment
    enriched_metadata = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 1, "ny": 1, "nz": 1
        }
    }
    flow_data = {
        "fluid_properties": {"density": 1.0},
        "initial_conditions": {"velocity": [0, 0, 0]},
        "simulation_parameters": {"timestep": 0.1}
    }
    boundary_conditions = {"inlet": {"velocity": 1.0}}
    geometry_masking = {
        "geometry_mask_flat": [1, 0, 1],
        "geometry_mask_shape": [3, 1, 1],
        "mask_encoding": {"fluid": 1, "solid": 0},
        "flattening_order": "x-major"
    }

    files = {
        ib.ENRICHED_METADATA_FILE: enriched_metadata,
        ib.FLOW_DATA_FILE: flow_data,
        ib.BOUNDARY_CONDITIONS_FILE: boundary_conditions,
        ib.GEOMETRY_MASKING_FILE: geometry_masking
    }
    for fname, content in files.items():
        with open(os.path.join(tmpdir, fname), "w", encoding="utf-8") as f:
            json.dump(content, f)

    yield tmpdir
    shutil.rmtree(tmpdir)


def test_load_json_file_success(temp_base_dir):
    path = os.path.join(temp_base_dir, ib.ENRICHED_METADATA_FILE)
    data = ib.load_json_file(path)
    assert "domain_definition" in data


def test_load_json_file_missing():
    with pytest.raises(FileNotFoundError):
        ib.load_json_file("/nonexistent/path.json")


def test_load_json_file_invalid_json(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{invalid json}")
    with pytest.raises(ValueError):
        ib.load_json_file(str(bad_file))


def test_build_fluid_simulation_input_happy_path(temp_base_dir):
    ib.build_fluid_simulation_input()
    output_path = os.path.join(temp_base_dir, ib.OUTPUT_FILE)
    assert os.path.exists(output_path)
    with open(output_path, encoding="utf-8") as f:
        merged = json.load(f)
    # ✅ Check aligned domain keys
    domain = merged["domain_definition"]
    for key in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "nx", "ny", "nz"]:
        assert key in domain
    assert "fluid_properties" in merged
    assert "geometry_definition" in merged
    assert merged["geometry_definition"]["geometry_mask_flat"] == [1, 0, 1]


@pytest.mark.parametrize("missing_file", [
    ib.ENRICHED_METADATA_FILE,
    ib.FLOW_DATA_FILE,
    ib.BOUNDARY_CONDITIONS_FILE,
    ib.GEOMETRY_MASKING_FILE
])
def test_missing_input_file_raises(temp_base_dir, missing_file):
    os.remove(os.path.join(temp_base_dir, missing_file))
    with pytest.raises(FileNotFoundError):
        ib.build_fluid_simulation_input()


@pytest.mark.parametrize("file_name,missing_key", [
    (ib.ENRICHED_METADATA_FILE, "domain_definition"),
    (ib.FLOW_DATA_FILE, "fluid_properties"),
    (ib.FLOW_DATA_FILE, "initial_conditions"),
    (ib.FLOW_DATA_FILE, "simulation_parameters"),
    (ib.GEOMETRY_MASKING_FILE, "geometry_mask_flat"),
    (ib.GEOMETRY_MASKING_FILE, "geometry_mask_shape"),
])
def test_missing_required_key_raises(temp_base_dir, file_name, missing_key):
    path = os.path.join(temp_base_dir, file_name)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data.pop(missing_key, None)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    with pytest.raises(KeyError):
        ib.build_fluid_simulation_input()


def test_geometry_masking_defaults_flattening_order(temp_base_dir):
    path = os.path.join(temp_base_dir, ib.GEOMETRY_MASKING_FILE)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data.pop("flattening_order", None)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    ib.build_fluid_simulation_input()
    output_path = os.path.join(temp_base_dir, ib.OUTPUT_FILE)
    with open(output_path, encoding="utf-8") as f:
        merged = json.load(f)
    assert merged["geometry_definition"]["flattening_order"] == "x-major"


def test_performance_large_geometry_mask(temp_base_dir):
    path = os.path.join(temp_base_dir, ib.GEOMETRY_MASKING_FILE)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data["geometry_mask_flat"] = [1, 0] * 500000
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    ib.build_fluid_simulation_input()
    output_path = os.path.join(temp_base_dir, ib.OUTPUT_FILE)
    with open(output_path, encoding="utf-8") as f:
        merged = json.load(f)
    assert len(merged["geometry_definition"]["geometry_mask_flat"]) == 1000000
