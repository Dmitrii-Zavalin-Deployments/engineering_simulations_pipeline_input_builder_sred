import numpy as np
import pytest
import tempfile
import os
import json

from src.generate_blender_format import generate_fluid_particles


def write_inputs(tmp, velocity, coords, time, metadata):
    np.save(os.path.join(tmp, "velocity_history.npy"), velocity)
    np.save(os.path.join(tmp, "nodes_coords.npy"), coords)
    np.save(os.path.join(tmp, "time_points.npy"), time)
    with open(os.path.join(tmp, "grid_metadata.json"), "w") as f:
        json.dump(metadata, f)


@pytest.mark.parametrize("velocity, coords, time", [
    (np.empty((0, 0, 3)), np.empty((0, 3)), np.array([])),     # Empty arrays
    (np.ones((3, 5, 3)), np.ones((4, 3)), np.ones(3)),         # Mismatched node counts
    (np.ones((4, 1, 3)), np.ones((1, 3)), np.ones(3)),         # Time mismatch
    (np.ones((3, 1, 2)), np.ones((1, 3)), np.ones(3)),         # Invalid velocity shape (not 3D)
])
def test_invalid_shapes_raise(velocity, coords, time):
    metadata = {}
    with tempfile.TemporaryDirectory() as tmp:
        write_inputs(tmp, velocity, coords, time, metadata)
        with pytest.raises(ValueError):
            generate_fluid_particles(
                os.path.join(tmp, "velocity_history.npy"),
                os.path.join(tmp, "nodes_coords.npy"),
                os.path.join(tmp, "time_points.npy"),
                os.path.join(tmp, "grid_metadata.json"),
                os.path.join(tmp, "output.npy")
            )


def test_negative_time_raises():
    velocity = np.ones((2, 1, 3))
    coords = np.ones((1, 3))
    time = np.array([0.0, -1.0])
    metadata = {}
    with tempfile.TemporaryDirectory() as tmp:
        write_inputs(tmp, velocity, coords, time, metadata)
        with pytest.raises(ValueError):
            generate_fluid_particles(
                os.path.join(tmp, "velocity_history.npy"),
                os.path.join(tmp, "nodes_coords.npy"),
                os.path.join(tmp, "time_points.npy"),
                os.path.join(tmp, "grid_metadata.json"),
                os.path.join(tmp, "out.npy")
            )


def test_missing_file_error():
    with tempfile.TemporaryDirectory() as tmp:
        # Only create one input file
        coords = np.ones((1, 3))
        np.save(os.path.join(tmp, "nodes_coords.npy"), coords)
        with pytest.raises(FileNotFoundError):
            generate_fluid_particles(
                os.path.join(tmp, "missing_velocity.npy"),
                os.path.join(tmp, "nodes_coords.npy"),
                os.path.join(tmp, "time_points.npy"),
                os.path.join(tmp, "grid_metadata.json"),
                os.path.join(tmp, "out.npy")
            )


def test_malformed_json_raises():
    velocity = np.ones((2, 1, 3))
    coords = np.ones((1, 3))
    time = np.array([0.0, 1.0])
    with tempfile.TemporaryDirectory() as tmp:
        np.save(os.path.join(tmp, "velocity_history.npy"), velocity)
        np.save(os.path.join(tmp, "nodes_coords.npy"), coords)
        np.save(os.path.join(tmp, "time_points.npy"), time)
        bad_json_path = os.path.join(tmp, "grid_metadata.json")
        with open(bad_json_path, "w") as f:
            f.write("{ this is not valid JSON }")
        with pytest.raises(json.JSONDecodeError):
            generate_fluid_particles(
                os.path.join(tmp, "velocity_history.npy"),
                os.path.join(tmp, "nodes_coords.npy"),
                os.path.join(tmp, "time_points.npy"),
                bad_json_path,
                os.path.join(tmp, "output.npy")
            )



