import numpy as np
import tempfile
import os
import json

from src.generate_blender_format import generate_fluid_particles


def write_inputs(tmpdir, velocity, coords, time, metadata):
    np.save(os.path.join(tmpdir, "velocity_history.npy"), velocity)
    np.save(os.path.join(tmpdir, "nodes_coords.npy"), coords)
    np.save(os.path.join(tmpdir, "time_points.npy"), time)
    with open(os.path.join(tmpdir, "grid_metadata.json"), "w") as f:
        json.dump(metadata, f)


def load_output(path):
    return np.load(path, allow_pickle=True).item()


def test_output_structure_and_keys():
    velocity = np.ones((3, 2, 3))
    coords = np.array([[0, 0, 0], [1, 1, 1]])
    time = np.array([0.0, 1.0, 2.0])
    metadata = {"description": "test output structure"}

    with tempfile.TemporaryDirectory() as tmp:
        write_inputs(tmp, velocity, coords, time, metadata)
        out_path = os.path.join(tmp, "result.npy")
        generate_fluid_particles(
            os.path.join(tmp, "velocity_history.npy"),
            os.path.join(tmp, "nodes_coords.npy"),
            os.path.join(tmp, "time_points.npy"),
            os.path.join(tmp, "grid_metadata.json"),
            out_path
        )

        result = load_output(out_path)

        # Top-level keys
        assert "grid_metadata" in result
        assert "initial_positions" in result
        assert "particles_motion" in result

        # Type and shape checks
        assert isinstance(result["particles_motion"], list)
        assert result["initial_positions"].shape == coords.shape
        np.testing.assert_allclose(result["initial_positions"], coords)
        assert result["grid_metadata"] == metadata


def test_particle_entry_shapes_and_ids():
    velocity = np.array([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    ])  # (3 time, 2 particles, 3D)
    coords = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    time = np.array([0.0, 1.0, 2.0])
    metadata = {}

    with tempfile.TemporaryDirectory() as tmp:
        write_inputs(tmp, velocity, coords, time, metadata)
        out_path = os.path.join(tmp, "test.npy")
        generate_fluid_particles(
            os.path.join(tmp, "velocity_history.npy"),
            os.path.join(tmp, "nodes_coords.npy"),
            os.path.join(tmp, "time_points.npy"),
            os.path.join(tmp, "grid_metadata.json"),
            out_path
        )

        result = load_output(out_path)
        motion = result["particles_motion"]

        assert len(motion) == 2

        for i, particle in enumerate(motion):
            assert particle["id"] == i
            assert particle["positions"].shape == (3, 3)
            assert particle["velocities"].shape == (3, 3)
            assert particle["times"].shape == (3,)
            assert np.allclose(particle["times"], time)



