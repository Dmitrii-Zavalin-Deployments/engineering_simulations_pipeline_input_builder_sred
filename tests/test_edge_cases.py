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


def test_single_time_step_with_multiple_particles():
    velocity = np.array([[
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]])  # (1, 3, 3)
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0]
    ])
    time = np.array([2.0])
    metadata = {"test": "single timestep, multi-particle"}

    with tempfile.TemporaryDirectory() as tmp:
        write_inputs(tmp, velocity, coords, time, metadata)
        out_path = os.path.join(tmp, "out.npy")
        generate_fluid_particles(
            os.path.join(tmp, "velocity_history.npy"),
            os.path.join(tmp, "nodes_coords.npy"),
            os.path.join(tmp, "time_points.npy"),
            os.path.join(tmp, "grid_metadata.json"),
            out_path
        )

        result = load_output(out_path)
        for i in range(3):
            expected = coords[i] + velocity[0, i] * time[0]
            actual = result["particles_motion"][i]["positions"][0]
            np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_non_uniform_time_steps():
    velocity = np.array([
        [[1.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0]]
    ])  # (3, 1, 3)
    coords = np.array([[0.0, 0.0, 0.0]])
    time = np.array([0.0, 0.5, 2.5])
    metadata = {"test": "non-uniform time steps"}

    expected_positions = [coords[0] + velocity[i, 0] * time[i] for i in range(len(time))]

    with tempfile.TemporaryDirectory() as tmp:
        write_inputs(tmp, velocity, coords, time, metadata)
        out_path = os.path.join(tmp, "out.npy")
        generate_fluid_particles(
            os.path.join(tmp, "velocity_history.npy"),
            os.path.join(tmp, "nodes_coords.npy"),
            os.path.join(tmp, "time_points.npy"),
            os.path.join(tmp, "grid_metadata.json"),
            out_path
        )

        result = load_output(out_path)
        actual = result["particles_motion"][0]["positions"]
        np.testing.assert_allclose(actual, expected_positions, rtol=1e-6)


def test_varied_particle_velocities_and_trajectories():
    velocity = np.array([
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        [[2.0, 0.0, 0.0], [0.0, -2.0, 0.0]]
    ])  # (2, 2, 3)
    coords = np.array([
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0]
    ])
    time = np.array([1.0, 2.0])
    metadata = {"test": "varied particle motion"}

    expected_0 = [coords[0] + velocity[0, 0] * time[0], coords[0] + velocity[1, 0] * time[1]]
    expected_1 = [coords[1] + velocity[0, 1] * time[0], coords[1] + velocity[1, 1] * time[1]]

    with tempfile.TemporaryDirectory() as tmp:
        write_inputs(tmp, velocity, coords, time, metadata)
        out_path = os.path.join(tmp, "out.npy")
        generate_fluid_particles(
            os.path.join(tmp, "velocity_history.npy"),
            os.path.join(tmp, "nodes_coords.npy"),
            os.path.join(tmp, "time_points.npy"),
            os.path.join(tmp, "grid_metadata.json"),
            out_path
        )

        result = load_output(out_path)
        actual_0 = result["particles_motion"][0]["positions"]
        actual_1 = result["particles_motion"][1]["positions"]
        np.testing.assert_allclose(actual_0, expected_0, rtol=1e-6)
        np.testing.assert_allclose(actual_1, expected_1, rtol=1e-6)



