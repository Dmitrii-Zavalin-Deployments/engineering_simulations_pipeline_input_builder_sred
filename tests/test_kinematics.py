import numpy as np
import tempfile
import os
import json

from src.generate_blender_format import generate_fluid_particles  # ✅ Updated import path


def write_inputs(tmpdir, velocity, coords, time, metadata):
    np.save(os.path.join(tmpdir, "velocity_history.npy"), velocity)
    np.save(os.path.join(tmpdir, "nodes_coords.npy"), coords)
    np.save(os.path.join(tmpdir, "time_points.npy"), time)
    with open(os.path.join(tmpdir, "grid_metadata.json"), "w") as f:
        json.dump(metadata, f)


def load_output(path):
    return np.load(path, allow_pickle=True).item()


def test_single_time_step_only():
    velocity = np.array([[[1.0, 1.0, 1.0]]])
    coords = np.array([[0.0, 0.0, 0.0]])
    time = np.array([2.0])
    metadata = {}

    expected = coords[0] + velocity[0, 0] * time[0]

    with tempfile.TemporaryDirectory() as tmp:
        write_inputs(tmp, velocity, coords, time, metadata)
        out_path = os.path.join(tmp, "output.npy")
        generate_fluid_particles(
            os.path.join(tmp, "velocity_history.npy"),
            os.path.join(tmp, "nodes_coords.npy"),
            os.path.join(tmp, "time_points.npy"),
            os.path.join(tmp, "grid_metadata.json"),
            out_path,
        )
        result = load_output(out_path)
        actual = result["particles_motion"][0]["positions"][0]
        np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_multiple_particles_uniform_velocity():
    velocity = np.array([
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    ])
    coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    time = np.array([1.0, 2.0])
    metadata = {}

    with tempfile.TemporaryDirectory() as tmp:
        write_inputs(tmp, velocity, coords, time, metadata)
        out_path = os.path.join(tmp, "res.npy")
        generate_fluid_particles(
            os.path.join(tmp, "velocity_history.npy"),
            os.path.join(tmp, "nodes_coords.npy"),
            os.path.join(tmp, "time_points.npy"),
            os.path.join(tmp, "grid_metadata.json"),
            out_path,
        )
        result = load_output(out_path)
        for i in range(2):
            pos = np.array(result["particles_motion"][i]["positions"])
            expected = [coords[i] + velocity[j, i] * time[j] for j in range(2)]
            np.testing.assert_allclose(pos, expected, rtol=1e-6)


def test_zero_velocity_all_particles_stationary():
    velocity = np.zeros((3, 2, 3))
    coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    time = np.array([0.0, 1.0, 2.0])
    metadata = {}

    with tempfile.TemporaryDirectory() as tmp:
        write_inputs(tmp, velocity, coords, time, metadata)
        out_path = os.path.join(tmp, "zero_out.npy")
        generate_fluid_particles(
            os.path.join(tmp, "velocity_history.npy"),
            os.path.join(tmp, "nodes_coords.npy"),
            os.path.join(tmp, "time_points.npy"),
            os.path.join(tmp, "grid_metadata.json"),
            out_path,
        )
        result = load_output(out_path)
        for i in range(2):
            pos = np.array(result["particles_motion"][i]["positions"])
            for p in pos:
                np.testing.assert_allclose(p, coords[i], rtol=1e-6)


def test_negative_velocity_backwards_motion():
    velocity = np.array([
        [[-1.0, 0.0, 0.0]],
        [[-2.0, 0.0, 0.0]],
    ])
    coords = np.array([[5.0, 0.0, 0.0]])
    time = np.array([1.0, 2.0])
    metadata = {}

    expected = [coords[0] + velocity[i, 0] * time[i] for i in range(2)]

    with tempfile.TemporaryDirectory() as tmp:
        write_inputs(tmp, velocity, coords, time, metadata)
        out_path = os.path.join(tmp, "neg_out.npy")
        generate_fluid_particles(
            os.path.join(tmp, "velocity_history.npy"),
            os.path.join(tmp, "nodes_coords.npy"),
            os.path.join(tmp, "time_points.npy"),
            os.path.join(tmp, "grid_metadata.json"),
            out_path,
        )
        result = load_output(out_path)
        actual = np.array(result["particles_motion"][0]["positions"])
        np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_exact_position_update_multiple_steps():
    velocity = np.array([
        [[2.0, 0.0, 0.0]],
        [[2.0, 0.0, 0.0]],
        [[2.0, 0.0, 0.0]]
    ])
    coords = np.array([[1.0, 1.0, 1.0]])
    time = np.array([0.0, 1.0, 2.0])
    metadata = {}

    expected_positions = [coords[0] + velocity[i, 0] * time[i] for i in range(3)]

    with tempfile.TemporaryDirectory() as tmp:
        write_inputs(tmp, velocity, coords, time, metadata)
        out_path = os.path.join(tmp, "output.npy")
        generate_fluid_particles(
            os.path.join(tmp, "velocity_history.npy"),
            os.path.join(tmp, "nodes_coords.npy"),
            os.path.join(tmp, "time_points.npy"),
            os.path.join(tmp, "grid_metadata.json"),
            out_path,
        )
        result = load_output(out_path)
        actual_positions = np.array(result["particles_motion"][0]["positions"])
        np.testing.assert_allclose(actual_positions, expected_positions, rtol=1e-6)



