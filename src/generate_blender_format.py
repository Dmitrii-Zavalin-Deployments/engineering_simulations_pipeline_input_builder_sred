import numpy as np
import json
import os

def generate_fluid_particles(velocity_file, nodes_file, time_file, grid_metadata_file, output_file):
    """
    Generate fluid particle motion data based on velocity history.

    Args:
        velocity_file (str): Path to velocity history `.npy` file.
        nodes_file (str): Path to nodes coordinates `.npy` file.
        time_file (str): Path to simulation time steps `.npy` file.
        grid_metadata_file (str): Path to grid metadata `.json` file.
        output_file (str): Path to output fluid particle `.npy` file.
    """

    # Resolve absolute paths using GITHUB_WORKSPACE
    workspace_dir = os.getenv("GITHUB_WORKSPACE", ".")
    velocity_file = os.path.join(workspace_dir, velocity_file)
    nodes_file = os.path.join(workspace_dir, nodes_file)
    time_file = os.path.join(workspace_dir, time_file)
    grid_metadata_file = os.path.join(workspace_dir, grid_metadata_file)
    output_file = os.path.join(workspace_dir, output_file)

    print(f"🔎 Checking file paths:")
    print(f"  - Velocity file: {velocity_file}")
    print(f"  - Nodes file: {nodes_file}")
    print(f"  - Time file: {time_file}")
    print(f"  - Grid Metadata file: {grid_metadata_file}")
    print(f"  - Output file: {output_file}")

    for file_path in [velocity_file, nodes_file, time_file, grid_metadata_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ Error: Required file not found - {file_path}")

    velocity_history = np.load(velocity_file)
    nodes_coords = np.load(nodes_file)
    time_steps = np.load(time_file)

    print(f"🔍 velocity_history shape: {velocity_history.shape}")
    print(f"🔍 nodes_coords shape: {nodes_coords.shape}")

    # Validate non-empty inputs
    if velocity_history.size == 0 or nodes_coords.size == 0 or time_steps.size == 0:
        raise ValueError("❌ One or more input arrays are empty.")

    if np.any(time_steps < 0):
        raise ValueError("❌ Negative values detected in time_points.")

    if velocity_history.shape[0] != time_steps.shape[0]:
        raise ValueError("❌ Time steps mismatch: velocity_history and time_points must have the same number of steps.")

    # Load grid metadata
    with open(grid_metadata_file, "r") as f:
        grid_metadata = json.load(f)

    num_timesteps = velocity_history.shape[0]

    if velocity_history.ndim > 2:
        num_nodes = np.prod(velocity_history.shape[1:-1])
        if velocity_history.shape[-1] != 3:
            raise ValueError("❌ Velocity vectors must have 3 components per node.")
    else:
        num_nodes = velocity_history.shape[1] if velocity_history.ndim == 2 else velocity_history.shape[1] // 3

    if nodes_coords.shape[0] != num_nodes and nodes_coords.size // 3 == num_nodes:
        print(f"⚠️ Attempting to reshape nodes_coords from {nodes_coords.shape} to ({num_nodes}, 3)...")
        nodes_coords = nodes_coords.reshape((num_nodes, 3))
    elif nodes_coords.shape[0] != num_nodes:
        raise ValueError(f"❌ Mismatch! Expected {num_nodes} nodes based on velocity history, but found {nodes_coords.shape[0]} in nodes_coords.")

    all_initial_positions = np.zeros((num_nodes, 3), dtype=np.float64)
    all_motion_data = []

    for node_id in range(num_nodes):
        all_initial_positions[node_id] = nodes_coords[node_id]

        particle_times = np.zeros(num_timesteps, dtype=np.float64)
        particle_positions = np.zeros((num_timesteps, 3), dtype=np.float64)
        particle_velocities = np.zeros((num_timesteps, 3), dtype=np.float64)

        for timestep_idx, time in enumerate(time_steps):
            current_velocities_timestep = velocity_history[timestep_idx].reshape(num_nodes, 3)
            velocity = current_velocities_timestep[node_id]
            new_position = nodes_coords[node_id] + velocity * time

            particle_times[timestep_idx] = time
            particle_positions[timestep_idx] = new_position
            particle_velocities[timestep_idx] = velocity

        all_motion_data.append({
            "id": node_id,
            "times": particle_times,
            "positions": particle_positions,
            "velocities": particle_velocities
        })

    output_data = {
        "grid_metadata": grid_metadata,
        "initial_positions": all_initial_positions,
        "particles_motion": all_motion_data
    }

    try:
        np.save(output_file, output_data)
        print(f"✅ Fluid particle data saved to {output_file}")
    except Exception as e:
        print(f"❌ Error saving data to {output_file}: {e}")


if __name__ == "__main__":
    generate_fluid_particles(
        velocity_file="data/testing-input-output/velocity_history.npy",
        nodes_file="data/testing-input-output/nodes_coords.npy",
        time_file="data/testing-input-output/time_points.npy",
        grid_metadata_file="data/testing-input-output/grid_metadata.json",
        output_file="data/testing-input-output/fluid_particles.npy"
    )



