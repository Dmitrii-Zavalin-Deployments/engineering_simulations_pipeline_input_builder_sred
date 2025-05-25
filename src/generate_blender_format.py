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
        output_file (str): Path to output fluid particle JSON file.
    """

    # Resolve absolute paths using GITHUB_WORKSPACE
    workspace_dir = os.getenv("GITHUB_WORKSPACE", ".")
    velocity_file = os.path.join(workspace_dir, velocity_file)
    nodes_file = os.path.join(workspace_dir, nodes_file)
    time_file = os.path.join(workspace_dir, time_file)
    grid_metadata_file = os.path.join(workspace_dir, grid_metadata_file)
    output_file = os.path.join(workspace_dir, output_file)

    # Debugging: Print resolved paths
    print(f"🔎 Checking file paths:")
    print(f"  - Velocity file: {velocity_file}")
    print(f"  - Nodes file: {nodes_file}")
    print(f"  - Time file: {time_file}")
    print(f"  - Grid Metadata file: {grid_metadata_file}")

    # Ensure required files exist before proceeding
    for file_path in [velocity_file, nodes_file, time_file, grid_metadata_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ Error: Required file not found - {file_path}")

    # Load velocity, nodes, and time step data
    velocity_history = np.load(velocity_file)

    # Debugging: Print actual shape of velocity history
    print(f"🔍 velocity_history shape: {velocity_history.shape}")

    # Ensure correct unpacking of shape
    num_timesteps, num_nodes = velocity_history.shape[:2]  # Only take first two dimensions

    nodes_coords = np.load(nodes_file)  # Shape: (num_nodes, 3)
    time_steps = np.load(time_file)  # Shape: (num_timesteps,)
    
    # Load grid metadata
    with open(grid_metadata_file, "r") as f:
        grid_metadata = json.load(f)

    # Ensure all data dimensions are consistent
    if nodes_coords.shape[0] != num_nodes:
        raise ValueError("Mismatch between velocity data and node coordinates!")

    # Initialize particle data structure
    particles = []
    
    for node_id in range(num_nodes):
        particle = {
            "id": node_id,
            "initial_position": nodes_coords[node_id].tolist(),
            "motion": []
        }

        for timestep_idx, time in enumerate(time_steps):
            velocity = velocity_history[timestep_idx, node_id].tolist()
            new_position = (nodes_coords[node_id] + np.array(velocity) * time).tolist()

            particle["motion"].append({
                "time": float(time),
                "position": new_position,
                "velocity": velocity
            })

        particles.append(particle)

    # Save particle motion data to JSON
    output_data = {
        "grid_metadata": grid_metadata,
        "particles": particles
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"✅ Fluid particle data saved to {output_file}")

# Example usage
if __name__ == "__main__":
    generate_fluid_particles(
        velocity_file="data/testing-input-output/velocity_history.npy",
        nodes_file="data/testing-input-output/nodes_coords.npy",
        time_file="data/testing-input-output/time_points.npy",
        grid_metadata_file="data/testing-input-output/grid_metadata.json",
        output_file="data/testing-input-output/fluid_particles.json"
    )
