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
    output_file = os.path.join(workspace_dir, output_file) # Output file will now be .npy

    # Debugging: Print resolved paths
    print(f"🔎 Checking file paths:")
    print(f"  - Velocity file: {velocity_file}")
    print(f"  - Nodes file: {nodes_file}")
    print(f"  - Time file: {time_file}")
    print(f"  - Grid Metadata file: {grid_metadata_file}")
    print(f"  - Output file: {output_file}")

    # Ensure required files exist before proceeding
    for file_path in [velocity_file, nodes_file, time_file, grid_metadata_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ Error: Required file not found - {file_path}")

    # Load velocity, nodes, and time step data
    velocity_history = np.load(velocity_file)

    # Debugging: Print actual shape of velocity history
    print(f"🔍 velocity_history shape: {velocity_history.shape}")

    # Ensure correct unpacking of shape dynamically
    num_timesteps = velocity_history.shape[0]  # Time steps
    # Compute total nodes dynamically, assuming last dimension is velocity components (e.g., 3 for 3D)
    # The reshape below assumes the velocity history is (timesteps, num_nodes, 3) or similar
    # If velocity_history.shape is (num_timesteps, dim1, dim2, ..., 3), then num_nodes is dim1 * dim2 * ...
    if velocity_history.ndim > 2:
        num_nodes = np.prod(velocity_history.shape[1:-1])
    else: # If velocity_history.shape is (num_timesteps, num_nodes * 3) or (num_timesteps, num_nodes, 3)
        num_nodes = velocity_history.shape[1] if velocity_history.ndim == 2 else velocity_history.shape[1] // 3


    # Load node coordinates
    nodes_coords = np.load(nodes_file)
    print(f"🔍 nodes_coords shape: {nodes_coords.shape}")  # Debug node shape

    time_steps = np.load(time_file)  # Shape: (num_timesteps,)
    
    # Load grid metadata
    with open(grid_metadata_file, "r") as f:
        grid_metadata = json.load(f)

    # Ensure all data dimensions are consistent
    # nodes_coords should be (num_nodes, 3)
    if nodes_coords.shape[0] != num_nodes and nodes_coords.size // 3 == num_nodes:
        print(f"⚠️ Attempting to reshape nodes_coords from {nodes_coords.shape} to ({num_nodes}, 3)...")
        nodes_coords = nodes_coords.reshape((num_nodes, 3))
    elif nodes_coords.shape[0] != num_nodes:
        raise ValueError(f"❌ Mismatch! Expected {num_nodes} nodes based on velocity history, but found {nodes_coords.shape[0]} in nodes_coords after potential reshape.")

    # Initialize particle data structure.
    # Since we're saving to .npy, we'll aim for structured NumPy arrays or a dictionary of arrays.
    # Storing a complex nested dictionary directly in .npy isn't ideal for performance/readability.
    # Instead, we'll create separate arrays for initial_positions, times, positions, and velocities.

    all_initial_positions = np.zeros((num_nodes, 3), dtype=np.float64)
    all_motion_data = [] # List to store motion data for each particle

    for node_id in range(num_nodes):
        all_initial_positions[node_id] = nodes_coords[node_id]

        # Prepare arrays for current particle's motion
        particle_times = np.zeros(num_timesteps, dtype=np.float64)
        particle_positions = np.zeros((num_timesteps, 3), dtype=np.float64)
        particle_velocities = np.zeros((num_timesteps, 3), dtype=np.float64)

        for timestep_idx, time in enumerate(time_steps):
            # Reshape velocities dynamically. Assuming velocity_history[timestep_idx] has data for all nodes.
            # It might be (num_nodes, 3) or (grid_dim1, grid_dim2, ..., 3)
            current_velocities_timestep = velocity_history[timestep_idx].reshape(num_nodes, 3)
            velocity = current_velocities_timestep[node_id]
            new_position = nodes_coords[node_id] + velocity * time # Using time as a scalar, not time_steps array.

            particle_times[timestep_idx] = time
            particle_positions[timestep_idx] = new_position
            particle_velocities[timestep_idx] = velocity

        all_motion_data.append({
            "id": node_id,
            "times": particle_times,
            "positions": particle_positions,
            "velocities": particle_velocities
        })

    # Save particle motion data to NPY.
    # We'll save a dictionary that contains the grid_metadata and then a structured array or
    # a list of dictionaries (which numpy can save as an object array).
    # Saving a list of dictionaries as an object array with np.save() is straightforward.
    
    output_data = {
        "grid_metadata": grid_metadata,
        "initial_positions": all_initial_positions,
        "particles_motion": all_motion_data # This will be saved as an object array if elements are dictionaries
    }

    try:
        np.save(output_file, output_data)
        print(f"✅ Fluid particle data saved to {output_file}")
    except Exception as e:
        print(f"❌ Error saving data to {output_file}: {e}")


# Example usage
if __name__ == "__main__":
    generate_fluid_particles(
        velocity_file="data/testing-input-output/velocity_history.npy",
        nodes_file="data/testing-input-output/nodes_coords.npy",
        time_file="data/testing-input-output/time_points.npy",
        grid_metadata_file="data/testing-input-output/grid_metadata.json",
        output_file="data/testing-input-output/fluid_particles.npy" # Changed output to .npy
    )



