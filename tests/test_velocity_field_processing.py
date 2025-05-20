import unittest
import os
import json
import numpy as np
from jsonschema import validate


class TestParticleMotionProcessing(unittest.TestCase):

    def setUp(self):
        """Define input and output file paths for testing."""
        self.input_npy_path = "data/testing-input-output/fluid_simulation.npy"
        self.output_json_path = "data/testing-input-output/fluid_particles.json"
        self.particle_data_path = "data/testing-input-output/fluid_particles.json"
        self.binary_particles_path = "data/testing-input-output/fluid_particles.npy"

        # Ensure a dummy .npy file exists for testing purposes
        if not os.path.exists(self.input_npy_path):
            np.save(self.input_npy_path, np.array([{'velocity': [1.0, 0.0, 0.0]}, {'velocity': [0.0, 1.0, 0.0]}]))

    ### JSON PARTICLE MOTION VALIDATION ###

    def test_json_schema(self):
        """Ensure particle motion data follows defined JSON schema"""
        with open(self.particle_data_path) as f:
            particle_data = json.load(f)

        schema = {
            "type": "object",
            "properties": {
                "particle_info": {"type": "object"},
                "velocity_fields": {"type": "array"},
                "global_parameters": {"type": "object"}
            },
            "required": ["particle_info", "velocity_fields", "global_parameters"]
        }
        validate(instance=particle_data, schema=schema)

    def test_velocity_field_integrity(self):
        """Ensure velocity fields are properly extracted"""
        with open(self.particle_data_path) as f:
            particle_data = json.load(f)

        assert len(particle_data["velocity_fields"]) > 0, "Velocity fields missing!"
        assert all("vx" in p and "vy" in p and "vz" in p for p in particle_data["velocity_fields"]), "Velocity components missing!"

    def test_physical_consistency(self):
        """Ensure fluid properties remain physically realistic"""
        with open(self.particle_data_path) as f:
            particle_data = json.load(f)

        assert particle_data["global_parameters"]["pressure"]["value"] > 100000, "Pressure too low!"
        assert particle_data["global_parameters"]["turbulence_intensity"]["value"] >= 0, "Turbulence intensity invalid!"

    ### PARTICLE CONVERSION VALIDATION ###

    def test_json_output_exists(self):
        """Ensure the fluid_particles.json file is created"""
        assert os.path.exists(self.output_json_path), f"JSON output file not found at {self.output_json_path}!"

    def test_json_output_structure(self):
        """Validate the basic structure of the fluid_particles.json file"""
        with open(self.output_json_path, "r") as f:
            particle_data = json.load(f)

        assert isinstance(particle_data, dict), "The root of the JSON should be a dictionary!"
        assert "particle_data" in particle_data, "The JSON should contain a 'particle_data' key!"
        assert isinstance(particle_data["particle_data"], list), "'particle_data' should be a list!"
        if particle_data["particle_data"]:
            assert "position" in particle_data["particle_data"][0], "Each particle should have a 'position'."
            assert isinstance(particle_data["particle_data"][0]["position"], list), "'position' should be a list."
            assert len(particle_data["particle_data"][0]["position"]) == 3, "'position' should have 3 elements (x, y, z)."

    def test_particle_data_content(self):
        """Check if the particle data contains expected information (e.g., position derived from velocity)"""
        with open(self.output_json_path, "r") as f:
            particle_data = json.load(f)

        if particle_data["particle_data"]:
            first_particle = particle_data["particle_data"][0]
            assert "position" in first_particle, "Each particle should have a position field."

    ### BINARY PARTICLE DATA VALIDATION ###

    def test_binary_output_exists(self):
        """Ensure .npy file storing structured particle motion data exists"""
        assert os.path.exists(self.binary_particles_path), "Binary file not found!"

    def test_binary_data_integrity(self):
        """Ensure extracted numerical data is correctly formatted in .npy file"""
        np_data = np.load(self.binary_particles_path)
        assert np_data.shape[0] > 0, "Invalid particle data structure!"
        assert "vx" in np_data.dtype.names and "vy" in np_data.dtype.names and "vz" in np_data.dtype.names, "Velocity components missing!"


if __name__ == "__main__":
    unittest.main()



