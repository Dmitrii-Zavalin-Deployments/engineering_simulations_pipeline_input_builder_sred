import unittest
import os
import json
import numpy as np
from jsonschema import validate


class TestParticleMotionPreprocessing(unittest.TestCase):

    def setUp(self):
        """Load generated JSON particle motion file"""
        with open("data/testing-input-output/fluid_particles.json") as f:
            self.particle_data = json.load(f)

    def test_json_schema(self):
        """Ensure particle motion data follows defined JSON schema"""
        schema = {
            "type": "object",
            "properties": {
                "particle_info": {"type": "object"},
                "velocity_fields": {"type": "array"},
                "global_parameters": {"type": "object"}
            },
            "required": ["particle_info", "velocity_fields", "global_parameters"]
        }
        validate(instance=self.particle_data, schema=schema)

    def test_velocity_field_integrity(self):
        """Ensure velocity fields are properly extracted"""
        assert len(self.particle_data["velocity_fields"]) > 0, "Velocity fields missing!"
        assert all("vx" in p and "vy" in p and "vz" in p for p in self.particle_data["velocity_fields"]), "Velocity components missing!"

    def test_physical_consistency(self):
        """Ensure fluid properties remain physically realistic"""
        assert self.particle_data["global_parameters"]["pressure"]["value"] > 100000, "Pressure too low!"
        assert self.particle_data["global_parameters"]["turbulence_intensity"]["value"] >= 0, "Turbulence intensity invalid!"

    def test_binary_output_exists(self):
        """Ensure .npy file storing structured particle motion data exists"""
        assert os.path.exists("data/testing-input-output/fluid_particles.npy"), "Binary file not found!"

    def test_binary_data_integrity(self):
        """Ensure extracted numerical data is correctly formatted in .npy file"""
        np_data = np.load("data/testing-input-output/fluid_particles.npy")
        assert np_data.shape[0] > 0, "Invalid particle data structure!"
        assert "vx" in np_data.dtype.names and "vy" in np_data.dtype.names and "vz" in np_data.dtype.names, "Velocity components missing!"


if __name__ == "__main__":
    unittest.main()



