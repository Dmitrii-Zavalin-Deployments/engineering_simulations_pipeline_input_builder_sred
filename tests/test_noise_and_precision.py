# test_noise_and_precision.py

import numpy as np
import pytest
import warnings

# Example placeholder: replace this with your actual simulation function
def simulate_step(u, v, Δt):
    # Simple dummy simulation: shifts inputs slightly
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return u + v * Δt

def test_small_time_step_precision():
    u = np.ones((10, 10)) * 1.0
    v = np.ones((10, 10)) * 1e-8  # tiny velocity
    Δt = 1e-5  # very small time step
    result = simulate_step(u, v, Δt)
    expected = u + v * Δt
    assert np.allclose(result, expected, atol=1e-12)

def test_irrational_velocity_field():
    grid = np.indices((10, 10))
    u = np.sin(np.sqrt(2) * grid[0])  # irrational frequencies
    v = np.cos(np.pi * grid[1])
    Δt = 0.01
    result = simulate_step(u, v, Δt)
    assert np.isfinite(result).all()

def test_high_noise_field():
    u = np.random.normal(loc=0.0, scale=1.0, size=(50, 50))
    v = np.random.normal(loc=0.0, scale=5.0, size=(50, 50))
    Δt = 0.05
    result = simulate_step(u, v, Δt)
    assert np.abs(result).mean() < 5.0  # sanity check
    assert np.isfinite(result).all()

def test_nan_input_propagation():
    u = np.zeros((10, 10))
    v = np.ones((10, 10))
    v[3, 4] = np.nan
    Δt = 0.1
    result = simulate_step(u, v, Δt)
    assert np.isnan(result[3, 4])
    assert not np.isnan(result).all()

def test_inf_input_handling():
    u = np.zeros((10, 10))
    v = np.zeros((10, 10))
    u[0, 0] = np.inf
    v[1, 1] = -np.inf
    Δt = 0.1
    result = simulate_step(u, v, Δt)
    assert np.isinf(result[0, 0])
    assert np.isinf(result[1, 1])
    assert not np.isinf(result).all()



