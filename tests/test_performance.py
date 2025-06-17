# test_performance.py

import numpy as np
import time
import tracemalloc
import pytest

# Placeholder for your real simulation step logic
def simulate_step(state, velocity, Δt):
    return state + velocity * Δt  # Replace with your actual implementation

@pytest.mark.parametrize("size", [256, 512, 1024])
def test_large_scale_performance(size):
    Δt = 0.01
    state = np.random.rand(size, size).astype(np.float32)
    velocity = np.random.rand(size, size).astype(np.float32)

    print(f"\n📊 Testing grid size: {size}x{size}...")

    tracemalloc.start()
    start_time = time.perf_counter()

    result = simulate_step(state, velocity, Δt)

    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    duration_ms = (end_time - start_time) * 1000
    print(f"⏱️ Runtime: {duration_ms:.2f} ms")
    print(f"📈 Peak Memory: {peak / (1024**2):.2f} MB")

    assert result.shape == (size, size)
    assert duration_ms < 2000  # basic sanity constraint
    assert np.isfinite(result).all()

def test_batch_simulation_scalability():
    Δt = 0.005
    batch = 100  # number of simulation steps
    size = 512

    state = np.zeros((size, size), dtype=np.float32)
    velocity = np.random.normal(0, 1, size=(size, size)).astype(np.float32)

    print("\n🚀 Running batched simulation...")

    start = time.perf_counter()
    for _ in range(batch):
        state = simulate_step(state, velocity, Δt)
    duration = time.perf_counter() - start

    print(f"🌀 Total simulation time ({batch} steps): {duration:.2f}s")
    assert np.isfinite(state).all()
    assert duration < 10.0  # threshold for batch performance


