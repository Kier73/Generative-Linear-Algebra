import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import x_matrix as sdk_x

N_SAMPLES = 1000

def run_test(force_rust=True):
    sdk_x.HAS_RUST = force_rust
    print(f"\nBackend: {'RUST (ACCELERATED)' if force_rust else 'PYTHON (FALLBACK)'}")
    print(f"Resolving {N_SAMPLES} elements...")
    
    mat = sdk_x.XMatrix(1024, 1024, seed=42)
    # Warmup
    _ = mat.get_element(0, 0)
    
    start = time.perf_counter()
    for i in range(N_SAMPLES):
        r = i % 1024
        c = (i * 7) % 1024
        _ = mat.get_element(r, c)
    total_time = (time.perf_counter() - start) * 1000
    avg_lat = total_time / N_SAMPLES
    
    print(f"Total Time:   {total_time:.4f} ms")
    print(f"Avg Latency:  {avg_lat:.6f} ms per element")
    print(f"Throughput:   {N_SAMPLES / (total_time/1000):.2f} Ops/Sec")
    return avg_lat

# Run both
rust_lat = run_test(force_rust=True)
py_lat = run_test(force_rust=False)

print("\n" + "="*60)
print(f"SPEEDUP FACTOR: {py_lat / rust_lat:.2f}x")
print("="*60)

# HDC Core Bench (Atomic Shift)
sdk_x.HAS_RUST = True
print("\nHDC Core Operations (Atomic Shift - Rust):")
a = sdk_x.HdcManifold(seed=1)
start = time.perf_counter()
for _ in range(N_SAMPLES):
    a = a.shift(1)
total_time_shift = (time.perf_counter() - start) * 1000
print(f"Shift Latency: {total_time_shift / N_SAMPLES:.6f} ms")

if __name__ == "__main__":
    # Benchmark is already triggered by the top-level calls in this script
    pass
