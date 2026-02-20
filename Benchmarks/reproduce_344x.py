import sys
import os
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import g_matrix as gm

def benchmark_speedup(size=512):
    sdk = gm.GMatrix()
    
    # Recurring 32x32 tiles for maximum Inductive cache hit rate
    tile = np.random.rand(32, 32).astype(np.float32)
    A = np.tile(tile, (size // 32, size // 32))
    B = np.tile(tile, (size // 32, size // 32))

    print(f"Benchmarking {size}x{size} PATTERNED Matrix...")

    # 1. COLD PASS
    start = time.perf_counter()
    _ = sdk.matmul(A, B)
    t_cold = (time.perf_counter() - start) * 1000
    print(f"Cold Pass: {t_cold:.3f} ms")

    # 2. WARM PASS
    start = time.perf_counter()
    _ = sdk.matmul(A, B)
    t_warm = (time.perf_counter() - start) * 1000
    print(f"Warm Pass: {t_warm:.3f} ms")

    speedup = t_cold / t_warm
    print(f"Ratio: {speedup:.1f}x speedup")

if __name__ == "__main__":
    benchmark_speedup(512)
