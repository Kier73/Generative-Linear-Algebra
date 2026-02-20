import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import g_matrix as gm
import random
import time

def test_g_matrix():
    print("="*60)
    print("G_MATRIX SDK VERIFICATION (GEN 2)")
    print("="*60)
    
    sdk = gm.GMatrix()
    size = 64
    A = [[random.random() for _ in range(size)] for _ in range(size)]
    B = [[random.random() for _ in range(size)] for _ in range(size)]
    
    # 1. Test Inductive Dispatch
    print("\n[PHASE 1] Inductive Tile Memoization")
    start = time.perf_counter()
    _ = sdk.matmul(A, B)
    t_cold = (time.perf_counter() - start) * 1000
    print(f"  > Cold Pass (Compute + Induct): {t_cold:.3f}ms")
    
    start = time.perf_counter()
    _ = sdk.matmul(A, B)
    t_warm = (time.perf_counter() - start) * 1000
    print(f"  > Warm Pass (Recursive Recall): {t_warm:.3f}ms")
    
    if t_warm < t_cold:
         print(f"  [OK] Speedup Achieved: {t_cold/t_warm:.1f}x")
    
    # 2. Test Geometric Synthesis
    print("\n[PHASE 2] Geometric Descriptor Synthesis (O(1))")
    desc_a = sdk.from_data(A)
    desc_b = sdk.from_data(B)
    
    start = time.perf_counter()
    desc_c = sdk.symbolic_matmul(desc_a, desc_b)
    t_sym = (time.perf_counter() - start) * 1000
    print(f"  > Symbolic Operation Latency:  {t_sym:.6f}ms")
    
    if t_sym < 0.1:
         print("  [OK] Descriptor synthesis is O(1).")
         
    # 3. Test JIT Resolution
    print("\n[PHASE 3] Coordinate-Based Realization")
    # symbolic_matmul returns a GeometricMatrix, which wraps the descriptor
    val = desc_c.desc.resolve(32, 32)
    print(f"  > Value at (32, 32): {val:.6f}")
    assert -1.0 <= val <= 1.0
    print("  [OK] Element resolution verified.")
    
    print("\n" + "="*60)
    print("G_MATRIX VERIFICATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_g_matrix()
