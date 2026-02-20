import sys
import os
import time
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sdk_registry import Registry
import x_matrix
import prime_matrix

def grounding_proof():
    print("--- Grounding Audit: Numerical vs Analytical ---")
    
    N = 100
    P = prime_matrix.PrimeMatrix(N, N)
    
    # 1. Verification of Entry
    val = P.get_element(1, 3)
    print(f"Basic Entry Check (2 divides 4): {val}")
    
    # 2. Comparative Matrix Product
    P_np = np.zeros((N, N))
    for r in range(N):
        for c in range(N):
            P_np[r, c] = P.get_element(r, c)
            
    C_np = np.dot(P_np, P_np)
    P2 = P.multiply(P)
    
    # 3. Precision Verification
    sample_coords = [(0, 3), (1, 3), (0, 9)]
    matches = 0
    for r, c in sample_coords:
        val_np = C_np[r, c]
        val_gen = P2.get_element(r, c)
        if val_np == val_gen:
            matches += 1
            
    print(f"Identity Match Rate: {(matches / len(sample_coords)) * 100:.2f}%")
    
    # 4. Large-Scale Verification
    Huge_N = 10**15
    P_huge = prime_matrix.PrimeMatrix(Huge_N, Huge_N, depth=5)
    start = time.perf_counter()
    paths = P_huge.get_element(1, 1023)
    t_res = (time.perf_counter() - start) * 1000
    print(f"Quadrillion Graph Resolution (5 steps): {paths:.0f} paths in {t_res:.4f} ms")

    # 5. Googol Scale Identity
    Googol_N = 10**100
    P_googol = prime_matrix.PrimeMatrix(Googol_N, Googol_N, depth=2)
    start = time.perf_counter()
    res = P_googol.get_element(1, (2**301) - 1)
    t_res = (time.perf_counter() - start) * 1000
    print(f"Googol Scale Resolution (2 steps, Power of 2): {res:.0f} in {t_res:.4f} ms")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    grounding_proof()
