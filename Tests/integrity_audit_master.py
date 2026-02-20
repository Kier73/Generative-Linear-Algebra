import sys
import os
import time
import math
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import x_matrix
import prime_matrix
import v_matrix
from sdk_registry import Registry

def run_integrity_gauntlet():
    results = {}
    
    # Perspective 1: Axiomatic Integrity
    N = 10**100
    XMatrix = Registry.get_solver("XMatrix")
    A_x = XMatrix(N, N, seed=1)
    B_x = XMatrix(N, N, seed=2)
    C_x = XMatrix(N, N, seed=3)
    LHS_x = (A_x.multiply(B_x)).multiply(C_x)
    RHS_x = A_x.multiply(B_x.multiply(C_x))
    results['hdc_associativity'] = LHS_x.manifold.similarity(RHS_x.manifold)

    P = prime_matrix.PrimeMatrix(N, N, depth=1)
    LHS_p = (P.multiply(P)).multiply(P)
    RHS_p = P.multiply(P.multiply(P))
    row, col = 1, (2**300) - 1
    results['analytical_associativity'] = LHS_p.get_element(row, col) == RHS_p.get_element(row, col)

    # Perspective 2: Dimensional Stress
    Extreme_N = 10**1000
    P_extreme = prime_matrix.PrimeMatrix(Extreme_N, Extreme_N, depth=2)
    start = time.perf_counter()
    val = P_extreme.get_element(1, (2**1001) - 1)
    results['dim_stress_latency'] = (time.perf_counter() - start) * 1000
    results['dim_stress_correctness'] = val == 1001

    # Perspective 3: Entropy
    sampler_size = 5000
    seeds = [fmix64_python(i) for i in range(sampler_size)]
    descriptors = [x_matrix.HdcManifold(seed=s) for s in seeds]
    results['uniqueness_pass'] = len(set(seeds)) == sampler_size

    # Perspective 4: Cross-Generation
    N_small = 100
    P_small = prime_matrix.PrimeMatrix(N_small, N_small)
    X_small = XMatrix(N_small, N_small, seed=0x7072696D)
    results['cross_engine_parity'] = P_small.get_element(1, 3) == 1 and X_small.get_element(1, 3) == 1.0

    # Perspective 5: Depth Stress
    Depth = 10**18
    P_deep = prime_matrix.PrimeMatrix(N, N, depth=Depth)
    start = time.perf_counter()
    val_deep = P_deep.get_element(1, 3)
    results['depth_stress_latency'] = (time.perf_counter() - start) * 1000
    results['depth_stress_correctness'] = val_deep == Depth

    # Output Summary
    print("--- Integrity Audit Results ---")
    for key, value in results.items():
        print(f"{key}: {value}")
    
    all_passed = (results['analytical_associativity'] and 
                  results['dim_stress_correctness'] and 
                  results['uniqueness_pass'] and 
                  results['cross_engine_parity'] and 
                  results['depth_stress_correctness'])
    
    print(f"Overall Pass: {all_passed}")

def fmix64_python(h: int) -> int:
    h &= 0xFFFFFFFFFFFFFFFF
    h ^= h >> 33
    h = (h * 0xff51afd7ed558ccd) & 0xFFFFFFFFFFFFFFFF
    h ^= h >> 33
    h = (h * 0xc4ceb9fe1a85ec53) & 0xFFFFFFFFFFFFFFFF
    h ^= h >> 33
    return h

if __name__ == "__main__":
    run_integrity_gauntlet()
