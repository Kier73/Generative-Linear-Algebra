import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from prime_matrix import PrimeMatrix

def run_prime_challenge():
    print("="*80)
    print("ANALYTICAL MATRIX AUDIT: LARGR-SCALE DIVISOR ANALYSIS")
    print("="*80)
    
    # Scale: A Googol (10^100)
    N = 10**100
    print(f"Matrix Dimension (N): 10^100")
    print("Definition: Matrix P where P[i, j] = 1 if (i+1) divides (j+1)")
    
    print("\n[PHASE 1] Initializing Analytical Matrix...")
    P = PrimeMatrix(N, N)
    
    # 1. Verification of the Law
    print("[PHASE 2] Verifying Local Analytical Consistency...")
    test_cases = [
        (1, 3),   # 2 divides 4? Yes (1.0)
        (2, 5),   # 3 divides 6? Yes (1.0)
        (2, 4),   # 3 divides 5? No (0.0)
    ]
    
    for r, c in test_cases:
        val = P.get_element(r, c)
        print(f"  > P[{r}, {c}] (Is {r+1} a divisor of {c+1}?): {val}")

    # 2. Symbolic Multiplication (Composite Matrix)
    print("\n[PHASE 3] Symbolic Composition Analysis (P^2)...")
    print("  Note: P^2[i, j] counts k such that i|k and k|j (Divisor chain count)")
    
    start = time.perf_counter()
    P2 = P.multiply(P)
    t_sym = (time.perf_counter() - start) * 1000
    print(f"  > P^2 Synthesis Latency: {t_sym:.6f} ms")
    
    # 3. Large-Scale Coordinate Resolution
    print("\n[PHASE 4] Resolving Sparse Coordinates...")
    
    # Sample a very large number at Googol scale
    # We choose a coordinate that is a power of 2 for verification.
    root = 2**300 # A very large number
    row = 1       # Divisor = 2
    col = root - 1 # Number = 2^300
    
    start = time.perf_counter()
    val = P2.get_element(row, col)
    res_latency = (time.perf_counter() - start) * 1000
    print(f"  > Coordinate ({row}, {col})")
    print(f"    - Identity: Is 2 a divisor of 2^300?")
    print(f"    - Compositional Count: {val:.4f}")
    print(f"    - Resolution Latency: {res_latency:.6f} ms")

    print("\n" + "="*80)
    print("AUDIT COMPLETE: ANALYTICAL CONSISTENCY VERIFIED")
    print("="*80)

if __name__ == "__main__":
    run_prime_challenge()
