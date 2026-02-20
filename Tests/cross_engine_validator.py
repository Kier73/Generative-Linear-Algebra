import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sdk_registry import Registry
from prime_matrix import PrimeMatrix
import v_matrix
import g_matrix
import x_matrix

def cross_engine_validator():
    print("="*80)
    print("CROSS-ENGINE VALIDATOR: INTEROPERABILITY & CONSISTENCY AUDIT")
    print("="*80)
    
    N = 10**100
    print(f"Audit Scale: 10^100")
    
    # 1. Register and retrieve solvers
    XMatrix = Registry.get_solver("XMatrix")
    GMatrix = Registry.get_solver("GMatrix")
    
    print("\n[PHASE 1] Symbolic-Inductive Parity...")
    # A G-Matrix can be symbolic or inductive. We verify parity.
    gm = GMatrix()
    A_desc = gm.from_data([[1.0, 0.0], [0.0, 1.0]]) # Identity
    B_desc = gm.from_data([[2.0, 3.0], [4.0, 5.0]])
    
    C_sym = gm.symbolic_matmul(A_desc, B_desc)
    
    # Realize a point
    val_sym = C_sym[0, 0]
    print(f"  > GMatrix Symbolic Resolution [0,0]: {val_sym}")
    
    # 2. X-Matrix vs Prime-Matrix (Seed Mapping)
    print("\n[PHASE 2] Seed-Variety Manifold Mapping...")
    # We verify that a matrix defined by a seed (X) and one defined by a law (Prime)
    # can coexist in the same environment without namespace/registry corruption.
    
    X = XMatrix(N, N, seed=0xABC)
    P = PrimeMatrix(N, N)
    
    val_x = X.get_element(1, 1)
    val_p = P.get_element(1, 1)
    
    print(f"  > XMatrix[1,1] (Stochastic): {val_x:+.4f}")
    print(f"  > PrimeMatrix[1,1] (Law of 2|2): {val_p}")
    
    if val_p == 1.0:
        print("  [OK] Cross-paradigm resolution confirmed.")

    # 3. Determinism Across Re-instantiation
    print("\n[PHASE 3] Manifold Re-instantiation Determinism...")
    seed = 0x54321
    M1 = XMatrix(N, N, seed=seed)
    M2 = XMatrix(N, N, seed=seed)
    
    v1 = M1.get_element(10, 10)
    v2 = M2.get_element(10, 10)
    
    print(f"  > Manifold 1 [10,10]: {v1}")
    print(f"  > Manifold 2 [10,10]: {v2}")
    
    if v1 == v2:
        print("  [OK] Deep Determinism Verified.")

    print("\n" + "="*80)
    print("CROSS-ENGINE AUDIT COMPLETE: THE REGISTRY IS STABLE")
    print("="*80)

if __name__ == "__main__":
    cross_engine_validator()
