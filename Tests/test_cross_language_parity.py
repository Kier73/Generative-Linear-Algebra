import sys
import os
import random
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sdk_registry import Registry
import x_matrix 

def test_cross_language_parity():
    print("="*60)
    print("CROSS-LANGUAGE PARITY AUDIT (PYTHON VS RUST)")
    print("="*60)
    
    # 1. Setup
    XMat = Registry.get_solver("XMatrix")
    rows, cols = 64, 64
    seed = 12345
    
    print(f"Matrix Size: {rows}x{cols}")
    print(f"Seed:        {seed}")
    
    # Instantiate
    xm = XMat(rows, cols, seed=seed)
    
    # 2. Python vs Rust Parity Check
    
    # A. Execute with Rust (Default if available)
    print("\n[Audit] Executing Rust Backend...")
    if not x_matrix.HAS_RUST:
        print("  [WARN] Rust backend not loaded! Skipping Parity Check (Pure Python only).")
        return

    # Operation: Bind(A, B) -> Shift(1)
    # We use the raw HdcManifold methods to avoid extra wrapping logic
    m1 = xm.manifold
    m2 = XMat(rows, cols, seed=67890).manifold
    
    t0 = time.perf_counter()
    res_rust = m1.bind(m2).shift(1)
    t_rust = (time.perf_counter() - t0) * 1000
    print(f"  > Rust Time: {t_rust:.4f} ms")
    
    # B. Execute with Pure Python (Force Fallback)
    print("\n[Audit] Executing Pure Python Backend...")
    # Monkeymatch the flag
    original_flag = x_matrix.HAS_RUST
    x_matrix.HAS_RUST = False 
    
    try:
        t0 = time.perf_counter()
        res_py = m1.bind(m2).shift(1)
        t_py = (time.perf_counter() - t0) * 1000
        print(f"  > Python Time: {t_py:.4f} ms")
        
        # C. Compare
        print("\n[Audit] Comparison...")
        if res_rust.data == res_py.data:
            print("  [PASS] Bit-Exact Match Verified!")
            print(f"  > Speedup: {t_py / t_rust:.2f}x")
        else:
            print("  [FAIL] State Mismatch!")
            print(f"  > Rust Sample: {res_rust.data[:4]}")
            print(f"  > Py   Sample: {res_py.data[:4]}")
            
    finally:
        # Restore state
        x_matrix.HAS_RUST = original_flag

    print("="*60)

if __name__ == "__main__":
    test_cross_language_parity()

    print("="*60)

if __name__ == "__main__":
    test_cross_language_parity()
