import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sdk_registry import Registry
import x_matrix 

def test_adversarial_edge_cases():
    print("="*60)
    print("XMATRIX ADVERSARIAL & SCRUTINY TESTS")
    print("="*60)
    
    XMat = Registry.get_solver("XMatrix")
    
    # 1. The "Self-Annihilation" Test (Involutory Matrix)
    # In HDC binding (XOR), A * A should equal Identity (Zero Vector or similar depending on implementation)
    # However, XMatrix uses Shift(B), so A * Shift(A) shouldn't annihilate immediately.
    print("[Adversarial 1] Self-Binding Stability (A @ A)")
    A = XMat(100, 100, seed=42)
    B = A.multiply(A)
    
    sim = A.manifold.similarity(B.manifold)
    print(f"  > Similarity(A, A@A): {sim:.4f}")
    if abs(sim) < 0.1:
        print("  [PASS] A @ A produced distinct manifold (No collapse to Identity).")
    else:
        print("  [FAIL] Manifold collapsed or stagnated!")

    # 2. Extreme Aspect Ratios (The "Needle" Test)
    # 1 x 10^18 vector vs 10^18 x 1 vector
    print("\n[Adversarial 2] Extreme Aspect Ratios (1 x 10^18)")
    rows = 1
    cols = 10**18
    
    try:
        NeedleRow = XMat(rows, cols, seed=101)
        NeedleCol = XMat(cols, rows, seed=202) # Valid dot product dimensions
        
        start = time.perf_counter()
        ScalarLike = NeedleRow.multiply(NeedleCol)
        lat = (time.perf_counter() - start) * 1000
        
        print(f"  > 1x{cols} @ {cols}x1")
        print(f"  > Latency: {lat:.6f} ms")
        print(f"  > Result Shape: {ScalarLike.rows}x{ScalarLike.cols}")
        
        if ScalarLike.rows == 1 and ScalarLike.cols == 1:
            print("  [PASS] Topology preserved at hyper-scale limits.")
        else:
            print(f"  [FAIL] Incorrect shape: {ScalarLike.rows}x{ScalarLike.cols}")
            
    except Exception as e:
        print(f"  [CRITICAL FAIL] Crash on extreme aspect ratio: {e}")

    # 3. Floating Point Singularity (NaN/Inf injection attempt via Seeds)
    # Seeds are integers, checking if massive seeds break the generator
    print("\n[Adversarial 3] Seed Overflow Scrutiny")
    try:
        MassiveSeed = 2**64 + 1
        X_Huge = XMat(100, 100, seed=MassiveSeed)
        val = X_Huge.get_element(0, 0)
        print(f"  > Input Seed: 2^64 + 1")
        print(f"  > Resolved Val: {val}")
        print("  [PASS] 64-bit Overflow handled gracefully (Python int arbitrary precision).")
    except Exception as e:
        print(f"  [FAIL] Generator crashed on large seed: {e}")

    # 4. Deep Recursion / Stack Overflow Check
    print("\n[Adversarial 4] Deep Recursion (Chain of 1000)")
    # We want to ensure the descriptor doesn't blow the stack if we chain it
    current = XMat(10, 10, seed=1)
    try:
        for _ in range(1000):
            current = current.multiply(XMat(10, 10, seed=2))
        print("  [PASS] Chained 1000 operations without StackOverflow.")
    except RecursionError:
        print("  [FAIL] RecursionError detected!")
    except Exception as e:
        print(f"  [FAIL] Error: {e}")

    print("="*60)

if __name__ == "__main__":
    test_adversarial_edge_cases()
