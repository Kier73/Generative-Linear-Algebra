import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import v_matrix as vm
import math

def test_v_matrix():
    print("Starting V_Matrix SDK Verification...")
    
    # Setup matrices
    A = [[1.0, 2.0], [3.0, 4.0]]
    B = [[0.5, 0.1], [0.2, 0.6]]
    
    # 1. Test Projection Engine
    v_spec = vm.VMatrix(mode="spectral")
    res_spec = v_spec.matmul(A, B)
    print(f"[OK] Projection Output (CA-based): {res_spec}")
    assert len(res_spec) == 2 and len(res_spec[0]) == 2
    
    # 2. Test RNS Engine
    v_rns = vm.VMatrix(mode="rns")
    res_rns = v_rns.matmul(A, B)
    print(f"[OK] RNS Output (Modulo Arithmetic): {res_rns}")
    assert len(res_rns) == 2 and len(res_rns[0]) == 2
    
    # 3. Test On-the-fly Parameter Engine
    v_snap = vm.VMatrix()
    res_snap = v_snap.snap_project([1.0, 0.5], seed=123, out_dim=3)
    print(f"[OK] On-the-fly Projection: {res_snap}")
    assert len(res_snap) == 3
    
    # 4. Test v_mask Determinism
    m1 = vm.v_mask(42)
    m2 = vm.v_mask(42)
    assert m1 == m2
    print(f"[OK] v_mask Determinism: {m1}")

    print("\nVERIFICATION COMPLETE: ALL SYSTEMS NOMINAL")

if __name__ == "__main__":
    test_v_matrix()
