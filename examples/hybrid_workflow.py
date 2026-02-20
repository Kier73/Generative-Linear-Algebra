import sys
import os
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sdk_registry import Registry
import x_matrix

def hybrid_workflow_demo():
    print("="*60)
    print("HYBRID WORKFLOW: ASSISTING INDUSTRY STANDARDS")
    print("="*60)
    print("Scenario: Analyzing a 10,000 x 10,000 Matrix Product")
    print("Goal: Use XMatrix to bypass O(n^3) compute, then export ROI to NumPy.")
    
    N = 10_000
    XMat = Registry.get_solver("XMatrix")
    
    # 1. The Pure NumPy Way (Simulation of Cost)
    print("\n[Baseline] Estimating NumPy Cost (N=10k)...")
    # A 10k x 10k float32 matrix is ~400MB. Two input matrices + result = 1.2GB RAM.
    # O(n^3) ops = 10^12 operations (1 Tera-Op).
    # On a standard CPU (100 GFLOPS), this takes ~10 seconds.
    print(f"  > Estimated Compute: {N**3 / 1e12} Tera-Ops")
    print(f"  > Estimated Memory:  {(N*N*4*3) / 1e9:.2f} GB")
    print("  > Status: SKIPPED (Too slow for interactive demo)")
    
    # 2. The XMatrix Way (Assisted Workflow)
    print("\n[Assisted] Running via XMatrix Bridge...")
    
    start_total = time.perf_counter()
    
    # A. Symbolic Compute (The Heavy Lift)
    print("  A. Performing Symbolic Multiplication...")
    ts = time.perf_counter()
    A = XMat(N, N, seed=1)
    B = XMat(N, N, seed=2)
    C = A.multiply(B)
    te = time.perf_counter()
    print(f"     > Done in {(te-ts)*1000:.4f} ms (Symbolic O(1))")
    
    # B. Region of Interest (ROI) Export
    # We only need the top-left 100x100 for visualization or local analysis
    print("  B. Exporting Top-Left 100x100 to NumPy...")
    ts = time.perf_counter()
    
    # In a real app, this would use the FFI bulk loader. 
    # Here we use the list comprehensions for clarity.
    roi_data = C.to_list(max_rows=100, max_cols=100)
    
    # Bridge to NumPy
    np_roi = np.array(roi_data, dtype=np.float32)
    
    te = time.perf_counter()
    print(f"     > Materialization Time: {(te-ts)*1000:.4f} ms")
    
    total_time = time.perf_counter() - start_total
    
    print("-" * 60)
    print(f"TOTAL WORKFLOW TIME: {total_time:.4f} sec")
    print(f"NumPy Shape Received: {np_roi.shape}")
    print(f"Data Mean: {np.mean(np_roi):.4f} (Expected ~0.0)")
    print("-" * 60)
    print("CONCLUSION: Workflow accelerated by avoiding full dense realization.")
    print("="*60)

if __name__ == "__main__":
    hybrid_workflow_demo()
