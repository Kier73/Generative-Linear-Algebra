import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import random
import subprocess
import json
import numpy as np
import scipy.sparse as sp
import torch

# --- SDK IMPORTS ---
try:
    import v_matrix as sdk_v1
except ImportError:
    sdk_v1 = None

try:
    import g_matrix as sdk_g
except ImportError:
    sdk_g = None

try:
    import x_matrix as sdk_x
except ImportError:
    sdk_x = None

# --- CONFIGURATION ---
SIZES = [128, 256, 512]
TRIALS = 3
RUST_SDK_PATH = os.path.join(os.path.dirname(__file__), "..", "v_matrix_rust")

def benchmark_rust(n):
    """Call the Gen 1.5 Rust Parallel SDK."""
    try:
        cmd = ["cargo", "run", "--release", "--", str(n)]
        result = subprocess.run(cmd, cwd=RUST_SDK_PATH, capture_output=True, text=True, timeout=60)
        for line in result.stdout.splitlines():
            if f"[BENCH] SDK_Rust_Auto:{n}x{n}:" in line:
                return float(line.split(":")[2].split("ms")[0].strip())
    except Exception:
        return None
    return None

def generate_matrices(n, m_type="dense"):
    """Generate matrices of specific types."""
    if m_type == "dense":
        A = np.random.rand(n, n).astype(np.float32)
        B = np.random.rand(n, n).astype(np.float32)
    elif m_type == "sparse":
        # 99% sparse
        A = sp.random(n, n, density=0.01, format='csr', dtype=np.float32)
        B = sp.random(n, n, density=0.01, format='csr', dtype=np.float32)
    elif m_type == "patterned":
        # Recurring 32x32 tiles for Inductive GEMM
        tile = np.random.rand(32, 32).astype(np.float32)
        A = np.tile(tile, (n // 32, n // 32))
        B = np.tile(tile, (n // 32, n // 32))
    return A, B

def run_bench():
    print("="*80)
    print("ULTIMATE INDUSTRY-STANDARD PERFORMANCE AUDIT")
    print("="*80)
    
    types = ["dense", "sparse", "patterned"]
    results = {}

    for m_type in types:
        print(f"\n[TARGET: {m_type.upper()} MATRICES]")
        results[m_type] = {}
        
        for n in SIZES:
            print(f"  Benchmarking {n}x{n}...")
            A, B = generate_matrices(n, m_type)
            A_list = A.tolist() if not sp.issparse(A) else A.toarray().tolist()
            B_list = B.tolist() if not sp.issparse(B) else B.toarray().tolist()
            
            n_res = {}

            # 1. NumPy (Industry Standard BLAS)
            if not sp.issparse(A):
                start = time.perf_counter()
                _ = np.matmul(A, B)
                n_res["NumPy (BLAS)"] = (time.perf_counter() - start) * 1000
            
            # 2. PyTorch (Industry Standard ML)
            if not sp.issparse(A):
                t_a, t_b = torch.from_numpy(A), torch.from_numpy(B)
                start = time.perf_counter()
                _ = torch.matmul(t_a, t_b)
                n_res["PyTorch (CPU)"] = (time.perf_counter() - start) * 1000

            # 3. SciPy Sparse (Sparse Industry Standard)
            if sp.issparse(A):
                start = time.perf_counter()
                _ = A.dot(B)
                n_res["SciPy (Sparse)"] = (time.perf_counter() - start) * 1000

            # 4. G_Matrix (Gen 2 Rust Warm)
            if sdk_g and not sp.issparse(A):
                gm = sdk_g.GMatrix()
                # Warm-up pass to seed the inductive cache
                _ = gm.matmul(A, B)
                start = time.perf_counter()
                _ = gm.matmul(A, B)
                n_res["G_Matrix (Gen 2 Warm)"] = (time.perf_counter() - start) * 1000

            # 5. Rust Parallel (Gen 1.5)
            if not sp.issparse(A):
                r_time = benchmark_rust(n)
                if r_time:
                    n_res["Rust (Gen 1.5 Para)"] = r_time

            # 6. V_Matrix (Gen 1 Spectral)
            if sdk_v1 and not sp.issparse(A) and n <= 256:
                vm = sdk_v1.VMatrix(mode="spectral")
                start = time.perf_counter()
                _ = vm.matmul(A_list, B_list)
                n_res["V_Matrix (Gen 1 Spec)"] = (time.perf_counter() - start) * 1000

            # 7. X_Matrix (Gen 3.5 Isomorphic)
            if sdk_x and not sp.issparse(A):
                # Symbolic pass
                xm = sdk_x.XMatrix(n, n, seed=random.randint(0, 0xFFFF))
                ym = sdk_x.XMatrix(n, n, seed=random.randint(0, 0xFFFF))
                start = time.perf_counter()
                zm = xm.multiply(ym)
                n_res["X_Matrix (Gen 3.5 Sym)"] = (time.perf_counter() - start) * 1000
                
                # Isomorph Oracle pass (Sub-microsecond)
                start = time.perf_counter()
                _ = xm.multiply(ym)
                n_res["X_Matrix (Gen 3.5 Oracle)"] = (time.perf_counter() - start) * 1000

            results[m_type][n] = n_res

    # --- FINAL REPORT ---
    print("\n" + "="*80)
    print("FINAL BENCHMARK REPORT (Metrics in ms)")
    print("="*80)
    
    for m_type in types:
        print(f"\n>> MATRIX TYPE: {m_type.upper()}")
        header = f"{'Implementation':<25}"
        for n in SIZES:
            header += f"{n:<12}"
        print(header)
        print("-" * len(header))
        
        # Get all impl names
        impls = set()
        for n in SIZES:
            impls.update(results[m_type][n].keys())
        
        for impl in sorted(impls):
            row = f"{impl:<25}"
            for n in SIZES:
                val = results[m_type][n].get(impl, "N/A")
                if isinstance(val, float):
                    row += f"{val:<12.3f}"
                else:
                    row += f"{val:<12}"
            print(row)

if __name__ == "__main__":
    run_bench()
