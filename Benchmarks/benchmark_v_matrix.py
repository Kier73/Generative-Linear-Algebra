import sys
import os
import time
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import v_matrix as vm

def generate_matrix(rows, cols):
    return [[random.random() for _ in range(cols)] for _ in range(rows)]

def benchmark():
    print("="*60)
    print("V_MATRIX SDK PERFORMANCE BENCHMARK")
    print("="*60)
    
    sizes = [8, 16, 32, 64]
    
    for n in sizes:
        print(f"\nMatrix Size: {n}x{n}")
        A = generate_matrix(n, n)
        B = generate_matrix(n, n)
        
        # Spectral Benchmark
        v_spec = vm.VMatrix(mode="spectral")
        start = time.perf_counter()
        _ = v_spec.matmul(A, B)
        end = time.perf_counter()
        spectral_time = (end - start) * 1000
        print(f"  > Spectral (omega=2): {spectral_time:.4f} ms")
        
        # RNS Benchmark
        v_rns = vm.VMatrix(mode="rns")
        start = time.perf_counter()
        _ = v_rns.matmul(A, B)
        end = time.perf_counter()
        rns_time = (end - start) * 1000
        print(f"  > RNS (Exact-ish):  {rns_time:.4f} ms")
        
        # Traditional (Pure Python reference)
        start = time.perf_counter()
        C_trad = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    C_trad[i][j] += A[i][k] * B[k][j]
        end = time.perf_counter()
        trad_time = (end - start) * 1000
        print(f"  > Trad (O(n^3)):    {trad_time:.4f} ms")
        
        # Speedup calculation
        if spectral_time > 0:
            speedup = trad_time / spectral_time
            print(f"  [RESULT] Spectral Speedup: {speedup:.2f}x")

    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)

if __name__ == "__main__":
    benchmark()
