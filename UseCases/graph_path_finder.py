import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from prime_matrix import PrimeMatrix

def run_graph_demo():
    print("="*80)
    print("USE CASE: LARGE-SCALE PATH COUNTING via SYMBOLIC MULTIPLICATION")
    print("="*80)
    
    # 1. Define the Scale
    # We simulate a graph with 1 million nodes.
    # The adjacency matrix P represents edges (i -> j) where i+1 divides j+1.
    N = 1_000_000
    print(f"Graph Nodes: {N}")
    
    print("\n[STEP 1] Initializing Adjacency Mapping...")
    P = PrimeMatrix(N, N)
    
    # 2. Path Counting via symbolic MatMul
    # P^m[i, j] counts paths of length m from node i to node j.
    m = 5
    print(f"[STEP 2] Computing Paths of Length {m} (Recursive Composition)...")
    
    start = time.perf_counter()
    P_m = PrimeMatrix(N, N, depth=m)
    t_sym = (time.perf_counter() - start) * 1000
    print(f"  > Symbolic Path Synthesis: {t_sym:.6f} ms")
    
    # 3. Resolve Path Counts for Specific Nodes
    print("\n[STEP 3] Resolving Path Count: Node 1 to Node 1023...")
    # Path count is the number of chains k_0, k_1, ..., k_m such that 2|k_1|...|1024
    start = time.perf_counter()
    paths = P_m.get_element(1, 1023)
    t_res = (time.perf_counter() - start) * 1000
    
    print(f"  > Result: {paths:.0f} unique paths.")
    print(f"  > Resolution Latency: {t_res:.6f} ms")
    
    # 4. Large-Scale Demonstration
    print(f"\n[STEP 4] Scaling to Large-Scale Graph (10^15 nodes)...")
    Huge_N = 10**15
    P_huge = PrimeMatrix(Huge_N, Huge_N, depth=m)
    
    start = time.perf_counter()
    huge_paths = P_huge.get_element(1, 10**12 - 1)
    t_huge = (time.perf_counter() - start) * 1000
    print(f"  > Paths from 2 to 10^12 in {m}-steps: {huge_paths:.0f}")
    print(f"  > Resolution Latency at Quadrillion scale: {t_huge:.6f} ms")

    print("\n" + "="*80)
    print("DEMO COMPLETE: GRAPH QUERY BOTTLE-NECK NEUTRALIZED")
    print("="*80)

if __name__ == "__main__":
    run_graph_demo()
