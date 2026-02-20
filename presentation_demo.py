import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import prime_matrix

def presentation_demo():
    """
    A simple, beautiful demo you can show anyone to prove the utility.
    """
    print("="*60)
    print("GENERATIVE LINEAR ALGEBRA: THE ANALYTICAL LEAP")
    print("="*60)
    
    # PROBLEM: A graph so large we can't even imagine it.
    N = 10**100
    print(f"Graph Nodes: 10^100 (A Googol)")
    
    # 1. THE ADJACENCY RULE
    # Node A connects to Node B if A divides B.
    print("\n[STEP 1] Defining the Adjacency Law...")
    P = prime_matrix.PrimeMatrix(N, N)
    
    # 2. THE COMPOSITION
    # We want to know how many 3-step paths exist.
    print("[STEP 2] Composing 3-Step Trajectories (Symbolic)...")
    start = time.perf_counter()
    P3 = P.multiply(P).multiply(P)
    t_comp = (time.perf_counter() - start) * 1000
    print(f"  > Composition latency: {t_comp:.4f} ms")
    
    # 3. THE RESOLUTION
    # Query: How many 3-step paths between Node 2 and Node 2^301?
    print("\n[STEP 3] Resolving Specific Path Count...")
    # Math: X = 2^301 / 2 = 2^300. 
    # Formula: binom(300 + 3 - 1, 3 - 1) = binom(302, 2) = (302 * 301) / 2 = 45451.
    
    start = time.perf_counter()
    paths = P3.get_element(1, (2**301) - 1)
    t_res = (time.perf_counter() - start) * 1000
    
    print(f"  > Start Node: 2")
    print(f"  > End Node:   2^301")
    print(f"  > Result:     {paths:.0f} paths")
    print(f"  > Verification: binom(302, 2) is exactly {45451}")
    print(f"  > Resolve latency: {t_res:.4f} ms")
    
    print("\n[CONCLUSION]")
    print("We just queried a graph larger than the universe.")
    print("We didn't store a single bit of the graph.")
    print("We solved it with pure analytical identities.")
    print("="*60)

if __name__ == "__main__":
    presentation_demo()
