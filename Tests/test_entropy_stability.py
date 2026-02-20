import sys
import os
import random
import statistics

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sdk_registry import Registry

# Ensure XMatrix is loaded
import x_matrix 

def test_entropy_stability():
    print("="*60)
    print("XMATRIX ENTROPY STABILITY AUDIT")
    print("="*60)
    
    # 1. Setup
    XMat = Registry.get_solver("XMatrix")
    N = 1000
    manifolds = []
    
    print(f"Generating {N} random 1024-bit manifolds...")
    for i in range(N):
        # We access the raw HdcManifold for direct auditing
        # Each matrix has a unique seed
        m = XMat(10, 10, seed=i).manifold
        manifolds.append(m)
        
    # 2. Collision Test
    print("\n[Audit 1] Signature Collision Check")
    sigs = set()
    for m in manifolds:
        # Signatures are 64-bit hashes of the 1024-bit state
        sig = XMat(10, 10).oracle._get_sig(m)
        sigs.add(sig)
        
    collisions = N - len(sigs)
    print(f"  > Unique Signatures: {len(sigs)} / {N}")
    print(f"  > Collisions:        {collisions}")
    
    if collisions == 0:
        print("  [PASS] Zero collisions detected in 1000 samples.")
    else:
        print(f"  [FAIL] {collisions} collisions detected!")
        
    # 3. Orthogonality / Similarity Distribution
    print("\n[Audit 2] Orthogonality Distribution (Hamming Distance)")
    print("  Checking pairwise similarity of random subset...")
    
    sims = []
    sample_size = 100
    for i in range(sample_size):
        # Compare m[i] with m[i+1] (random vs random)
        s = manifolds[i].similarity(manifolds[i+1])
        sims.append(s)
        
    avg_sim = statistics.mean(sims)
    std_dev = statistics.stdev(sims)
    
    print(f"  > Avg Similarity: {avg_sim:.4f} (Target: ~0.0000)")
    print(f"  > Std Deviation:  {std_dev:.4f}")
    
    # In High-Dimensional Space, random vectors are nearly orthogonal
    # Similarity should be close to 0 (0.5 Hamming distance normalized to 0)
    if abs(avg_sim) < 0.05:
        print("  [PASS] Manifolds are strictly orthogonal (High Entropy).")
    else:
        print("  [FAIL] Manifolds show bias/saturation!")
        
    print("="*60)

if __name__ == "__main__":
    test_entropy_stability()
