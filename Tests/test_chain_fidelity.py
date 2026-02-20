import sys
import os
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sdk_registry import Registry
import g_matrix # Should import both G and X if needed, but test focuses on X
import x_matrix 

def test_chain_fidelity():
    N_OPS = 100
    print("="*60)
    print(f"XMATRIX LONG-CHAIN FIDELITY AUDIT ({N_OPS} Operations)")
    print("="*60)
    
    # 1. Setup
    XMat = Registry.get_solver("XMatrix")
    
    # Chain: M = A1 * A2 * ... * An
    # To check fidelity, we track a single bit/element path symbolically?
    # Actually, simpler: Is the result distinct from random noise?
    
    chain = XMat(10, 10, seed=0)
    print(f"Initial: {chain.manifold.label} | Sig: {chain.oracle._get_sig(chain.manifold):x}")
    
    # 2. Perform Chain
    for i in range(1, N_OPS):
        next_mat = XMat(10, 10, seed=i)
        chain = chain.multiply(next_mat)
        
    print(f"Final:   {chain.manifold.label[:50]}... (truncated)")
    print(f"Sig:     {chain.oracle._get_sig(chain.manifold):x}")
    
    # 3. Resolve Element
    # If the manifold collapsed, all elements might be +1 or -1
    print("\n[Audit] Element Distribution Check (Center Sample)")
    
    ones = 0
    zeros = 0
    
    # Sample 100 random coordinates from the final product
    for _ in range(100):
        r = random.randint(0, 9999) # Using large coords to stress hash
        c = random.randint(0, 9999)
        val = chain.get_element(r, c)
        if val > 0: ones += 1
        else: zeros += 1
        
    print(f"  > +1s: {ones}")
    print(f"  > -1s: {zeros}")
    
    ratio = ones / (ones + zeros)
    print(f"  > Balance Ratio: {ratio:.2f}")
    
    if 0.4 <= ratio <= 0.6:
        print("  [PASS] Signal remains balanced after 100 ops.")
    else:
        print("  [FAIL] Signal collapsed to constant state!")
        
    print("="*60)

if __name__ == "__main__":
    test_chain_fidelity()
