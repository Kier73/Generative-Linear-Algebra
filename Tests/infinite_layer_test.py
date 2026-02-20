import sys
import os
import time
import psutil
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sdk_registry import Registry
from x_matrix import XMatrix

class InfiniteLinear:
    """
    A Linear Layer with procedurally generated weights (Fixed State).
    Capable of scaling to dimensions exceeding available memory.
    
    Mathematically: W ~ XMatrix(out, in)
    Operation: y = W @ x + b (bias not implemented for this core demo)
    """
    def __init__(self, in_features: int, out_features: int, seed: int = 0):
        self.in_features = in_features
        self.out_features = out_features
        self.seed = seed
        # The weight matrix is virtual. No allocation happens.
        self.weights = XMatrix(out_features, in_features, seed=seed)
        
    def forward_sparse(self, indices: list[int], values: list[float]) -> list[float]:
        """
        Computes y = W @ x where x is a sparse vector.
        
        Args:
            indices: List of indices where x is non-zero.
            values: List of values at those indices.
            
        Returns:
            Dense output vector y of size out_features.
        """
        # For a massive output dimension, returning a full dense vector might be efficient 
        # only if out_features is manageable, OR if we only want specific outputs.
        # IF out_features is 1,000,000, we compute a 1M vector.
        # This is strictly O(active_inputs * out_features).
        
        # Optimization:
        # Instead of resolving W[row, col] one by one, XMatrix generates 
        # rows/cols procedurally.
        # W @ x = sum(x[j] * Column_j) for active j.
        # We need to construct the output column vector.
        
        # For this implementation, we simulate the standard 'Dense Output' case.
        output = [0.0] * self.out_features
        
        # To make this fast for specific queries (like getting 1000 features),
        # we iterate over the inputs.
        for idx, val in zip(indices, values):
            # We add val * Column_j to the output.
            # XMatrix can interpret rows/cols symmetrically.
            # Let's verify get_element access pattern in XMatrix to ensure efficiency.
            # XMatrix.get_element(r, c) resolves one bit.
            
            # CRITICAL OPTIMIZATION: 
            # If we iterate 1M output rows for every input, it's slow.
            # Python loop overhead is significant.
            # But the MEMORY capability is the point here.
            
            for r in range(self.out_features):
                # W[r, idx]
                w_val = self.weights.get_element(r, idx)
                output[r] += w_val * val
                
        return output
    
    def forward_sparse_sliced(self, indices: list[int], values: list[float], out_indices: list[int]) -> list[float]:
        """
        Computes specific output neurons only.
        y_subset = W[out_indices, :] @ x
        """
        output = []
        for r in out_indices:
            acc = 0.0
            for idx, val in zip(indices, values):
                acc += self.weights.get_element(r, idx) * val
            output.append(acc)
        return output

def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def run_functional_test():
    print("--- InfiniteLinear Functional Test ---")
    
    # 1. Setup: Trillion Parameter Layer (1M inputs -> 1M outputs)
    # This represents a sparse auto-encoder or massive random projection.
    print(f"Initializing InfiniteLinear(1,000,000, 1,000,000)")
    start_mem = get_memory_mb()
    
    layer = InfiniteLinear(1_000_000, 1_000_000, seed=42)
    
    end_mem = get_memory_mb()
    print(f"Memory Overhead: {end_mem - start_mem:.4f} MB")
    
    # 2. Input: Sparse Feature Vector
    # Simulating a document with 50 active tokens out of 1M vocabulary.
    print("\nDefining Sparse Input (50 active features out of 1M)...")
    active_indices = [i * 100 for i in range(50)] # 0, 100, 200...
    active_values = [1.0] * 50
    
    # 3. Execution: Sliced Projection
    # We want to inspect the first 100 output features (Dense Slice).
    # Standard use case: Embedding projection.
    print("Executing Forward Pass (Sliced Output: First 100 neurons)...")
    
    start_time = time.perf_counter()
    
    target_outputs = list(range(100))
    result = layer.forward_sparse_sliced(active_indices, active_values, target_outputs)
    
    latency = (time.perf_counter() - start_time) * 1000
    print(f"Compute Latency: {latency:.4f} ms")
    print(f"Result Vector (First 5): {result[:5]}")
    
    # 4. Consistency Check
    print("\nVerifying Determinism...")
    result_2 = layer.forward_sparse_sliced(active_indices, active_values, target_outputs)
    if result == result_2:
        print("Status: DETERMINISTIC")
    else:
        print("Status: NON-DETERMINISTIC (FAIL)")
        
    print("\n--- Test Complete ---")

if __name__ == "__main__":
    run_functional_test()
