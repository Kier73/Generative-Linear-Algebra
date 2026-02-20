import sys
import os
import time
import math
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from prime_matrix import PrimeMatrix

class SimpleMLP:
    """A minimal 2-layer MLP implemented in NumPy."""
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def train_step(self, x, y, lr=0.01):
        m = x.shape[0]
        # Forward
        a2 = self.forward(x)
        
        # Loss (Binary Cross Entropy)
        loss = -np.mean(y * np.log(a2 + 1e-8) + (1 - y) * np.log(1 - a2 + 1e-8))
        
        # Backward
        dz2 = a2 - y
        dw2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        da1 = np.dot(dz2, self.w2.T)
        dz1 = da1 * (self.z1 > 0)
        dw1 = np.dot(x.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update
        self.w1 -= lr * dw1
        self.b1 -= lr * db1
        self.w2 -= lr * dw2
        self.b2 -= lr * db2
        
        return loss

def run_generative_training():
    print("="*80)
    print("USE CASE: GENERATIVE SYNTHETIC DATA STREAMING")
    print("="*80)
    
    # 1. Setup the Synthetic Data Source
    N = 10**100
    print(f"Data Source: Analytical Matrix (N=10^100)")
    P = PrimeMatrix(N, N)
    
    # 2. Setup the Neural Network
    input_size = 2 # (row_index, col_index)
    mlp = SimpleMLP(input_size, hidden_size=64, output_size=1)
    
    # 3. Training Loop (Streaming data from the descriptor)
    print("\n[PHASE 1] Training on Procedural Dataset...")
    EPOCHS = 10
    BATCH_SIZE = 128
    STEPS_PER_EPOCH = 100
    
    # We'll normalize the input indices to help convergence
    def normalize(val):
        return (val % 100) / 100.0 # Focus on local periodicity
    
    start_time = time.perf_counter()
    for epoch in range(EPOCHS):
        total_loss = 0
        for _ in range(STEPS_PER_EPOCH):
            batch_x = []
            batch_y = []
            for _ in range(BATCH_SIZE):
                # Sample from a specific coordinate region
                r = random.randint(1, 100)
                c = random.randint(1, 100)
                batch_x.append([normalize(r), normalize(c)])
                batch_y.append([P.get_element(r-1, c-1)])
            
            x = np.array(batch_x)
            y = np.array(batch_y)
            loss = mlp.train_step(x, y, lr=0.2)
            total_loss += loss
            
        avg_loss = total_loss / STEPS_PER_EPOCH
        if (epoch + 1) % 2 == 0:
            print(f"  > Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.6f}")
        
    t_train = time.perf_counter() - start_time
    print(f"\n[PHASE 2] Training Complete in {t_train:.2f} seconds.")
    print(f"  Total samples ingested: {EPOCHS * STEPS_PER_EPOCH * BATCH_SIZE}")
    print(f"  Storage occupied by dataset: 0.00 bytes (Zero Storage)")
    
    # 4. Pattern Recognition Test
    print("\n[PHASE 3] Pattern Recognition Test (Predicting Divisibility)...")
    test_cases = [(2, 4), (3, 6), (2, 3), (10, 20), (5, 7)]
    for r, c in test_cases:
        x_norm = np.array([[normalize(r), normalize(c)]])
        pred = mlp.forward(x_norm)[0][0]
        actual = P.get_element(r-1, c-1)
        print(f"  - Query ({r}, {c}) | Pred: {pred:.4f} | Actual: {actual} | {'OK' if round(pred) == actual else 'MISS'}")

    print("\n" + "="*80)
    print("DEMO COMPLETE: MODEL TRAINED ON PROCEDURAL DATA")
    print("="*80)

if __name__ == "__main__":
    import random
    run_generative_training()
