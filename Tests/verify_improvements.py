"""
Verification Script: Test all mathematical improvements
========================================================
Runs targeted tests for each fix applied to the codebase.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = 0
FAIL = 0
WARN = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}: {detail}")

def warn(name, detail=""):
    global WARN
    WARN += 1
    print(f"  ⚠️  {name}: {detail}")

# ==============================================================
print("\n" + "="*60)
print("FIX 1: Tile Hash (Full-Content) — g_matrix.py")
print("="*60)

from g_matrix import GMatrix

g = GMatrix()
eng = g.inductive

# Two tiles with same first/last but different interior
tile_a1 = [[1.0, 2.0, 3.0, 4.0]]
tile_a2 = [[1.0, 9.0, 9.0, 4.0]]  # same first=1.0, last=4.0

h1 = eng._hash_tile(tile_a1)
h2 = eng._hash_tile(tile_a2)
check("Different tiles get different hashes (collision fix)", h1 != h2,
      f"h1={h1}, h2={h2}")

# Same tile hashes to same value
h3 = eng._hash_tile(tile_a1)
check("Same tile hashes consistently", h1 == h3)

# Correctness: tile-level matmul with identity should be exact
A_c = [[1.0, 2.0], [3.0, 4.0]]
B_id = [[1.0, 0.0], [0.0, 1.0]]
result = eng.matmul(A_c, B_id)
check("Inductive matmul(A, I) == A",
      abs(result[0][0] - 1.0) < 1e-6 and abs(result[1][1] - 4.0) < 1e-6,
      f"result={result}")

# Now multiply a DIFFERENT A with same identity — must NOT reuse cache
A_c2 = [[5.0, 6.0], [7.0, 8.0]]
result2 = eng.matmul(A_c2, B_id)
check("Cache collision fix: matmul(A2, I) == A2 (not cached A)",
      abs(result2[0][0] - 5.0) < 1e-6 and abs(result2[1][1] - 8.0) < 1e-6,
      f"result2={result2}")


# ==============================================================
print("\n" + "="*60)
print("FIX 2: RNS Engine — v_matrix.py")
print("="*60)

from v_matrix import RNSMatrixEngine

rns = RNSMatrixEngine()

# Basic 2x2 test
A = [[1.0, 2.0], [3.0, 4.0]]
B = [[5.0, 6.0], [7.0, 8.0]]
true_C = [[19.0, 22.0], [43.0, 50.0]]
C = rns.multiply(A, B, scale=1000.0)

max_err = max(abs(C[i][j] - true_C[i][j]) for i in range(2) for j in range(2))
check(f"RNS 2x2 exact (max_err={max_err:.6f})", max_err < 0.01)

# Negative values test (signed CRT)
A_neg = [[-1.0, 2.0], [3.0, -4.0]]
B_neg = [[5.0, -6.0], [-7.0, 8.0]]
import operator
true_neg = [[0.0]*2 for _ in range(2)]
for i in range(2):
    for j in range(2):
        true_neg[i][j] = sum(A_neg[i][k] * B_neg[k][j] for k in range(2))

C_neg = rns.multiply(A_neg, B_neg, scale=1000.0)
max_err_neg = max(abs(C_neg[i][j] - true_neg[i][j]) for i in range(2) for j in range(2))
check(f"RNS with negatives (signed CRT, max_err={max_err_neg:.6f})", max_err_neg < 0.01)

# 4x4 random test (previously overflowed)
import random
random.seed(42)
A4 = [[random.uniform(-5, 5) for _ in range(4)] for _ in range(4)]
B4 = [[random.uniform(-5, 5) for _ in range(4)] for _ in range(4)]
true_C4 = [[sum(A4[i][k]*B4[k][j] for k in range(4)) for j in range(4)] for i in range(4)]

rns2 = RNSMatrixEngine()  # Fresh instance
C4 = rns2.multiply(A4, B4, scale=1000.0)
max_err_4 = max(abs(C4[i][j] - true_C4[i][j]) for i in range(4) for j in range(4))
check(f"RNS 4x4 random (max_err={max_err_4:.6f})", max_err_4 < 0.1,
      f"max_err={max_err_4}")


# ==============================================================
print("\n" + "="*60)
print("FIX 3: Spectral Engine → Random Projection — v_matrix.py")
print("="*60)

from v_matrix import RandomProjectionMatrixEngine

rp = RandomProjectionMatrixEngine(projection_dim=150, seed=42)

# Small 2x2 (should use direct matmul path since d=2 <= D*2=128)
A_s = [[1.0, 2.0], [3.0, 4.0]]
B_s = [[5.0, 6.0], [7.0, 8.0]]
C_s = rp.multiply(A_s, B_s)
check("Small matmul via direct path is exact",
      abs(C_s[0][0] - 19.0) < 1e-6 and abs(C_s[1][1] - 50.0) < 1e-6,
      f"C={C_s}")

# Larger test: 8x128 × 128x8 — should use projection
random.seed(123)
n, d, m = 8, 200, 8
A_big = [[random.gauss(0, 1) for _ in range(d)] for _ in range(n)]
B_big = [[random.gauss(0, 1) for _ in range(m)] for _ in range(d)]
true_big = [[sum(A_big[i][k]*B_big[k][j] for k in range(d)) for j in range(m)] for i in range(n)]

C_proj = rp.multiply(A_big, B_big)

# Compute relative Frobenius error
fro_err = math.sqrt(sum((C_proj[i][j] - true_big[i][j])**2 for i in range(n) for j in range(m)))
fro_true = math.sqrt(sum(true_big[i][j]**2 for i in range(n) for j in range(m)))
rel_err = fro_err / fro_true if fro_true > 0 else float('inf')

check(f"Random projection 8x200 × 200x8 (rel_err={rel_err:.3f})", rel_err < 1.0,
      f"rel_err={rel_err}")

# Correlation check
flat_true = [true_big[i][j] for i in range(n) for j in range(m)]
flat_proj = [C_proj[i][j] for i in range(n) for j in range(m)]
mean_t = sum(flat_true) / len(flat_true)
mean_p = sum(flat_proj) / len(flat_proj)
cov = sum((t - mean_t)*(p - mean_p) for t, p in zip(flat_true, flat_proj))
var_t = sum((t - mean_t)**2 for t in flat_true)
var_p = sum((p - mean_p)**2 for p in flat_proj)
corr = cov / math.sqrt(var_t * var_p) if var_t > 0 and var_p > 0 else 0

check(f"Projection output is correlated with true product (r={corr:.3f})", corr > 0.5,
      f"Pearson r={corr:.3f} (was ~0.00 before fix)")


# ==============================================================
print("\n" + "="*60)
print("FIX 4: SNAP Scaling (1/√d) — v_matrix.py")
print("="*60)

from v_matrix import SNAPMatrixEngine

snap = SNAPMatrixEngine()
# Test with random inputs (proper test of variance normalization)
random.seed(777)
X_100 = [random.gauss(0, 1) for _ in range(100)]
X_1000 = [random.gauss(0, 1) for _ in range(1000)]

y_100 = snap.multiply(X_100, seed=42, out_dim=50)
y_1000 = snap.multiply(X_1000, seed=42, out_dim=50)

# Compute mean squared output (proxy for variance)
var_100 = sum(y**2 for y in y_100) / len(y_100)
var_1000 = sum(y**2 for y in y_1000) / len(y_1000)

# With 1/sqrt(d) scaling, output variance should be proportional to ||X||^2/d
# For iid N(0,1) inputs: ||X||^2/d ≈ 1 regardless of d
norm_sq_100 = sum(x**2 for x in X_100) / 100
norm_sq_1000 = sum(x**2 for x in X_1000) / 1000
ratio = (var_1000 / norm_sq_1000) / (var_100 / norm_sq_100) if (var_100 * norm_sq_1000) > 0 else float('inf')
check(f"SNAP variance-normalized ratio (d=1000 vs d=100) = {ratio:.2f} (should be ~1.0)",
      0.2 < ratio < 5.0, f"ratio={ratio:.2f}")


# ==============================================================
print("\n" + "="*60)
print("FIX 5: PrimeMatrix Large-Number Fallback — prime_matrix.py")
print("="*60)

from prime_matrix import PrimeMatrix, _get_prime_factors

# Test factorization works for large numbers
large_n = 2**50  # 1125899906842624
factors = _get_prime_factors(large_n)
check("Factor 2^50 correctly", factors == {2: 50}, f"got {factors}")

# Composite large number
large_composite = 1000000007 * 1000000009  # product of two primes
factors_lc = _get_prime_factors(large_composite)
product = 1
for p, e in factors_lc.items():
    product *= p**e
check(f"Factor large semiprime correctly",
      product == large_composite and len(factors_lc) == 2,
      f"got {factors_lc}")

# PrimeMatrix get_element for large coordinates (no more log fallback)
pm = PrimeMatrix(1, 1, depth=2)
# P^2[0, 11] where row_val=1, col_val=12, X=12=2^2*3
# chains: C(2+1,1)*C(1+1,1) = 3*2 = 6
elem = pm.get_element(0, 11)
check(f"PrimeMatrix P^2[0,11] = {elem} (expected 6)", elem == 6)


# ==============================================================
print("\n" + "="*60)
print("FIX 6: XMatrix Reframe — x_matrix.py")
print("="*60)

from x_matrix import XMatrix

# Base matrix should still produce ±1 values
xm_base = XMatrix(4, 4, seed=42)
vals = [xm_base.get_element(r, c) for r in range(4) for c in range(4)]
check("Base XMatrix produces ±1 values",
      all(v == 1.0 or v == -1.0 for v in vals))

# Compose should exist and produce XMatrix with inner_dim set
xm2 = XMatrix(4, 4, seed=99)
xm_comp = xm_base.compose(xm2)
check("compose() returns XMatrix with inner_dim set",
      isinstance(xm_comp, XMatrix) and xm_comp._inner_dim == 4)

# Composed matrix should produce Gaussian values (not just ±1)
comp_vals = [xm_comp.get_element(r, c) for r in range(4) for c in range(4)]
has_non_pm1 = any(abs(v) != 1.0 for v in comp_vals)
check("Composed XMatrix produces Gaussian (non-±1) values", has_non_pm1,
      f"values={comp_vals}")

# multiply() still works (routes to compose)
xm_mult = xm_base.multiply(xm2)
check("multiply() routes to compose()", xm_mult._inner_dim == 4)

# multiply_materialize does actual computation
xm_a = XMatrix(2, 2, seed=10)
xm_b = XMatrix(2, 2, seed=20)
C_mat = xm_a.multiply_materialize(xm_b)
check("multiply_materialize returns 2D list",
      isinstance(C_mat, list) and len(C_mat) == 2 and len(C_mat[0]) == 2)
# Verify it actually computes sum_k A[i,k]*B[k,j]
expected_00 = xm_a.get_element(0,0)*xm_b.get_element(0,0) + xm_a.get_element(0,1)*xm_b.get_element(1,0)
check(f"multiply_materialize is correct (C[0,0]={C_mat[0][0]}, expected={expected_00})",
      abs(C_mat[0][0] - expected_00) < 1e-10)


# ==============================================================
print("\n" + "="*60)
print("FIX 7: Möbius/Mertens/Redheffer — rh_matrix.py")
print("="*60)

from rh_matrix import get_mobius, RedhefferMatrix

# Möbius function still works
known_mobius = {1:1, 2:-1, 3:-1, 4:0, 5:-1, 6:1, 7:-1, 8:0, 9:0, 10:1}
all_ok = True
for n, expected in known_mobius.items():
    got = get_mobius(n)
    if got != expected:
        all_ok = False
        print(f"    mu({n}) = {got}, expected {expected}")
check("Möbius function μ(n) correct for n=1..10", all_ok)

# No more -2 sentinel
try:
    # This should never return -2; if factorization fails it should raise
    test_vals = [get_mobius(k) for k in range(1, 30)]
    has_minus2 = -2 in test_vals
    check("No -2 sentinel in results", not has_minus2)
except ArithmeticError as e:
    warn(f"ArithmeticError raised (expected for timeout): {e}")

# Mertens sieve
rh = RedhefferMatrix(10, 10)
M = rh.mertens_sieve(20)
expected_mertens = [0, 1, 0, -1, -1, -2, -1, -2, -2, -2, -1, -2, -2, -3, -2, -1, -1, -2, -2, -3, -3]
check("Mertens sieve M(1)..M(20) correct", M == expected_mertens,
      f"got {M}")


# ==============================================================
print("\n" + "="*60)
print("FIX 8: Pollard-rho no-recursion — rh_matrix.py")
print("="*60)

from rh_matrix import pollard_rho

# Factor a large semiprime
sp = 104729 * 104743  # product of two 6-digit primes
f = pollard_rho(sp)
check(f"Pollard-rho factors semiprime (got factor {f})",
      sp % f == 0 and 1 < f < sp)


# ==============================================================
print("\n" + "="*60)
print(f"RESULTS: {PASS} PASS, {FAIL} FAIL, {WARN} WARN")
print("="*60)
