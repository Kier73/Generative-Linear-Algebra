/// V_Matrix Rust SDK
/// Optimized implementations for Spectral, RNS, and On-the-fly matrix paradigms.
/// Supports configurable Execution Policies for performance tuning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionPolicy {
    /// Forces single-threaded execution.
    Sequential,
    /// Forces multi-threaded execution via Rayon.
    Parallel,
    /// Automatically switches based on a threshold (default: n >= 32).
    Auto(usize),
}

impl ExecutionPolicy {
    pub fn is_parallel(&self, size: usize) -> bool {
        match self {
            ExecutionPolicy::Sequential => false,
            ExecutionPolicy::Parallel => true,
            ExecutionPolicy::Auto(threshold) => size >= *threshold,
        }
    }
}

pub mod core_primitives {
    /// Deterministic Feistel hash for parameter generation.
    pub fn v_mask(addr: u64) -> f64 {
        let mut l = (addr >> 32) as u32;
        let mut r = (addr & 0xFFFFFFFF) as u32;
        let key: u32 = 0xBF58476D;
        let mul: u32 = 0x94D049BB;

        for _ in 0..4 {
            let mut f = (r ^ key).wrapping_mul(mul);
            f = (f >> 16) ^ f;
            let temp = r;
            r = l ^ f;
            l = temp;
        }

        let combined = ((l as u64) << 32) | (r as u64);
        combined as f64 / (u64::MAX as f64)
    }

    /// Data fingerprint for structural feature space matching.
    pub fn signature(data: &[f64]) -> u64 {
        let n = data.len();
        if n == 0 {
            return 0;
        }
        let first = ((data[0] * 1e6) as i64 as u64) & 0xFFFFFFFF;
        let last = ((data[n - 1] * 1e6) as i64 as u64) & 0xFFFFFFFF;
        let mid = ((data[n / 2] * 1e6) as i64 as u64) & 0xFFFFFFFF;
        first ^ last ^ mid ^ (n as u64)
    }
}

pub mod engines {
    use super::ExecutionPolicy;
    use super::core_primitives::{signature, v_mask};
    use rayon::prelude::*;

    pub struct SpectralMatrixEngine {
        pub steps: u32,
        pub policy: ExecutionPolicy,
    }

    impl SpectralMatrixEngine {
        pub fn new(steps: u32, policy: ExecutionPolicy) -> Self {
            Self { steps, policy }
        }

        fn rule30(state: u64) -> u64 {
            // Wolfram Rule 30
            let l = state << 1;
            let r = state >> 1;
            l ^ (state | r)
        }

        fn rule90(state: u64) -> u64 {
            // Wolfram Rule 90
            (state << 1) ^ (state >> 1)
        }

        fn rule110(state: u64) -> u64 {
            // Wolfram Rule 110
            let l = state << 1;
            let r = state >> 1;
            (state | r) ^ (l & state & r)
        }

        pub fn project_element(&self, seed_a: u64, seed_b: u64, dim: usize) -> f64 {
            let interaction = seed_a ^ seed_b;
            let mut s30 = interaction;
            let mut s90 = interaction;
            let mut s110 = interaction;

            for _ in 0..self.steps {
                s30 = Self::rule30(s30);
                s90 = Self::rule90(s90);
                s110 = Self::rule110(s110);
            }

            let r30 = s30.count_ones() as f64 / 64.0;
            let r90 = s90.count_ones() as f64 / 64.0;
            let r110 = s110.count_ones() as f64 / 64.0;

            let composite = 0.5 * r30 + 0.3 * r90 + 0.2 * r110;
            (composite * 2.0 - 1.0) * (dim as f64).sqrt()
        }

        pub fn multiply(&self, a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
            let rows = a.len();
            let cols = b[0].len();
            let dim = a[0].len();

            let use_parallel = self.policy.is_parallel(rows.max(cols));

            let a_seeds: Vec<u64> = if use_parallel {
                a.par_iter().map(|row| signature(row)).collect()
            } else {
                a.iter().map(|row| signature(row)).collect()
            };

            let b_seeds: Vec<u64> = if use_parallel {
                (0..cols)
                    .into_par_iter()
                    .map(|j| {
                        let col_data: Vec<f64> = b.iter().map(|row| row[j]).collect();
                        signature(&col_data)
                    })
                    .collect()
            } else {
                (0..cols)
                    .map(|j| {
                        let col_data: Vec<f64> = b.iter().map(|row| row[j]).collect();
                        signature(&col_data)
                    })
                    .collect()
            };

            if use_parallel {
                (0..rows)
                    .into_par_iter()
                    .map(|i| {
                        (0..cols)
                            .map(|j| self.project_element(a_seeds[i], b_seeds[j], dim))
                            .collect()
                    })
                    .collect()
            } else {
                (0..rows)
                    .map(|i| {
                        (0..cols)
                            .map(|j| self.project_element(a_seeds[i], b_seeds[j], dim))
                            .collect()
                    })
                    .collect()
            }
        }
    }

    pub struct RNSMatrixEngine {
        pub primes: Vec<u64>,
    }

    impl RNSMatrixEngine {
        pub fn new(primes: Vec<u64>) -> Self {
            Self { primes }
        }

        fn extended_gcd(a: i128, b: i128) -> (i128, i128, i128) {
            if a == 0 {
                return (b, 0, 1);
            }
            let (gcd, x1, y1) = Self::extended_gcd(b % a, a);
            let x = y1 - (b / a) * x1;
            let y = x1;
            (gcd, x, y)
        }

        fn mod_inverse(a: i128, m: i128) -> i128 {
            let (_gcd, x, _y) = Self::extended_gcd(a, m);
            (x % m + m) % m
        }

        pub fn multiply(&self, a: &[Vec<f64>], b: &[Vec<f64>], scale: f64) -> Vec<Vec<f64>> {
            let rows = a.len();
            let cols = b[0].len();
            let dim = a[0].len();
            let mut final_c = vec![vec![0.0; cols]; rows];
            let mod_m: i128 = self.primes.iter().map(|&p| p as i128).product();

            let mut residues = Vec::new();
            for &p in &self.primes {
                let a_p: Vec<Vec<u64>> = a
                    .iter()
                    .map(|row| row.iter().map(|&x| (x * scale) as u64 % p).collect())
                    .collect();

                let mut b_p: Vec<Vec<u64>> = vec![vec![0u64; cols]; dim];
                for k in 0..dim {
                    for j in 0..cols {
                        b_p[k][j] = (b[k][j] * scale) as u64 % p;
                    }
                }

                let mut p_res = vec![vec![0u64; cols]; rows];
                for i in 0..rows {
                    for j in 0..cols {
                        let mut dot = 0u64;
                        for k in 0..dim {
                            dot = (dot + a_p[i][k] * b_p[k][j]) % p;
                        }
                        p_res[i][j] = dot;
                    }
                }
                residues.push(p_res);
            }

            for i in 0..rows {
                for j in 0..cols {
                    let mut val: i128 = 0;
                    for (idx, &p) in self.primes.iter().enumerate() {
                        let r = residues[idx][i][j] as i128;
                        let m_i = mod_m / p as i128;
                        let y_i = Self::mod_inverse(m_i, p as i128);
                        val = (val + r * m_i * y_i) % mod_m;
                    }
                    final_c[i][j] = (val as f64) / (scale * scale);
                }
            }
            final_c
        }
    }

    pub struct SNAPMatrixEngine {
        /// On-the-fly parameterization policy.
        pub policy: ExecutionPolicy,
    }

    impl SNAPMatrixEngine {
        pub fn multiply(&self, x: &[f64], seed: u64, out_dim: usize) -> Vec<f64> {
            if self.policy.is_parallel(out_dim) {
                (0..out_dim)
                    .into_par_iter()
                    .map(|j| {
                        let mut val = 0.0;
                        for i in 0..x.len() {
                            let w_ij = v_mask(seed ^ (j as u64) ^ (i as u64)) * 2.0 - 1.0;
                            val += x[i] * w_ij;
                        }
                        val
                    })
                    .collect()
            } else {
                (0..out_dim)
                    .map(|j| {
                        let mut val = 0.0;
                        for i in 0..x.len() {
                            let w_ij = v_mask(seed ^ (j as u64) ^ (i as u64)) * 2.0 - 1.0;
                            val += x[i] * w_ij;
                        }
                        val
                    })
                    .collect()
            }
        }
    }
}

pub struct VMatrix {
    pub spectral: engines::SpectralMatrixEngine,
    pub rns: engines::RNSMatrixEngine,
    pub on_the_fly: engines::SNAPMatrixEngine,
}

impl VMatrix {
    pub fn new(policy: ExecutionPolicy) -> Self {
        Self {
            spectral: engines::SpectralMatrixEngine::new(16, policy),
            rns: engines::RNSMatrixEngine::new(vec![10007, 10009, 10037, 10039]),
            on_the_fly: engines::SNAPMatrixEngine { policy },
        }
    }

    pub fn set_policy(&mut self, policy: ExecutionPolicy) {
        self.spectral.policy = policy;
        self.on_the_fly.policy = policy;
    }

    pub fn matmul_spectral(&self, a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
        self.spectral.multiply(a, b)
    }

    pub fn matmul_rns(&self, a: &[Vec<f64>], b: &[Vec<f64>], scale: f64) -> Vec<Vec<f64>> {
        self.rns.multiply(a, b, scale)
    }

    pub fn on_the_fly_project(&self, x: &[f64], seed: u64, out_dim: usize) -> Vec<f64> {
        self.on_the_fly.multiply(x, seed, out_dim)
    }
}
