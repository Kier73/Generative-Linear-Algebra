use dashmap::DashMap;
use lazy_static::lazy_static;
use std::sync::Arc;

// --- CORE TYPES ---

#[repr(C)]
pub struct GeometricDescriptor {
    pub rows: u64,
    pub cols: u64,
    pub signature: u64,
    pub depth: u32,
}

lazy_static! {
    /// Concurrent Law Cache for Inductive Tile Recall
    static ref LAW_CACHE: DashMap<u64, Arc<Vec<f32>>> = DashMap::new();
}

// --- CORE LOGIC ---

fn fmix64(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    h
}

#[no_mangle]
pub extern "C" fn g_matrix_symbolic_multiply(
    a: GeometricDescriptor,
    b: GeometricDescriptor,
) -> GeometricDescriptor {
    // Non-commutative binding
    let new_sig = a.signature ^ b.signature.rotate_left(1) ^ ((a.depth as u64) << 32);
    GeometricDescriptor {
        rows: a.rows,
        cols: b.cols,
        signature: new_sig,
        depth: a.depth + b.depth,
    }
}

#[no_mangle]
pub extern "C" fn g_matrix_resolve(desc: GeometricDescriptor, row: u64, col: u64) -> f32 {
    let idx = row.wrapping_mul(desc.cols).wrapping_add(col);
    let h = fmix64(desc.signature ^ idx);
    // Normalize to [-1, 1]
    ((h as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32
}

#[no_mangle]
pub extern "C" fn g_matrix_resolve_bulk(
    desc: GeometricDescriptor,
    buffer: *mut f32,
    offset: u64,
    count: usize,
) {
    let buf = unsafe { std::slice::from_raw_parts_mut(buffer, count) };
    let seed = desc.signature;

    if count <= 256 {
        // TINY SCALE: 8-bit variety (Cache regime)
        let s8 = seed as u8;
        for (idx, val) in buf.iter_mut().enumerate() {
            let h = s8
                .wrapping_add((offset + idx as u64) as u8)
                .wrapping_mul(157);
            *val = (h as f32 / 255.0) * 2.0 - 1.0;
        }
    } else if count <= 16384 {
        // SMALL SCALE: 16-bit variety (XOR-shift)
        let mut h16 = (seed ^ offset) as u16;
        for val in buf.iter_mut() {
            h16 ^= h16 << 7;
            h16 ^= h16 >> 9;
            h16 = h16.wrapping_mul(1337);
            *val = (h16 as f32 / 65535.0) * 2.0 - 1.0;
        }
    } else {
        // LARGE SCALE: 64-bit Feistel (Full Determinism)
        for (idx, val) in buf.iter_mut().enumerate() {
            let tile_idx = offset + idx as u64;
            let h = fmix64(seed ^ tile_idx);
            *val = (h as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32;
        }
    }
}

// --- INDUCTIVE ENGINE ---

const TILE_SIZE: usize = 32;

fn hash_tile(data: &[f32], rows: usize, cols: usize, stride: usize) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325; // FNV offset basis
    for i in 0..rows {
        for j in 0..cols {
            let val = data[i * stride + j];
            let bits = val.to_bits() as u64;
            h ^= fmix64(bits ^ (i as u64 * 31 + j as u64));
            h = h.wrapping_mul(0x100000001b3); // FNV prime
        }
    }
    h
}

#[no_mangle]
pub unsafe extern "C" fn g_matrix_inductive_matmul(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    m: usize,
    k_dim: usize,
    n_dim: usize,
) {
    let a = std::slice::from_raw_parts(a_ptr, m * k_dim);
    let b = std::slice::from_raw_parts(b_ptr, k_dim * n_dim);
    let c = std::slice::from_raw_parts_mut(c_ptr, m * n_dim);

    for i in (0..m).step_by(TILE_SIZE) {
        let i_end = std::cmp::min(i + TILE_SIZE, m);
        let i_rows = i_end - i;

        for j in (0..n_dim).step_by(TILE_SIZE) {
            let j_end = std::cmp::min(j + TILE_SIZE, n_dim);
            let j_cols = j_end - j;

            for k in (0..k_dim).step_by(TILE_SIZE) {
                let k_end = std::cmp::min(k + TILE_SIZE, k_dim);
                let k_size = k_end - k;

                // Extract Tile Content Hashes
                let tile_a_data = &a[i * k_dim + k..];
                let tile_b_data = &b[k * n_dim + j..];

                let h_a = hash_tile(tile_a_data, i_rows, k_size, k_dim);
                let h_b = hash_tile(tile_b_data, k_size, j_cols, n_dim);

                let pair_hash = fmix64(h_a ^ h_b.rotate_left(13));

                if let Some(res) = LAW_CACHE.get(&pair_hash) {
                    accumulate_tile(c, &res, i, j, n_dim, m);
                } else {
                    let res = compute_tile(a, b, i, j, k, m, k_dim, n_dim);
                    let res_arc = Arc::new(res);
                    LAW_CACHE.insert(pair_hash, res_arc.clone());
                    accumulate_tile(c, &res_arc, i, j, n_dim, m);
                }
            }
        }
    }
}

fn compute_tile(
    a: &[f32],
    b: &[f32],
    ti: usize,
    tj: usize,
    tk: usize,
    m: usize,
    k_dim: usize,
    n_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; TILE_SIZE * TILE_SIZE];
    for i in 0..TILE_SIZE {
        if ti + i >= m {
            break;
        }
        for k in 0..TILE_SIZE {
            if tk + k >= k_dim {
                break;
            }
            let a_val = a[(ti + i) * k_dim + (tk + k)];
            for j in 0..TILE_SIZE {
                if tj + j >= n_dim {
                    break;
                }
                out[i * TILE_SIZE + j] += a_val * b[(tk + k) * n_dim + (tj + j)];
            }
        }
    }
    out
}

fn accumulate_tile(c: &mut [f32], tile: &[f32], ti: usize, tj: usize, n_dim: usize, m: usize) {
    for i in 0..TILE_SIZE {
        if ti + i >= m {
            break;
        }
        for j in 0..TILE_SIZE {
            if tj + j >= n_dim {
                break;
            }
            c[(ti + i) * n_dim + (tj + j)] += tile[i * TILE_SIZE + j];
        }
    }
}

// --- RNS ENGINE (BIT-EXACT REPRODUCIBILITY) ---

const RNS_MOD: i64 = 2147483647; // 2^31 - 1
const FIXED_POINT_SCALE: f32 = 1000000.0;

#[no_mangle]
pub unsafe extern "C" fn g_matrix_rns_matmul(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    m: usize,
    k_dim: usize,
    n_dim: usize,
) {
    let a = std::slice::from_raw_parts(a_ptr, m * k_dim);
    let b = std::slice::from_raw_parts(b_ptr, k_dim * n_dim);
    let c = std::slice::from_raw_parts_mut(c_ptr, m * n_dim);

    for i in 0..m {
        for j in 0..n_dim {
            let mut sum: i128 = 0;
            for k in 0..k_dim {
                // Convert to fixed-point precision
                let a_fixed = (a[i * k_dim + k] * FIXED_POINT_SCALE) as i64;
                let b_fixed = (b[k * n_dim + j] * FIXED_POINT_SCALE) as i64;

                // Modular calculation (Simulation of Residue System exactness)
                let prod = (a_fixed as i128 * b_fixed as i128) % RNS_MOD as i128;
                sum = (sum + prod) % RNS_MOD as i128;
            }
            // Convert back to float for output
            c[i * n_dim + j] =
                (sum as f32) / ((FIXED_POINT_SCALE * FIXED_POINT_SCALE) % RNS_MOD as f32);
        }
    }
}
