use std::slice;

pub const HDC_DIM: usize = 1024;
pub const U64_COUNT: usize = 16; // 1024 / 64

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hdc1024 {
    pub data: [u64; U64_COUNT],
}

impl Hdc1024 {
    pub fn new() -> Self {
        Hdc1024 {
            data: [0; U64_COUNT],
        }
    }

    /// Binding operation (XOR).
    pub fn bind(&self, other: &Self) -> Self {
        let mut res = [0u64; U64_COUNT];
        for i in 0..U64_COUNT {
            res[i] = self.data[i] ^ other.data[i];
        }
        Hdc1024 { data: res }
    }

    /// Directional Permutation (Circular Shift).
    /// Highly optimized for 1024 bits using block moves and bit shifts.
    pub fn shift(&self, n: usize) -> Self {
        let n = n % HDC_DIM;
        if n == 0 {
            return *self;
        }

        let word_shift = n / 64;
        let bit_shift = n % 64;
        let mut res = [0u64; U64_COUNT];

        if bit_shift == 0 {
            for i in 0..U64_COUNT {
                res[(i + word_shift) % U64_COUNT] = self.data[i];
            }
        } else {
            let inv_bit_shift = 64 - bit_shift;
            for i in 0..U64_COUNT {
                let target_idx = (i + word_shift) % U64_COUNT;
                let next_idx = (target_idx + 1) % U64_COUNT;

                res[target_idx] |= self.data[i] << bit_shift;
                res[next_idx] |= self.data[i] >> inv_bit_shift;
            }
        }
        Hdc1024 { data: res }
    }

    /// Hamming Distance based Similarity.
    pub fn similarity(&self, other: &Self) -> f32 {
        let mut hamming: u32 = 0;
        for i in 0..U64_COUNT {
            hamming += (self.data[i] ^ other.data[i]).count_ones();
        }
        1.0 - (2.0 * hamming as f32 / HDC_DIM as f32)
    }

    pub fn generate_signature(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for &word in self.data.iter() {
            h ^= word;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }
}

// --- FFI EXPORTS ---

#[no_mangle]
pub extern "C" fn x_matrix_bind(a: *const Hdc1024, b: *const Hdc1024, out: *mut Hdc1024) {
    let a = unsafe { &*a };
    let b = unsafe { &*b };
    let out = unsafe { &mut *out };
    *out = a.bind(b);
}

#[no_mangle]
pub extern "C" fn x_matrix_shift(a: *const Hdc1024, n: usize, out: *mut Hdc1024) {
    let a = unsafe { &*a };
    let out = unsafe { &mut *out };
    *out = a.shift(n);
}

#[no_mangle]
pub extern "C" fn x_matrix_similarity(a: *const Hdc1024, b: *const Hdc1024) -> f32 {
    let a = unsafe { &*a };
    let b = unsafe { &*b };
    a.similarity(b)
}

#[no_mangle]
pub extern "C" fn x_matrix_generate_signature(a: *const Hdc1024) -> u64 {
    let a = unsafe { &*a };
    a.generate_signature()
}

#[no_mangle]
pub extern "C" fn x_matrix_fmix64(h: u64) -> u64 {
    let mut h = h;
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    h
}

#[no_mangle]
pub extern "C" fn x_matrix_resolve_interaction(
    base_seed: u64,
    row_seed: u64,
    col_seed: u64,
    out: *mut Hdc1024,
) {
    let base = Hdc1024::from_seed(base_seed);
    let row_desc = Hdc1024::from_seed(row_seed);
    let col_desc = Hdc1024::from_seed(col_seed);

    // interaction = row_desc.bind(col_desc.shift(1))
    let interaction = row_desc.bind(&col_desc.shift(1));

    // resolved = base.bind(interaction)
    let resolved = base.bind(&interaction);

    let out = unsafe { &mut *out };
    *out = resolved;
}

impl Hdc1024 {
    pub fn from_seed(seed: u64) -> Self {
        let mut data = [0u64; U64_COUNT];
        let mut s = seed;
        for i in 0..U64_COUNT {
            s = s.wrapping_add(0x9e3779b97f4a7c15);
            data[i] = x_matrix_fmix64(s);
        }
        Hdc1024 { data }
    }
}
