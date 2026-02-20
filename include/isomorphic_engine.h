/**
 * Isomorphic Matrix Engine - Unified FFI Header
 * --------------------------------------------
 * This header provides the C-compatible signatures for interacting with 
 * the high-performance Rust backends (G-Matrix and X-Matrix).
 */

#ifndef ISOMORPHIC_ENGINE_H
#define ISOMORPHIC_ENGINE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// --- Generation 2: G-Matrix (Inductive Logic) ---

typedef struct {
    uint64_t rows;
    uint64_t cols;
    uint64_t signature;
    uint32_t depth;
} GeometricDescriptor;

GeometricDescriptor g_matrix_symbolic_multiply(GeometricDescriptor a, GeometricDescriptor b);
float g_matrix_resolve_element(GeometricDescriptor desc, uint64_t row, uint64_t col);

// --- Generation 3.5: X-Matrix (HDC Manifold) ---

typedef struct {
    uint64_t data[16]; // 1024-bit bitset
} Hdc1024;

void x_matrix_bind(const Hdc1024* a, const Hdc1024* b, Hdc1024* out);
void x_matrix_shift(const Hdc1024* a, size_t n, Hdc1024* out);
float x_matrix_similarity(const Hdc1024* a, const Hdc1024* b);
uint64_t x_matrix_generate_signature(const Hdc1024* a);

/**
 * Combined high-speed resolution path for semantic interactions.
 */
void x_matrix_resolve_interaction(
    uint64_t base_seed, 
    uint64_t row_seed, 
    uint64_t col_seed, 
    Hdc1024* out
);

#ifdef __cplusplus
}
#endif

#endif // ISOMORPHIC_ENGINE_H
