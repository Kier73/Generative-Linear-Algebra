use std::time::Instant;
use v_matrix_rust::{ExecutionPolicy, VMatrix};

fn generate_matrix(rows: usize, cols: usize) -> Vec<Vec<f64>> {
    let mut mat = Vec::with_capacity(rows);
    for i in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for j in 0..cols {
            row.push(((i ^ j) as f64 / 100.0) % 1.0);
        }
        mat.push(row);
    }
    mat
}

fn main() {
    println!("============================================================");
    println!("MATRIX MULTIPLICATION SDK: PERFORMANCE POLICY BENCHMARK");
    println!("============================================================");

    // Test with Auto policy (threshold 32)
    let mut vm = VMatrix::new(ExecutionPolicy::Auto(32));
    let sizes = vec![8, 16, 32, 64, 128, 256];

    for &n in &sizes {
        println!("\n--- Matrix Size: {}x{} ---", n, n);
        let a = generate_matrix(n, n);
        let b = generate_matrix(n, n);

        // 1. Sequential Mode
        vm.set_policy(ExecutionPolicy::Sequential);
        let start = Instant::now();
        let _ = vm.matmul_spectral(&a, &b);
        let seq_time = start.elapsed().as_secs_f64() * 1000.0;
        println!("  Sequential : {:.4} ms", seq_time);

        // 2. Parallel Mode
        vm.set_policy(ExecutionPolicy::Parallel);
        let start = Instant::now();
        let _ = vm.matmul_spectral(&a, &b);
        let par_time = start.elapsed().as_secs_f64() * 1000.0;
        println!("  Parallel   : {:.4} ms", par_time);

        // 3. Auto Mode (threshold 32)
        vm.set_policy(ExecutionPolicy::Auto(32));
        let start = Instant::now();
        let _ = vm.matmul_spectral(&a, &b);
        let auto_time = start.elapsed().as_secs_f64() * 1000.0;
        let selected = if n >= 32 { "Parallel" } else { "Sequential" };
        println!(
            "  Auto (32)  : {:.4} ms (Selected: {})",
            auto_time, selected
        );
        println!("[BENCH] SDK_Rust_Auto:{}x{}:{:.4}ms", n, n, auto_time);

        let best = if seq_time < par_time {
            "Sequential"
        } else {
            "Parallel"
        };
        println!("  [RECOMMENDED] {}", best);
    }

    println!("\n============================================================");
    println!("BENCHMARK COMPLETE");
    println!("============================================================");
}
