import sys
import os
import subprocess
import glob
import time

def full_sweep():
    start_time = time.perf_counter()
    print("="*80)
    print("GENERATIVE LINEAR ALGEBRA - FINAL FULL SWEEP VALIDATION")
    print("="*80)
    
    # 1. Discover Tests
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_files = glob.glob(os.path.join(test_dir, "test_*.py"))
    
    results = []
    
    print(f"Found {len(test_files)} test suites in {test_dir}\n")
    
    # 2. Execute Each Test
    for test_file in test_files:
        test_name = os.path.basename(test_file)
        print(f"Running {test_name}...", end=" ", flush=True)
        
        t0 = time.perf_counter()
        try:
            # Run in a subprocess to ensure isolation
            result = subprocess.run(
                [sys.executable, test_file],
                capture_output=True,
                text=True,
                check=False # Don't raise exception, manually check return code
            )
            duration = (time.perf_counter() - t0)
            
            if result.returncode == 0:
                print(f"[PASS] ({duration:.2f}s)")
                results.append((test_name, "PASS", duration))
            else:
                print(f"[FAIL] ({duration:.2f}s)")
                print("-" * 40)
                print(f"ERROR OUTPUT For {test_name}:")
                print(result.stderr)
                print(result.stdout)
                print("-" * 40)
                results.append((test_name, "FAIL", duration))
                
        except Exception as e:
            print(f"[CRASH] {e}")
            results.append((test_name, "CRASH", 0))

    # 3. Summary Report
    print("\n" + "="*80)
    print("FULL SWEEP SUMMARY REPORT")
    print("="*80)
    
    passed = 0
    failed = 0
    
    print(f"{'Test Suite':<40} | {'Status':<10} | {'Time'}")
    print("-" * 70)
    
    for name, status, duration in results:
        print(f"{name:<40} | {status:<10} | {duration:.2f}s")
        if status == "PASS":
            passed += 1
        else:
            failed += 1
            
    print("-" * 70)
    total_time = time.perf_counter() - start_time
    print(f"Total Time: {total_time:.2f}s")
    print(f"Passed:     {passed}/{len(test_files)}")
    print(f"Failed:     {failed}/{len(test_files)}")
    
    if failed == 0:
        print("\n[OK] SYSTEM INTEGRITY VERIFIED. RELEASE CANDIDATE READY.")
        sys.exit(0)
    else:
        print("\n[X] CRITICAL FAILURES DETECTED. DO NOT RELEASE.")
        sys.exit(1)

if __name__ == "__main__":
    full_sweep()
