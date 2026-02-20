import sys
import os
import ctypes

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sdk_registry import Registry

# Import modules to trigger registration
import v_matrix
import g_matrix
import x_matrix

def test_unified_api():
    print("="*60)
    print("UNIFIED API REGISTRY VERIFICATION")
    print("="*60)

    solvers = Registry.list_solvers()
    print(f"Registered Solvers: {solvers}")
    
    expected = ['vmatrix', 'gmatrix', 'xmatrix']
    for e in expected:
        if e not in solvers:
            print(f"FAILED: Solver {e} not found.")
            return
        
    print("\n[Method Verification]")
    for s in solvers:
        print(f"Solver: {s.upper()}")
        methods = Registry._methods.get(s, {})
        for m_name, func in methods.items():
            print(f"  - @method: {m_name} -> {func.__name__}")
            
    # Instantiate and check via Registry
    print("\n[Instantiation Test]")
    try:
        XMatClass = Registry.get_solver("XMatrix")
        xm = XMatClass(128, 128)
        print(f"  Successfully instantiated {xm.__class__.__name__}")
        
        # Check if we can invoke the multiply method via standard object call (decorators shouldn't break this)
        ym = XMatClass(128, 128)
        zm = xm.multiply(ym)
        print(f"  Successfully invoked decorated method 'multiply'")
        print(f"    Result Label: {zm.manifold.label}")
        
    except Exception as e:
        print(f"FAILED Instantiation/Invocation: {e}")
        import traceback
        traceback.print_exc()

    print("="*60)

if __name__ == "__main__":
    test_unified_api()
