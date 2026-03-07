# tests/test_pipeline/test_balloon_pipeline.py
import trimesh
import logging

from sigil.pipeline.balloon_pipeline import *

def test_balloon_sphere():
    mesh   = trimesh.creation.icosphere(subdivisions=4)
    config = BalloonConfig(
        n_surface        = 5000,
        epsilon          = 0.01,
        max_degree       = 8,
        gd_steps         = 1000,
        rmse_threshold   = 1e-3,
        plateau_patience = 2,
    )
    equation = compile_mesh_balloon(mesh, config)
    # expect degree 2 to converge -- sphere IS a degree 2 polynomial
    assert equation.rmse < 1e-2
    assert equation.degree == 2   # should not need to go higher
    return True, equation

def test_balloon_bunny():
    mesh = trimesh.load(r'C:\Users\yalam\Documents\sigil-compiler\assets\meshes\bunny-high.obj')
    original_scale = mesh.scale
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale(1.0 / original_scale)
    logging.info(f"Bunny original scale: {original_scale:.6f}")
    config = BalloonConfig(
        n_surface        = 20000,
        epsilon          = 0.05,    # 10x larger -- strong signal
        max_degree       = 20,
        gd_steps         = 3000,
        gd_lr            = 0.001,
        rmse_threshold   = 1e-2,    # relative to epsilon now
        plateau_patience = 3,
        visualize_progress = True,
    )
    equation = compile_mesh_pysr(mesh, config)

    import sympy
    # Save equation to file
    x0, x1, x2 = sympy.symbols('x0 x1 x2')
    with open(r'C:\Users\yalam\Documents\sigil-compiler\outputs\bunny_equation_pysr.txt', 'w') as out:
        out.write(str(equation.sympy_expr) + '\n')
        out.write(str(sympy.diff(equation.sympy_expr, x0)) + '\n')
        out.write(str(sympy.diff(equation.sympy_expr, x1)) + '\n')
        out.write(str(sympy.diff(equation.sympy_expr, x2)) + '\n')    


    return True, equation

def test_balloon_torus():
    # Create torus mesh -- trimesh can generate this directly
    mesh = trimesh.creation.torus(major_radius=1.0, minor_radius=0.3)
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale(1.0 / mesh.scale)

    config = BalloonConfig(
        n_surface        = 10000,
        epsilon          = 0.05,
        max_degree       = 8,      # torus is degree 4, so 8 is generous
        gd_steps         = 2000,
        gd_lr            = 0.001,
        rmse_threshold   = 1e-3,
        plateau_patience = 3,
        visualize_progress = True,
    )
    equation = compile_mesh_balloon(mesh, config)
    return True, equation    

def test_balloon_cube():
    # Create cube mesh -- trimesh can generate this directly
    mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale(1.0 / mesh.scale)

    config = BalloonConfig(
        n_surface        = 10000,
        epsilon          = 0.05,
        max_degree       = 8,      
        gd_steps         = 2000,
        gd_lr            = 0.001,
        rmse_threshold   = 1e-3,
        plateau_patience = 3,
        visualize_progress = True,
    )
    equation = compile_mesh_balloon(mesh, config)
    return True, equation    

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    # ("sphere", test_balloon_sphere),    
    ("torus", test_balloon_torus),
    # ("cube", test_balloon_cube),
    # ("bunny", test_balloon_bunny),
]

def run_all():
    print("\n=== balloon_pipeline tests ===\n")
    passed = 0
    failed = 0

    for name, test_fn in TESTS:
        try:
            ok, result = test_fn()
            print(f"\n  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"\n  FAIL  {name}: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"\n  ERROR {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed}/{passed+failed} tests passed")
    if failed > 0:
        sys.exit(1)

if __name__ == '__main__':
    run_all()