# tests/test_pipeline/test_geometry_pipeline.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
import trimesh
import logging

from sigil.pipeline.geometry_pipeline import compile_mesh, PipelineConfig

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def test_pipeline_sphere():
    """
    Full pipeline smoke test on a unit icosphere.

    Verifies:
        1. Pipeline runs to completion without error
        2. Returns a single Equation
        3. Equation is finite (no NaN/Inf)
        4. Zero level set of the equation lies close to radius 1.0
           -- the only geometric ground truth we have

    This is intentionally a coarse test. The visual quality check
    happens in the prototype ray tracer.
    """
    logging.basicConfig(level=logging.INFO)

    mesh = trimesh.creation.icosphere(subdivisions=4)

    config = PipelineConfig(
        n_seeds            = 16,
        sr_degree          = 4,
        sr_refine_steps    = 500,
        max_gpr_points     = 300,
        gpr_resolution     = 20,
        pysr_niterations   = 25,
        pysr_populations   = 15,
        pysr_root_multiplier = 4,
        smin_k             = 0.1,
        merge_refine_steps = 500,
        merge_refine_lr    = 1e-3,
        n_workers          = 4,
        merge_strategy     = 'polynomial'
    ) 

    print("\n=== pipeline smoke test: sphere ===\n")
    print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"Config: {config}\n")

    equation = compile_mesh(mesh, config)

    # 1. Got an equation back
    assert equation is not None, "compile_mesh returned None"

    from sigil.geometry.sr.base import Equation
    assert isinstance(equation, Equation), \
        f"compile_mesh should return Equation, got {type(equation)}"

    print(f"\nFinal equation: {equation.sympy_expr}")
    print(f"Final RMSE:     {equation.rmse:.6f}")

    # 2. Equation evaluates without error
    # Sample a grid over [-1.5, 1.5]^3
    lin   = np.linspace(-1.5, 1.5, 30)
    xx, yy, zz = np.meshgrid(lin, lin, lin, indexing='ij')
    X_grid = np.column_stack([
        xx.ravel(),
        yy.ravel(),
        zz.ravel(),
    ])   # (27000, 3)

    y_grid = equation(X_grid)   # (27000,)

    assert not np.any(np.isnan(y_grid)), \
        "NaN in equation output over evaluation grid"
    assert not np.any(np.isinf(y_grid)), \
        "Inf in equation output over evaluation grid"

    print(f"Grid evaluation: {len(X_grid)} points, no NaN/Inf")
    print(f"Value range: [{y_grid.min():.4f}, {y_grid.max():.4f}]")

    # 3. Zero level set lies near radius 1.0
    threshold = 0.1
    near_zero = np.abs(y_grid) < threshold
    n_near    = near_zero.sum()

    assert n_near > 0, \
        (f"No points near zero level set (threshold={threshold}). "
         f"Pipeline may have failed silently. "
         f"Value range: [{y_grid.min():.4f}, {y_grid.max():.4f}]")

    radii  = np.linalg.norm(X_grid[near_zero], axis=1)
    mean_r = radii.mean()
    std_r  = radii.std()

    print(f"Zero level set: {n_near} near-zero points")
    print(f"Mean radius:    {mean_r:.4f} (expect 1.0)")
    print(f"Std radius:     {std_r:.4f}")

    assert 0.85 < mean_r < 1.15, \
        (f"Zero level set mean radius {mean_r:.4f} too far from 1.0. "
         f"Pipeline produced a geometrically incorrect equation.")

    print(f"\nSMOKE TEST PASSED")
    print(f"equation rmse={equation.rmse:.6f}, "
          f"zero level set mean_r={mean_r:.4f}, std={std_r:.4f}")

    return True, equation


def test_pipeline_bunny():
    """
    Full pipeline smoke test on a unit icosphere.

    Verifies:
        1. Pipeline runs to completion without error
        2. Returns a single Equation
        3. Equation is finite (no NaN/Inf)
        4. Zero level set of the equation lies close to radius 1.0
           -- the only geometric ground truth we have

    This is intentionally a coarse test. The visual quality check
    happens in the prototype ray tracer.
    """
    logging.basicConfig(level=logging.INFO)

    mesh = trimesh.load(r'C:\Users\yalam\Documents\sigil-compiler\assets\meshes\bunny-high.obj')
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale(1.0 / mesh.scale)    

    config = PipelineConfig(
        n_seeds            = 16,
        sr_degree          = 4,
        sr_refine_steps    = 500,
        max_gpr_points     = 300,
        gpr_resolution     = 20,
        pysr_niterations   = 25,
        pysr_populations   = 15,
        pysr_root_multiplier = 4,
        smin_k             = 0.1,
        merge_refine_steps = 500,
        merge_refine_lr    = 1e-3,
        n_workers          = 4,
    )

    print("\n=== pipeline smoke test: sphere ===\n")
    print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"Config: {config}\n")

    equation = compile_mesh(mesh, config)

    import sympy
    # Save equation to file
    x0, x1, x2 = sympy.symbols('x0 x1 x2')
    with open(r'C:\Users\yalam\Documents\sigil-compiler\outputs\bunny_equation.txt', 'w') as out:
        out.write(str(equation.sympy_expr) + '\n')
        out.write(str(sympy.diff(equation.sympy_expr, x0)) + '\n')
        out.write(str(sympy.diff(equation.sympy_expr, x1)) + '\n')
        out.write(str(sympy.diff(equation.sympy_expr, x2)) + '\n')    

    # 1. Got an equation back
    assert equation is not None, "compile_mesh returned None"

    from sigil.geometry.sr.base import Equation
    assert isinstance(equation, Equation), \
        f"compile_mesh should return Equation, got {type(equation)}"

    print(f"\nFinal equation: {equation.sympy_expr}")
    print(f"Final RMSE:     {equation.rmse:.6f}")

    # 2. Equation evaluates without error
    # Sample a grid over [-1.5, 1.5]^3
    lin   = np.linspace(-1.5, 1.5, 30)
    xx, yy, zz = np.meshgrid(lin, lin, lin, indexing='ij')
    X_grid = np.column_stack([
        xx.ravel(),
        yy.ravel(),
        zz.ravel(),
    ])   # (27000, 3)

    y_grid = equation(X_grid)   # (27000,)

    assert not np.any(np.isnan(y_grid)), \
        "NaN in equation output over evaluation grid"
    assert not np.any(np.isinf(y_grid)), \
        "Inf in equation output over evaluation grid"

    print(f"Grid evaluation: {len(X_grid)} points, no NaN/Inf")
    print(f"Value range: [{y_grid.min():.4f}, {y_grid.max():.4f}]")

    # 3. Zero level set lies near radius 1.0
    threshold = 0.1
    near_zero = np.abs(y_grid) < threshold
    n_near    = near_zero.sum()

    assert n_near > 0, \
        (f"No points near zero level set (threshold={threshold}). "
         f"Pipeline may have failed silently. "
         f"Value range: [{y_grid.min():.4f}, {y_grid.max():.4f}]")

    radii  = np.linalg.norm(X_grid[near_zero], axis=1)
    mean_r = radii.mean()
    std_r  = radii.std()

    print(f"Zero level set: {n_near} near-zero points")
    print(f"Mean radius:    {mean_r:.4f} (expect 1.0)")
    print(f"Std radius:     {std_r:.4f}")

    assert n_near > 100, \
        f"Too few points near zero level set: {n_near}"

    assert np.isfinite(radii).all(), \
        "Non-finite values in zero level set"

    print(f"\nSMOKE TEST PASSED")
    print(f"equation rmse={equation.rmse:.6f}, "
          f"zero level set mean_r={mean_r:.4f}, std={std_r:.4f}")

    return True, equation


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    # ("pipeline_sphere", test_pipeline_sphere),
    ("pipeline_bunny", test_pipeline_bunny),
]

def run_all():
    print("\n=== geometry_pipeline tests ===\n")
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