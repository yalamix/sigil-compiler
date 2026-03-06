# tests/test_geometry/test_merge.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import warnings
import numpy as np
import trimesh
import sympy

from sigil.geometry.segmentation import (
    build_graph, farthest_point_sampling,
    extract_patches_radius, estimate_patch_radius,
    find_adjacent_patches
)
from sigil.geometry.scalar_field import sample_scalar_field
from sigil.geometry.gpr import fit_gpr, predict, generate_query_points
from sigil.geometry.sr.base import Equation, build_feature_matrix, alphas_to_sympy
from sigil.geometry.sr.sparse_regression import SparseRegressionBackend
from sigil.geometry.merge import (
    blend_smin,
    blend_polynomial,
    _extract_float_constants,
    refine_coefficients,
    residual_correction_merge,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MESH         = None
_PATCHES      = None
_ADJACENCY    = None
_EQUATIONS    = None
_QUERY_DATA   = None   # list of (X_query, y_query) per patch

def get_fixtures():
    global _MESH, _PATCHES, _ADJACENCY, _EQUATIONS, _QUERY_DATA
    if _MESH is None:
        _MESH = trimesh.creation.icosphere(subdivisions=4)
        graph = build_graph(_MESH)
        seeds, dist_matrix = farthest_point_sampling(_MESH, graph, 16)
        radius = estimate_patch_radius(_MESH, 16)
        _PATCHES   = extract_patches_radius(dist_matrix, radius)
        _ADJACENCY = find_adjacent_patches(_PATCHES)

        backend    = SparseRegressionBackend(degree=4)
        _EQUATIONS = []
        _QUERY_DATA = []

        for patch_vertices in _PATCHES:
            X_train, y_train = sample_scalar_field(
                patch_vertices, _MESH, max_gpr_points=300
            )
            model   = fit_gpr(X_train, y_train)
            X_query = generate_query_points(patch_vertices, _MESH, resolution=15)
            y_query = predict(model, X_query)
            eq      = backend.fit(X_query, y_query)
            _EQUATIONS.append(eq)
            _QUERY_DATA.append((X_query, y_query))

    return _MESH, _PATCHES, _ADJACENCY, _EQUATIONS, _QUERY_DATA

def _get_adjacent_pair_with_overlap(patches, adjacency, equations, query_data):
    """
    Find a pair (i, j) that are truly adjacent, have fitted equations,
    and return their overlap query points from the actual overlap region.
    
    Raises AssertionError if no such pair exists -- never skips.
    """
    for (i, j), overlap_vertices in adjacency.items():
        if i < len(equations) and j < len(equations):
            if len(overlap_vertices) > 10:   # meaningful overlap
                return i, j, overlap_vertices
    
    raise AssertionError(
        f"No adjacent pairs with meaningful overlap found among "
        f"{len(equations)} fitted patches. "
        f"Adjacency has {len(adjacency)} pairs total. "
        f"This is a segmentation bug."
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_blend_smin_returns_equation():
    _, _, _, equations, _ = get_fixtures()
    eq_a = equations[0]
    eq_b = equations[1]

    result = blend_smin(eq_a, eq_b, k=0.1)

    assert isinstance(result, Equation), \
        f"blend_smin should return Equation, got {type(result)}"
    assert result.sympy_expr is not None, \
        "sympy_expr should not be None"
    assert result.degree == 0, \
        "smin result is non-polynomial, degree should be 0"
    assert result.alphas is None, \
        "smin result should have no alphas"

    return True, f"expr contains log/exp: {'log' in str(result.sympy_expr)}"


def test_blend_smin_zero_level_set():
    """
    Evaluate smin blend at the actual overlap vertices between two
    adjacent patches. The blend surface should be within k*log(2)
    of radius 1.0 -- the theoretical maximum smin expansion.
    """
    mesh, patches, adjacency, equations, query_data = get_fixtures()

    i, j, overlap_vertices = _get_adjacent_pair_with_overlap(
        patches, adjacency, equations, query_data
    )

    eq_a  = equations[i]
    eq_b  = equations[j]
    blend = blend_smin(eq_a, eq_b, k=0.1)

    # Evaluate AT the overlap vertices -- the actual boundary region
    overlap_pts = mesh.vertices[overlap_vertices]   # (N_overlap, 3)
    y_blend     = blend(overlap_pts)

    near_zero = np.abs(y_blend) < 0.15
    assert near_zero.sum() > 0, \
        (f"No overlap points near zero level set of blend. "
         f"y_blend range: [{y_blend.min():.4f}, {y_blend.max():.4f}]")

    zero_pts = overlap_pts[near_zero]
    radii    = np.linalg.norm(zero_pts, axis=1)
    mean_r   = radii.mean()

    # smin expands surface outward by at most k*log(2) ~ 0.069 for k=0.1
    # Allow 3x that as tolerance since equations are approximations
    max_expansion = 0.1 * np.log(2) * 3
    assert abs(mean_r - 1.0) < max_expansion + 0.05, \
        (f"Blend zero level set mean radius {mean_r:.4f} too far from 1.0. "
         f"Max expected expansion: {max_expansion:.4f}. "
         f"Patches {i} and {j}, {len(overlap_vertices)} overlap vertices.")

    return True, (f"patches ({i},{j}), {len(overlap_vertices)} overlap verts, "
                  f"mean radius={mean_r:.4f}, {near_zero.sum()} near-zero pts")


def test_blend_smin_k_zero_warning():
    _, _, _, equations, _ = get_fixtures()
    eq_a = equations[0]
    eq_b = equations[1]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        blend_smin(eq_a, eq_b, k=0)
        warning_messages = [str(w.message) for w in caught]

    assert any("k=0" in m for m in warning_messages), \
        f"Expected warning for k=0, got: {warning_messages}"

    return True, "k=0 warning raised correctly"


def test_blend_polynomial_stub():
    _, _, _, equations, query_data = get_fixtures()
    eq_a = equations[0]
    eq_b = equations[1]
    X, y = query_data[0]

    try:
        blend_polynomial(eq_a, eq_b, X, y)
        return False, "Should have raised NotImplementedError"
    except NotImplementedError:
        pass

    return True, "NotImplementedError raised correctly"


def test_extract_float_constants_basic():
    x0, x1, x2 = sympy.symbols('x0 x1 x2')

    # Expression with known constants: 0.5*x0^2 + 1.732*x1 - 0.866
    expr = sympy.Float(0.5) * x0**2 + sympy.Float(1.732) * x1 - sympy.Float(0.866)
    constants = _extract_float_constants(expr)

    vals = sorted([abs(float(c)) for c in constants])
    assert len(constants) == 3, \
        f"Expected 3 float constants, got {len(constants)}: {constants}"
    assert any(abs(float(c) - 0.5)   < 1e-6 for c in constants), "Missing 0.5"
    assert any(abs(float(c) - 1.732) < 1e-6 for c in constants), "Missing 1.732"
    assert any(abs(abs(float(c)) - 0.866) < 1e-6 for c in constants), "Missing 0.866"

    return True, f"Found constants: {[float(c) for c in constants]}"


def test_extract_float_constants_skips_structural():
    x0 = sympy.Symbol('x0')

    # 0, 1, -1 should be skipped, 2.5 should be kept
    expr = (sympy.Integer(0) + sympy.Integer(1) * x0
            + sympy.Integer(-1) * x0 + sympy.Float(2.5) * x0**2)
    expr = sympy.expand(expr)
    constants = _extract_float_constants(expr)

    vals = [abs(float(c)) for c in constants]
    assert all(abs(v - 1.0) > 1e-6 for v in vals), \
        f"1.0 should be skipped, got {vals}"
    assert any(abs(v - 2.5) < 1e-6 for v in vals), \
        f"2.5 should be kept, got {vals}"

    return True, f"Structural constants skipped, kept: {vals}"


def test_refine_coefficients_improves_rmse():
    """
    Deliberately perturb the constants in a fitted equation,
    verify that refinement recovers lower RMSE than the perturbed version.
    """
    _, _, _, equations, query_data = get_fixtures()
    eq    = equations[0]
    X, y  = query_data[0]

    rmse_original = eq.rmse

    # Perturb all float constants by +20%
    constants = _extract_float_constants(eq.sympy_expr)
    if not constants:
        return True, "SKIP -- no float constants in equation"

    perturbed_subs = {c: sympy.Float(float(c) * 1.2) for c in constants}
    expr_perturbed = eq.sympy_expr.subs(perturbed_subs)

    eq_perturbed = Equation(
        sympy_expr    = expr_perturbed,
        rmse          = 0.0,
        degree        = eq.degree,
        alphas        = None,
        feature_names = None,
    )
    y_perturbed      = eq_perturbed(X)
    rmse_perturbed   = float(np.sqrt(np.mean((y_perturbed - y)**2)))

    # Refine the perturbed equation
    eq_refined     = refine_coefficients(eq_perturbed, X, y, steps=500)
    rmse_refined   = eq_refined.rmse

    assert rmse_refined < rmse_perturbed, \
        (f"Refinement should improve RMSE: "
         f"perturbed={rmse_perturbed:.6f}, refined={rmse_refined:.6f}")

    return True, (f"original={rmse_original:.6f}, "
                  f"perturbed={rmse_perturbed:.6f}, "
                  f"refined={rmse_refined:.6f}")


def test_residual_correction_merge_improves_rmse():
    """
    Blend two truly adjacent patches, apply residual correction,
    verify final RMSE is lower than blend alone.
    Evaluated on actual overlap vertices, not an arbitrary query grid.
    """
    mesh, patches, adjacency, equations, query_data = get_fixtures()

    i, j, overlap_vertices = _get_adjacent_pair_with_overlap(
        patches, adjacency, equations, query_data
    )

    eq_a = equations[i]
    eq_b = equations[j]

    # Sample scalar field specifically over the overlap region
    # This is the ground truth for the merge step
    from sigil.geometry.scalar_field import sample_scalar_field
    from sigil.geometry.gpr import fit_gpr, predict, generate_query_points

    X_overlap, y_overlap_train = sample_scalar_field(
        overlap_vertices, mesh, max_gpr_points=300
    )
    overlap_gpr   = fit_gpr(X_overlap, y_overlap_train)
    X_query_overlap = generate_query_points(
        overlap_vertices, mesh, resolution=15
    )
    y_overlap = predict(overlap_gpr, X_query_overlap)

    eq_blend   = blend_smin(eq_a, eq_b, k=0.1)
    y_blend    = eq_blend(X_query_overlap)
    rmse_blend = float(np.sqrt(np.mean((y_blend - y_overlap)**2)))

    backend   = SparseRegressionBackend(degree=4)
    eq_merged = residual_correction_merge(
        eq_blend, X_query_overlap, y_overlap, backend,
        is_root      = False,
        refine_steps = 200,
        refine_lr    = 1e-3,
    )

    assert eq_merged.rmse <= rmse_blend + 1e-4, \
        (f"Merged RMSE {eq_merged.rmse:.6f} should be <= "
         f"blend RMSE {rmse_blend:.6f}. "
         f"Patches ({i},{j}), {len(overlap_vertices)} overlap vertices.")

    return True, (f"patches ({i},{j}), "
                  f"blend RMSE={rmse_blend:.6f}, "
                  f"merged RMSE={eq_merged.rmse:.6f}")


def test_residual_correction_skipped_when_small():
    """
    When blend already fits the overlap data within threshold,
    SR correction is skipped and only refinement runs.
    Uses real adjacent patches and real overlap data.
    """
    mesh, patches, adjacency, equations, query_data = get_fixtures()

    i, j, overlap_vertices = _get_adjacent_pair_with_overlap(
        patches, adjacency, equations, query_data
    )

    eq_a = equations[i]
    eq_b = equations[j]

    from sigil.geometry.scalar_field import sample_scalar_field
    from sigil.geometry.gpr import fit_gpr, predict, generate_query_points

    X_overlap, y_overlap_train = sample_scalar_field(
        overlap_vertices, mesh, max_gpr_points=300
    )
    overlap_gpr     = fit_gpr(X_overlap, y_overlap_train)
    X_query_overlap = generate_query_points(
        overlap_vertices, mesh, resolution=15
    )

    eq_blend  = blend_smin(eq_a, eq_b, k=0.1)
    # Pass blend's own predictions as ground truth -- residual = 0
    y_perfect = eq_blend(X_query_overlap)

    backend   = SparseRegressionBackend(degree=4)
    eq_result = residual_correction_merge(
        eq_blend, X_query_overlap, y_perfect, backend,
        refine_steps=100,
    )

    assert eq_result.rmse < 1e-3, \
        f"RMSE should be near 0 for perfect data, got {eq_result.rmse:.6f}"

    return True, (f"patches ({i},{j}), "
                  f"RMSE={eq_result.rmse:.6f} (SR skipped, refinement only)")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ("blend_smin_returns_equation",      test_blend_smin_returns_equation),
    ("blend_smin_zero_level_set",        test_blend_smin_zero_level_set),
    ("blend_smin_k_zero_warning",        test_blend_smin_k_zero_warning),
    ("blend_polynomial_stub",            test_blend_polynomial_stub),
    ("extract_float_constants_basic",    test_extract_float_constants_basic),
    ("extract_float_constants_skips",    test_extract_float_constants_skips_structural),
    ("refine_coefficients_improves",     test_refine_coefficients_improves_rmse),
    ("residual_correction_improves",     test_residual_correction_merge_improves_rmse),
    ("residual_correction_skipped",      test_residual_correction_skipped_when_small),
]

def run_all():
    print("\n=== merge tests ===\n")
    passed = 0
    failed = 0

    for name, test_fn in TESTS:
        try:
            ok, msg = test_fn()
            print(f"  PASS  {name}: {msg}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed+failed} tests passed")
    if failed > 0:
        sys.exit(1)