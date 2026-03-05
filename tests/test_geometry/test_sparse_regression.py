# tests/test_geometry/test_sparse_regression.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import trimesh
import sympy
from sigil.geometry.segmentation import (
    build_graph, farthest_point_sampling,
    extract_patches_radius, estimate_patch_radius
)
from sigil.geometry.scalar_field import sample_scalar_field
from sigil.geometry.gpr import fit_gpr, predict, generate_query_points
from sigil.geometry.sr.base import (
    Equation, build_feature_matrix,
    alphas_to_sympy, sympy_to_alphas
)
from sigil.geometry.sr.sparse_regression import (
    SparseRegressionBackend,
    _lasso_fit,
    _refine_torch,
    _compute_rmse,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MESH       = None
_X_TRAIN    = None
_Y_TRAIN    = None
_X_QUERY    = None
_Y_QUERY    = None
_PATCH_VERTS = None

def get_fixtures():
    global _MESH, _X_TRAIN, _Y_TRAIN, _X_QUERY, _Y_QUERY, _PATCH_VERTS
    if _MESH is None:
        _MESH = trimesh.creation.icosphere(subdivisions=4)
        graph = build_graph(_MESH)
        seeds, dist_matrix = farthest_point_sampling(_MESH, graph, 16)
        radius = estimate_patch_radius(_MESH, 16)
        patches = extract_patches_radius(dist_matrix, radius)
        _PATCH_VERTS = patches[0]

        # GPR training data
        _X_TRAIN, _Y_TRAIN = sample_scalar_field(_PATCH_VERTS, _MESH,
                                                  max_gpr_points=300)
        # Query points for SR
        model = fit_gpr(_X_TRAIN, _Y_TRAIN)
        _X_QUERY = generate_query_points(_PATCH_VERTS, _MESH, resolution=20)
        _Y_QUERY = predict(model, _X_QUERY)

    return _MESH, _PATCH_VERTS, _X_TRAIN, _Y_TRAIN, _X_QUERY, _Y_QUERY


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_build_feature_matrix():
    _, _, _, _, X_query, _ = get_fixtures()

    degree = 2
    Phi, feature_names = build_feature_matrix(X_query, degree)

    # degree 2 in 3 variables: C(3+2, 2) = 10 features
    expected_n_features = 10
    assert Phi.shape == (len(X_query), expected_n_features), \
        f"Phi shape {Phi.shape} != ({len(X_query)}, {expected_n_features})"
    assert len(feature_names) == expected_n_features, \
        f"feature_names length {len(feature_names)} != {expected_n_features}"

    # First feature should be the bias term '1'
    assert feature_names[0] == '1', \
        f"First feature should be '1', got '{feature_names[0]}'"

    # Bias column should be all ones
    assert np.allclose(Phi[:, 0], 1.0), \
        "Bias column (feature '1') should be all ones"

    # degree 4: C(3+4, 4) = 35 features
    Phi4, names4 = build_feature_matrix(X_query, degree=4)
    assert Phi4.shape[1] == 35, \
        f"degree=4 should give 35 features, got {Phi4.shape[1]}"

    return True, (f"degree=2: {Phi.shape}, features={feature_names[:5]}..., "
                  f"degree=4: {Phi4.shape[1]} features")


def test_alphas_to_sympy():
    # Construct known alphas for x0^2 + x1^2 + x2^2 (unit sphere)
    _, names = build_feature_matrix(np.zeros((1, 3)), degree=2)

    alphas = np.zeros(len(names))
    alphas[names.index('x0^2')] = 1.0
    alphas[names.index('x1^2')] = 1.0
    alphas[names.index('x2^2')] = 1.0

    expr = alphas_to_sympy(alphas, names)

    x0, x1, x2 = sympy.symbols('x0 x1 x2')
    expected = x0**2 + x1**2 + x2**2

    assert sympy.simplify(expr - expected) == 0, \
        f"Expected x0^2+x1^2+x2^2, got {expr}"

    return True, f"expr={expr}"


def test_sympy_to_alphas_roundtrip():
    # Build a known expression, convert to alphas, convert back
    _, names = build_feature_matrix(np.zeros((1, 3)), degree=2)

    alphas_orig = np.zeros(len(names))
    alphas_orig[names.index('x0^2')] = 1.0
    alphas_orig[names.index('x1^2')] = 1.0
    alphas_orig[names.index('x2^2')] = 1.0
    alphas_orig[names.index('1')]     = -1.0   # x0^2 + x1^2 + x2^2 - 1 = 0

    expr = alphas_to_sympy(alphas_orig, names)
    alphas_recovered = sympy_to_alphas(expr, names)

    assert alphas_recovered is not None, \
        "sympy_to_alphas returned None for a polynomial expression"
    assert np.allclose(alphas_orig, alphas_recovered, atol=1e-6), \
        f"Roundtrip failed:\n  orig={alphas_orig}\n  recovered={alphas_recovered}"

    return True, "Roundtrip alphas -> sympy -> alphas exact to 1e-6"


def test_sympy_to_alphas_nonpolynomial():
    # Non-polynomial expressions should return None
    x0, x1, x2 = sympy.symbols('x0 x1 x2')
    expr = sympy.sin(x0**2 + x1**2 + x2**2) - 1
    _, names = build_feature_matrix(np.zeros((1, 3)), degree=2)

    result = sympy_to_alphas(expr, names)
    assert result is None, \
        f"sympy_to_alphas should return None for non-polynomial, got {result}"

    return True, "Non-polynomial correctly returns None"


def test_lasso_fit():
    _, _, _, _, X_query, Y_query = get_fixtures()

    Phi, names = build_feature_matrix(X_query, degree=4)
    alphas = _lasso_fit(Phi, Y_query)

    assert alphas.shape == (len(names),), \
        f"alphas shape {alphas.shape} != ({len(names)},)"

    n_nonzero = np.sum(alphas != 0)
    assert n_nonzero > 0, "All alphas are zero -- Lasso over-regularized"
    assert n_nonzero < len(names), \
        f"No sparsity: all {len(names)} features nonzero"

    return True, f"alphas shape {alphas.shape}, nonzero={n_nonzero}/{len(names)}"


def test_lasso_fit_warm_start():
    _, _, _, _, X_query, Y_query = get_fixtures()

    Phi, names = build_feature_matrix(X_query, degree=4)

    # Cold start
    alphas_cold = _lasso_fit(Phi, Y_query)

    # Warm start from cold solution -- should produce similar result
    alphas_warm = _lasso_fit(Phi, Y_query, initial_alphas=alphas_cold)

    assert alphas_warm.shape == alphas_cold.shape, \
        "Warm start alphas shape mismatch"

    # Both should have low RMSE -- warm start shouldn't degrade quality
    rmse_cold = _compute_rmse(Phi, Y_query, alphas_cold)
    rmse_warm = _compute_rmse(Phi, Y_query, alphas_warm)
    assert rmse_warm < 0.1, \
        f"Warm start RMSE {rmse_warm:.6f} too high"

    return True, (f"cold rmse={rmse_cold:.6f}, "
                  f"warm rmse={rmse_warm:.6f}, "
                  f"nonzero cold={np.sum(alphas_cold!=0)}, "
                  f"warm={np.sum(alphas_warm!=0)}")


def test_refine_torch():
    _, _, _, _, X_query, Y_query = get_fixtures()

    Phi, names = build_feature_matrix(X_query, degree=4)
    alphas_sparse = _lasso_fit(Phi, Y_query)

    rmse_before = _compute_rmse(Phi, Y_query, alphas_sparse)
    alphas_refined = _refine_torch(Phi, Y_query, alphas_sparse, n_steps=500)
    rmse_after = _compute_rmse(Phi, Y_query, alphas_refined)

    # Sparsity pattern must be preserved
    assert np.all((alphas_refined != 0) == (alphas_sparse != 0)), \
        "Refinement changed sparsity pattern -- zero terms became nonzero"

    # Refinement should not make things worse
    assert rmse_after <= rmse_before + 1e-6, \
        f"Refinement degraded RMSE: {rmse_before:.6f} -> {rmse_after:.6f}"

    return True, (f"RMSE before={rmse_before:.6f}, "
                  f"after={rmse_after:.6f}, "
                  f"nonzero={np.sum(alphas_refined!=0)}")


def test_sparse_regression_fit():
    _, patch_verts, _, _, X_query, Y_query = get_fixtures()

    backend = SparseRegressionBackend(degree=4)
    eq = backend.fit(X_query, Y_query)

    assert isinstance(eq, Equation), \
        f"fit() should return Equation, got {type(eq)}"
    assert eq.sympy_expr is not None, \
        "sympy_expr should not be None"
    assert np.isfinite(eq.rmse), \
        f"rmse should be finite, got {eq.rmse}"
    assert eq.alphas is not None, \
        "alphas should not be None for sparse regression backend"
    assert eq.feature_names is not None, \
        "feature_names should not be None for sparse regression backend"
    assert eq.additive_terms is not None, \
        "additive_terms should be populated by __post_init__"

    return True, (f"expr={eq.sympy_expr}, "
                  f"rmse={eq.rmse:.6f}, "
                  f"nonzero={np.sum(eq.alphas!=0)}")


def test_equation_call():
    _, _, _, _, X_query, Y_query = get_fixtures()

    backend = SparseRegressionBackend(degree=4)
    eq = backend.fit(X_query, Y_query)

    # Evaluate equation at query points
    y_pred = eq(X_query)

    assert y_pred.shape == (len(X_query),), \
        f"eq(X) shape {y_pred.shape} != ({len(X_query)},)"
    assert not np.any(np.isnan(y_pred)), "NaN in equation output"
    assert not np.any(np.isinf(y_pred)), "Inf in equation output"

    return True, f"eq(X_query) shape {y_pred.shape}, no NaN/Inf"


def test_equation_gradient():
    _, _, _, _, X_query, Y_query = get_fixtures()

    backend = SparseRegressionBackend(degree=4)
    eq = backend.fit(X_query, Y_query)

    grad = eq.gradient(X_query)

    assert grad.shape == (len(X_query), 3), \
        f"gradient shape {grad.shape} != ({len(X_query)}, 3)"
    assert not np.any(np.isnan(grad)), "NaN in gradient"

    # Normals should be unit length
    normals = eq.normal(X_query)
    norms = np.linalg.norm(normals, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), \
        f"Normals not unit length: min={norms.min():.6f}, max={norms.max():.6f}"

    return True, f"gradient shape {grad.shape}, normals unit length"


def test_zero_level_set_geometry():
    """
    The key geometric test for SR: fit an equation to a sphere patch,
    find the zero level set, verify it lies at radius ~1.0.

    This catches the critical failure mode: SR randomly initialized on
    the full sample space would produce an equation whose zero level set
    bears no relationship to the actual surface geometry.
    """
    _, patch_verts, _, _, X_query, Y_query = get_fixtures()

    backend = SparseRegressionBackend(degree=4)
    eq = backend.fit(X_query, Y_query)

    y_pred = eq(X_query)

    # Find points close to zero level set
    threshold = 0.05
    near_zero = np.abs(y_pred) < threshold
    n_near_zero = near_zero.sum()

    assert n_near_zero > 0, \
        (f"No points near zero level set (threshold={threshold}). "
         f"SR may have failed. y_pred range: "
         f"[{y_pred.min():.4f}, {y_pred.max():.4f}]")

    zero_points = X_query[near_zero]
    radii = np.linalg.norm(zero_points, axis=1)
    mean_r = radii.mean()
    std_r  = radii.std()

    assert 0.90 < mean_r < 1.10, \
        f"Zero level set mean radius {mean_r:.4f} too far from 1.0"
    assert std_r < 0.10, \
        f"Zero level set radius std {std_r:.4f} too high -- SR surface is noisy"

    return True, (f"{n_near_zero} near-zero points, "
                  f"mean radius={mean_r:.4f} (expect 1.0), "
                  f"std={std_r:.4f}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ("build_feature_matrix",         test_build_feature_matrix),
    ("alphas_to_sympy",              test_alphas_to_sympy),
    ("sympy_to_alphas_roundtrip",    test_sympy_to_alphas_roundtrip),
    ("sympy_to_alphas_nonpolynomial",test_sympy_to_alphas_nonpolynomial),
    ("lasso_fit",                    test_lasso_fit),
    ("lasso_fit_warm_start",         test_lasso_fit_warm_start),
    ("refine_torch",                 test_refine_torch),
    ("sparse_regression_fit",        test_sparse_regression_fit),
    ("equation_call",                test_equation_call),
    ("equation_gradient",            test_equation_gradient),
    ("zero_level_set_geometry",      test_zero_level_set_geometry),
]

def run_all():
    print("\n=== sparse_regression tests ===\n")
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