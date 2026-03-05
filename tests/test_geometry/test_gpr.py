# tests/test_geometry/test_gpr.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import trimesh
from sigil.geometry.segmentation import (
    build_graph, farthest_point_sampling,
    extract_patches_radius, estimate_patch_radius
)
from sigil.geometry.scalar_field import sample_scalar_field
from sigil.geometry.gpr import (
    make_kernel,
    fit_gpr,
    predict,
    generate_query_points,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MESH = None
_PATCH_VERTICES = None
_X = None
_Y = None

def get_fixtures():
    global _MESH, _PATCH_VERTICES, _X, _Y
    if _MESH is None:
        _MESH = trimesh.creation.icosphere(subdivisions=4)
        graph = build_graph(_MESH)
        seeds, dist_matrix = farthest_point_sampling(_MESH, graph, 16)
        radius = estimate_patch_radius(_MESH, 16)
        patches = extract_patches_radius(dist_matrix, radius)
        _PATCH_VERTICES = patches[0]
        _X, _Y = sample_scalar_field(_PATCH_VERTICES, _MESH,
                                     max_gpr_points=300)
    return _MESH, _PATCH_VERTICES, _X, _Y


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_make_kernel():
    from sklearn.gaussian_process.kernels import Sum, Product

    kernel = make_kernel(length_scale_init=0.5)

    # Should be a valid sklearn kernel -- has a callable interface
    assert callable(kernel), "Kernel should be callable"

    # Kernel evaluated on identical points should return positive values
    X_test = np.array([[0.0, 0.0, 0.0],
                        [0.1, 0.0, 0.0],
                        [0.0, 0.1, 0.0]])
    K = kernel(X_test)                  # (3, 3) kernel matrix
    assert K.shape == (3, 3), \
        f"Kernel matrix shape {K.shape} != (3, 3)"
    assert np.all(np.diag(K) > 0), \
        f"Diagonal entries should be positive: {np.diag(K)}"

    # Kernel matrix should be symmetric
    assert np.allclose(K, K.T, atol=1e-10), \
        "Kernel matrix should be symmetric"

    return True, f"Kernel valid, K diagonal={np.diag(K).round(4)}"


def test_fit_gpr():
    _, _, X, y = get_fixtures()

    model = fit_gpr(X, y)

    # sklearn GPR exposes kernel_ (fitted) and kernel (prior)
    assert hasattr(model, 'kernel_'), \
        "Fitted model should have kernel_ attribute"
    assert hasattr(model, 'X_train_'), \
        "Fitted model should have X_train_ attribute"
    assert model.X_train_.shape == X.shape, \
        f"Stored X_train_ shape {model.X_train_.shape} != {X.shape}"

    # Log marginal likelihood should be finite
    lml = model.log_marginal_likelihood_value_
    assert np.isfinite(lml), \
        f"Log marginal likelihood should be finite, got {lml}"

    return True, f"GPR fitted, log_marginal_likelihood={lml:.4f}"


def test_fit_gpr_unknown_backend():
    _, _, X, y = get_fixtures()

    try:
        fit_gpr(X, y, backend='invalid')
        return False, "Should have raised ValueError"
    except ValueError:
        pass

    try:
        fit_gpr(X, y, backend='gpytorch')
        return False, "Should have raised NotImplementedError"
    except NotImplementedError:
        pass

    return True, "ValueError and NotImplementedError raised correctly"


def test_predict_shape():
    _, patch_vertices, X, y = get_fixtures()
    mesh, _, _, _ = get_fixtures()

    model = fit_gpr(X, y)
    query_points = generate_query_points(patch_vertices, mesh, resolution=10)
    y_pred = predict(model, query_points)

    assert y_pred.shape == (len(query_points),), \
        f"Prediction shape {y_pred.shape} != ({len(query_points)},)"
    assert not np.any(np.isnan(y_pred)), "NaN in predictions"
    assert not np.any(np.isinf(y_pred)), "Inf in predictions"

    return True, f"Predictions shape {y_pred.shape}, no NaN/Inf"


def test_predict_zero_crossing():
    """
    The key geometric test: fit GPR on a sphere patch, evaluate on a grid,
    check that the zero level set lies close to the unit sphere surface.

    We find grid points where |f(x,y,z)| < threshold and check their
    radii are close to 1.0. This is the first end-to-end geometric
    check in the pipeline.
    """
    mesh, patch_vertices, X, y = get_fixtures()

    model = fit_gpr(X, y)

    # Higher resolution for this test -- we need enough points near
    # the zero crossing to get a meaningful sample
    query_points = generate_query_points(patch_vertices, mesh, resolution=25)
    y_pred = predict(model, query_points)

    # Find points close to the zero level set
    threshold = 0.01
    near_zero = np.abs(y_pred) < threshold      # (M,) bool
    n_near_zero = near_zero.sum()

    assert n_near_zero > 0, \
        f"No points near zero level set (threshold={threshold}). " \
        f"GPR may not have fitted correctly. " \
        f"y_pred range: [{y_pred.min():.4f}, {y_pred.max():.4f}]"

    # Check radii of near-zero points -- should be close to 1.0
    zero_points = query_points[near_zero]       # (n_near_zero, 3)
    radii = np.linalg.norm(zero_points, axis=1) # (n_near_zero,)

    mean_r = radii.mean()
    std_r = radii.std()

    assert 0.92 < mean_r < 1.08, \
        f"Zero level set mean radius {mean_r:.4f} too far from 1.0"
    assert std_r < 0.08, \
        f"Zero level set radius std {std_r:.4f} too high -- surface is noisy"

    return True, (f"{n_near_zero} near-zero points, "
                  f"mean radius={mean_r:.4f} (expect 1.0), "
                  f"std={std_r:.4f}")


def test_generate_query_points():
    mesh, patch_vertices, _, _ = get_fixtures()

    resolution = 15
    query_points = generate_query_points(patch_vertices, mesh, resolution)

    expected_n = resolution ** 3
    assert query_points.shape == (expected_n, 3), \
        f"Query points shape {query_points.shape} != ({expected_n}, 3)"

    # Grid should cover patch bounding box (with margin)
    positions = mesh.vertices[patch_vertices]
    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)

    # Query points should extend beyond the patch bbox due to margin
    assert query_points[:, 0].min() < bbox_min[0], \
        "Query grid should extend beyond patch bbox (x min)"
    assert query_points[:, 0].max() > bbox_max[0], \
        "Query grid should extend beyond patch bbox (x max)"

    return True, (f"shape {query_points.shape}, "
                  f"grid extends beyond bbox as expected")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ("make_kernel",             test_make_kernel),
    ("fit_gpr",                 test_fit_gpr),
    ("fit_gpr_unknown_backend", test_fit_gpr_unknown_backend),
    ("predict_shape",           test_predict_shape),
    ("predict_zero_crossing",   test_predict_zero_crossing),
    ("generate_query_points",   test_generate_query_points),
]

def run_all():
    print("\n=== gpr tests ===\n")
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