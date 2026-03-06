# tests/test_geometry/test_pysr_backend.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
import trimesh
import sympy

from sigil.geometry.segmentation import (
    build_graph, farthest_point_sampling,
    extract_patches_radius, estimate_patch_radius
)
from sigil.geometry.scalar_field import sample_scalar_field
from sigil.geometry.gpr import fit_gpr, predict, generate_query_points
from sigil.geometry.sr.base import Equation

try:
    from sigil.geometry.sr.pysr_backend import PySRBackend
    from pysr import PySRRegressor
    PYSR_AVAILABLE = True
except (ImportError, Exception):
    PYSR_AVAILABLE = False

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MESH        = None
_PATCH_VERTS = None
_X_QUERY     = None
_Y_QUERY     = None

def get_fixtures():
    global _MESH, _PATCH_VERTS, _X_QUERY, _Y_QUERY
    if _MESH is None:
        _MESH = trimesh.creation.icosphere(subdivisions=4)
        graph = build_graph(_MESH)
        seeds, dist_matrix = farthest_point_sampling(_MESH, graph, 16)
        radius = estimate_patch_radius(_MESH, 16)
        patches = extract_patches_radius(dist_matrix, radius)
        _PATCH_VERTS = patches[0]

        X_train, y_train = sample_scalar_field(_PATCH_VERTS, _MESH,
                                                max_gpr_points=300)
        model    = fit_gpr(X_train, y_train)
        _X_QUERY = generate_query_points(_PATCH_VERTS, _MESH, resolution=20)
        _Y_QUERY = predict(model, _X_QUERY)

    return _MESH, _PATCH_VERTS, _X_QUERY, _Y_QUERY


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pysr_available():
    if not PYSR_AVAILABLE:
        return True, "SKIP -- PySR/Julia not available"
    return True, "PySR available"


def test_pysr_fit_returns_equation():
    if not PYSR_AVAILABLE:
        return True, "SKIP -- PySR/Julia not available"

    _, _, X_query, y_query = get_fixtures()
    backend = PySRBackend(niterations=25, random_state=42)
    eq = backend.fit(X_query, y_query)

    assert isinstance(eq, Equation), \
        f"fit() should return Equation, got {type(eq)}"
    assert eq.sympy_expr is not None, \
        "sympy_expr should not be None"
    assert np.isfinite(eq.rmse), \
        f"rmse should be finite, got {eq.rmse}"
    assert eq.alphas is None, \
        "PySR backend should not produce alphas"
    assert eq.feature_names is None, \
        "PySR backend should not produce feature_names"

    return True, f"expr={eq.sympy_expr}, rmse={eq.rmse:.6f}"


def test_pysr_equation_call():
    if not PYSR_AVAILABLE:
        return True, "SKIP -- PySR/Julia not available"

    _, _, X_query, y_query = get_fixtures()
    backend = PySRBackend(niterations=25, random_state=42)
    eq = backend.fit(X_query, y_query)

    y_pred = eq(X_query)

    assert y_pred.shape == (len(X_query),), \
        f"eq(X) shape {y_pred.shape} != ({len(X_query)},)"
    assert not np.any(np.isnan(y_pred)), "NaN in equation output"
    assert not np.any(np.isinf(y_pred)), "Inf in equation output"

    return True, f"eq(X_query) shape {y_pred.shape}, no NaN/Inf"


def test_pysr_equation_gradient():
    if not PYSR_AVAILABLE:
        return True, "SKIP -- PySR/Julia not available"

    _, _, X_query, y_query = get_fixtures()
    backend = PySRBackend(niterations=25, random_state=42)
    eq = backend.fit(X_query, y_query)

    grad    = eq.gradient(X_query)
    normals = eq.normal(X_query)

    assert grad.shape == (len(X_query), 3), \
        f"gradient shape {grad.shape} != ({len(X_query)}, 3)"
    assert not np.any(np.isnan(grad)), "NaN in gradient"

    norms = np.linalg.norm(normals, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), \
        f"Normals not unit length: min={norms.min():.6f}"

    return True, f"gradient {grad.shape}, normals unit length"


def test_pysr_world_coordinates():
    """
    Verify that the returned equation is in world coordinates,
    not normalized coordinates. Evaluate at a known surface point
    (a mesh vertex) -- f should be close to 0.
    """
    if not PYSR_AVAILABLE:
        return True, "SKIP -- PySR/Julia not available"

    mesh, patch_verts, X_query, y_query = get_fixtures()
    backend = PySRBackend(niterations=25, random_state=42)
    eq = backend.fit(X_query, y_query)

    # Sample a few patch vertices -- these are on the surface
    surface_pts = mesh.vertices[patch_verts[:10]]   # (10, 3)
    f_vals = eq(surface_pts)                         # (10,)

    # On the surface, f should be close to 0
    # Tolerance is generous -- PySR with 25 iterations won't be perfect
    assert np.abs(f_vals).mean() < 0.2, \
        (f"Equation evaluates far from 0 at surface vertices -- "
         f"may be in wrong coordinate frame. "
         f"Mean |f| = {np.abs(f_vals).mean():.4f}")

    return True, f"Mean |f| at surface vertices = {np.abs(f_vals).mean():.4f}"


def test_pysr_zero_level_set_geometry():
    """
    Geometric validation: zero level set should lie near sphere surface.
    Looser tolerance than sparse regression -- 25 iterations is limited.
    """
    if not PYSR_AVAILABLE:
        return True, "SKIP -- PySR/Julia not available"

    _, patch_verts, X_query, y_query = get_fixtures()
    backend = PySRBackend(niterations=25, random_state=42)
    eq = backend.fit(X_query, y_query)

    y_pred    = eq(X_query)
    threshold = 0.1                          # looser than sparse regression
    near_zero = np.abs(y_pred) < threshold
    n_near    = near_zero.sum()

    assert n_near > 0, \
        (f"No points near zero level set (threshold={threshold}). "
         f"y_pred range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")

    radii  = np.linalg.norm(X_query[near_zero], axis=1)
    mean_r = radii.mean()
    std_r  = radii.std()

    assert 0.85 < mean_r < 1.15, \
        f"Zero level set mean radius {mean_r:.4f} too far from 1.0"

    return True, (f"{n_near} near-zero points, "
                  f"mean radius={mean_r:.4f}, std={std_r:.4f}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ("pysr_available",            test_pysr_available),
    ("pysr_fit_returns_equation", test_pysr_fit_returns_equation),
    ("pysr_equation_call",        test_pysr_equation_call),
    ("pysr_equation_gradient",    test_pysr_equation_gradient),
    ("pysr_world_coordinates",    test_pysr_world_coordinates),
    ("pysr_zero_level_set",       test_pysr_zero_level_set_geometry),
]

def run_all():
    print("\n=== pysr_backend tests ===\n")
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