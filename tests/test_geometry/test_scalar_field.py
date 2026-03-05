# tests/test_geometry/test_scalar_field.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import trimesh
from sigil.geometry.segmentation import build_graph, farthest_point_sampling, \
    extract_patches_radius, estimate_patch_radius
from sigil.geometry.scalar_field import (
    estimate_epsilon,
    get_sample_count,
    sample_surface,
    build_training_data,
    subsample,
    sample_scalar_field,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MESH = None
_PATCH_VERTICES = None
_PATCH_FACES = None

def get_fixtures():
    global _MESH, _PATCH_VERTICES, _PATCH_FACES
    if _MESH is None:
        from sigil.geometry.segmentation import get_patch_faces
        _MESH = trimesh.creation.icosphere(subdivisions=4)
        graph = build_graph(_MESH)
        seeds, dist_matrix = farthest_point_sampling(_MESH, graph, 16)
        radius = estimate_patch_radius(_MESH, 16)
        patches = extract_patches_radius(dist_matrix, radius)
        _PATCH_VERTICES = patches[0]
        _PATCH_FACES = get_patch_faces(_MESH, _PATCH_VERTICES)
    return _MESH, _PATCH_VERTICES, _PATCH_FACES


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_estimate_epsilon():
    mesh, patch_vertices, _ = get_fixtures()

    eps = estimate_epsilon(patch_vertices, mesh)

    assert isinstance(eps, float), \
        f"epsilon should be float, got {type(eps)}"
    assert eps > 0, \
        f"epsilon should be positive, got {eps}"

    # A larger patch (all vertices) should give larger epsilon
    all_vertices = np.arange(len(mesh.vertices))
    eps_full = estimate_epsilon(all_vertices, mesh)
    assert eps_full > eps, \
        f"Full mesh epsilon {eps_full} should be > patch epsilon {eps}"

    return True, f"epsilon={eps:.6f}, full mesh epsilon={eps_full:.6f}"


def test_get_sample_count():
    mesh, patch_vertices, patch_faces = get_fixtures()

    n = get_sample_count(patch_faces, mesh)

    assert isinstance(n, int), \
        f"sample count should be int, got {type(n)}"
    assert 30 <= n <= 600, \
        f"sample count {n} outside expected range [30, 600]"

    # More faces = more samples (use full mesh faces as larger reference)
    all_faces = np.arange(len(mesh.faces))
    n_full = get_sample_count(all_faces, mesh)
    assert n_full >= n, \
        f"Full mesh sample count {n_full} should be >= patch count {n}"

    return True, f"patch n={n}, full mesh n={n_full}"


def test_sample_surface():
    mesh, patch_vertices, patch_faces = get_fixtures()
    n_samples = 100

    points, normals = sample_surface(patch_faces, mesh, n_samples)

    assert points.shape == (n_samples, 3), \
        f"points shape {points.shape} != ({n_samples}, 3)"
    assert normals.shape == (n_samples, 3), \
        f"normals shape {normals.shape} != ({n_samples}, 3)"

    # Normals must be unit length
    norms = np.linalg.norm(normals, axis=1)   # (n_samples,)
    assert np.allclose(norms, 1.0, atol=1e-5), \
        f"Normals not unit length: min={norms.min():.6f}, max={norms.max():.6f}"

    # Points should lie close to mesh surface
    # On a unit icosphere, all vertices are at radius 1.
    # Face-interior points are slightly inside, so radius < 1.
    # All should be within 1% of radius 1.
    radii = np.linalg.norm(points, axis=1)    # (n_samples,)
    assert radii.min() > 0.98, \
        f"Points too far from surface: min radius={radii.min():.6f}"
    assert radii.max() <= 1.01, \
        f"Points outside sphere: max radius={radii.max():.6f}"

    return True, f"shapes correct, normals unit, radii in [{radii.min():.4f}, {radii.max():.4f}]"


def test_build_training_data():
    mesh, patch_vertices, patch_faces = get_fixtures()

    points, normals = sample_surface(patch_faces, mesh, 100)
    epsilon = estimate_epsilon(patch_vertices, mesh)

    X, y = build_training_data(points, normals, epsilon,
                               patch_vertices, mesh,
                               include_vertices=False)

    assert X.shape[1] == 3, \
        f"X should have 3 columns, got {X.shape[1]}"
    assert len(X) == len(y), \
        f"X and y length mismatch: {len(X)} vs {len(y)}"

    # Exactly three distinct label values
    unique_labels = np.unique(np.round(y, decimals=10))
    assert len(unique_labels) == 3, \
        f"Expected 3 label values (0, +eps, -eps), got {len(unique_labels)}: {unique_labels}"

    # On a convex mesh (sphere), outside points should be further from
    # origin than inside points
    idx_outside = np.where(y > 0)[0]
    idx_inside  = np.where(y < 0)[0]
    r_outside = np.linalg.norm(X[idx_outside], axis=1).mean()
    r_inside  = np.linalg.norm(X[idx_inside],  axis=1).mean()
    assert r_outside > r_inside, \
        f"Outside points (r={r_outside:.4f}) should be further from origin " \
        f"than inside points (r={r_inside:.4f})"

    # Test with include_vertices=True
    X_v, y_v = build_training_data(points, normals, epsilon,
                                   patch_vertices, mesh,
                                   include_vertices=True)
    assert len(X_v) > len(X), \
        f"include_vertices=True should produce more points"

    return True, f"X shape {X.shape}, 3 labels, r_outside={r_outside:.4f} > r_inside={r_inside:.4f}"


def test_subsample():
    mesh, patch_vertices, patch_faces = get_fixtures()

    points, normals = sample_surface(patch_faces, mesh, 200)
    epsilon = estimate_epsilon(patch_vertices, mesh)
    X, y = build_training_data(points, normals, epsilon,
                               patch_vertices, mesh,
                               include_vertices=False)

    max_gpr_points = 90  # divisible by 3 for clean equal split
    X_sub, y_sub = subsample(X, y, epsilon, max_gpr_points)

    assert len(X_sub) <= max_gpr_points, \
        f"Subsampled {len(X_sub)} points, expected <= {max_gpr_points}"
    assert len(X_sub) == len(y_sub), \
        f"X_sub and y_sub length mismatch"

    # All three categories must be present
    tol = epsilon * 0.01
    has_on      = np.any(np.abs(y_sub) < tol)
    has_outside = np.any(y_sub > epsilon - tol)
    has_inside  = np.any(y_sub < -epsilon + tol)
    assert has_on,      "No on-surface points in subsample"
    assert has_outside, "No outside points in subsample"
    assert has_inside,  "No inside points in subsample"

    # Roughly equal split
    n_on      = np.sum(np.abs(y_sub) < tol)
    n_outside = np.sum(y_sub > epsilon - tol)
    n_inside  = np.sum(y_sub < -epsilon + tol)
    assert n_on == n_outside == n_inside, \
        f"Unequal split: on={n_on}, outside={n_outside}, inside={n_inside}"

    return True, f"{len(X_sub)} points, equal split {n_on}/{n_outside}/{n_inside}"


def test_sample_scalar_field():
    mesh, patch_vertices, _ = get_fixtures()

    X, y = sample_scalar_field(patch_vertices, mesh, max_gpr_points=300)

    assert X.shape[1] == 3, \
        f"X should have 3 columns, got {X.shape[1]}"
    assert len(X) == len(y), \
        f"X and y length mismatch"
    assert len(X) <= 300, \
        f"Exceeded max_gpr_points: {len(X)} > 300"

    # No NaNs or Infs -- catches silent failures in normal interpolation
    assert not np.any(np.isnan(X)), "NaN in X"
    assert not np.any(np.isinf(X)), "Inf in X"
    assert not np.any(np.isnan(y)), "NaN in y"
    assert not np.any(np.isinf(y)), "Inf in y"

    return True, f"X shape {X.shape}, no NaN/Inf, {len(X)} points"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ("estimate_epsilon",      test_estimate_epsilon),
    ("get_sample_count",      test_get_sample_count),
    ("sample_surface",        test_sample_surface),
    ("build_training_data",   test_build_training_data),
    ("subsample",             test_subsample),
    ("sample_scalar_field",   test_sample_scalar_field),
]

def run_all():
    import sys
    print("\n=== scalar_field tests ===\n")
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