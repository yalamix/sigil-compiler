# tests/test_segmentation.py

import sys
import numpy as np
import trimesh
from sigil.geometry.segmentation import (
    build_graph,
    compute_geodesic_distances,
    estimate_n_seeds,
    farthest_point_sampling,
    extract_patches_radius,
    extract_patches_voronoi,
    find_adjacent_patches,
    get_patch_faces,
    estimate_patch_radius,
)

def make_test_mesh():
    """Unit sphere, subdivisions=4. 2562 vertices, 5120 faces."""
    return trimesh.creation.icosphere(subdivisions=4)

# ---------------------------------------------------------------------------
# Individual tests — each returns (passed: bool, message: str)
# ---------------------------------------------------------------------------

def test_build_graph():
    mesh = make_test_mesh()
    graph = build_graph(mesh)
    V = len(mesh.vertices)
    E = len(mesh.edges_unique)

    assert graph.shape == (V, V), \
        f"Graph shape {graph.shape} != ({V}, {V})"
    assert graph.nnz == 2 * E, \
        f"Expected {2*E} nonzero entries (both directions), got {graph.nnz}"
    assert graph.data.min() > 0, \
        f"Edge weights should be positive, got min={graph.data.min()}"

    return True, f"Graph shape {graph.shape}, {graph.nnz} edges, all weights positive"


def test_geodesic_distances():
    mesh = make_test_mesh()
    graph = build_graph(mesh)

    # Distance from vertex 0 to itself must be 0
    d = compute_geodesic_distances(graph, 0)
    assert d.shape == (len(mesh.vertices),), \
        f"Distance array shape {d.shape} != ({len(mesh.vertices)},)"
    assert d[0] == 0.0, \
        f"Distance to self should be 0, got {d[0]}"
    assert d.min() >= 0.0, \
        f"Negative distances found: min={d.min()}"

    # On a unit sphere, max geodesic distance ≈ pi (antipodal point)
    # Our approximation via edges will be close but not exact
    max_d = d.max()
    assert 2.5 < max_d < 4.0, \
        f"Max geodesic on unit sphere should be ~pi≈3.14, got {max_d}"

    return True, f"Self-distance=0, all>=0, max={max_d:.4f} (expect ~pi=3.14)"


def test_estimate_n_seeds():
    mesh = make_test_mesh()
    n = estimate_n_seeds(mesh)

    assert isinstance(n, int), f"n_seeds should be int, got {type(n)}"
    assert n >= 2, f"n_seeds should be >= 2, got {n}"
    assert n % 2 == 0, f"n_seeds should be even, got {n}"

    return True, f"n_seeds={n} (int, even, >=2)"


def test_farthest_point_sampling():
    mesh = make_test_mesh()
    graph = build_graph(mesh)
    n_seeds = 16

    seeds, dist_matrix = farthest_point_sampling(mesh, graph, n_seeds)

    assert seeds.shape == (n_seeds,), \
        f"seeds shape {seeds.shape} != ({n_seeds},)"
    assert dist_matrix.shape == (n_seeds, len(mesh.vertices)), \
        f"dist_matrix shape {dist_matrix.shape} != ({n_seeds}, {len(mesh.vertices)})"
    assert len(np.unique(seeds)) == n_seeds, \
        f"Duplicate seeds found: {n_seeds - len(np.unique(seeds))} duplicates"
    assert seeds.max() < len(mesh.vertices), \
        f"Seed index {seeds.max()} out of range (V={len(mesh.vertices)})"
    assert seeds.min() >= 0, \
        f"Negative seed index: {seeds.min()}"

    return True, f"{n_seeds} unique seeds, dist_matrix shape {dist_matrix.shape}"


def test_extract_patches_radius():
    mesh = make_test_mesh()
    graph = build_graph(mesh)
    n_seeds = 16

    seeds, dist_matrix = farthest_point_sampling(mesh, graph, n_seeds)
    radius = estimate_patch_radius(mesh, n_seeds)
    patches = extract_patches_radius(dist_matrix, radius)

    assert len(patches) == n_seeds, \
        f"Expected {n_seeds} patches, got {len(patches)}"

    for i, (patch, seed) in enumerate(zip(patches, seeds)):
        assert len(patch) > 0, f"Patch {i} is empty"
        assert seed in patch, \
            f"Seed vertex {seed} not in its own patch {i}"

    return True, f"{len(patches)} patches, all non-empty, all seeds in own patch"


def test_extract_patches_voronoi():
    mesh = make_test_mesh()
    graph = build_graph(mesh)
    n_seeds = 16
    V = len(mesh.vertices)

    seeds, dist_matrix = farthest_point_sampling(mesh, graph, n_seeds)
    patches = extract_patches_voronoi(dist_matrix)

    assert len(patches) == n_seeds, \
        f"Expected {n_seeds} patches, got {len(patches)}"

    # Every vertex must appear in exactly one patch
    all_vertices = np.concatenate(patches)
    assert len(all_vertices) == V, \
        f"Voronoi covers {len(all_vertices)} vertices, expected {V}"
    assert len(np.unique(all_vertices)) == V, \
        f"Some vertices appear in multiple Voronoi cells"

    return True, f"All {V} vertices covered, no overlaps"


def test_find_adjacent_patches():
    mesh = make_test_mesh()
    graph = build_graph(mesh)
    n_seeds = 16

    seeds, dist_matrix = farthest_point_sampling(mesh, graph, n_seeds)
    radius = estimate_patch_radius(mesh, n_seeds)
    patches = extract_patches_radius(dist_matrix, radius)
    adjacency = find_adjacent_patches(patches)

    assert len(adjacency) > 0, "No adjacent patches found -- radius may be too small"

    patch_sets = [set(p) for p in patches]
    for (i, j), overlap in adjacency.items():
        assert i < j, f"Key ({i},{j}) not in canonical order i<j"
        overlap_set = set(overlap)
        assert overlap_set.issubset(patch_sets[i]), \
            f"Overlap ({i},{j}) contains vertices not in patch {i}"
        assert overlap_set.issubset(patch_sets[j]), \
            f"Overlap ({i},{j}) contains vertices not in patch {j}"

    return True, f"{len(adjacency)} adjacent pairs, all overlaps verified"


def test_get_patch_faces():
    mesh = make_test_mesh()
    graph = build_graph(mesh)
    n_seeds = 16

    seeds, dist_matrix = farthest_point_sampling(mesh, graph, n_seeds)
    radius = estimate_patch_radius(mesh, n_seeds)
    patches = extract_patches_radius(dist_matrix, radius)

    patch_faces = get_patch_faces(mesh, patches[0])
    F = len(mesh.faces)

    assert len(patch_faces) > 0, "No faces found for patch 0"
    assert patch_faces.max() < F, \
        f"Face index {patch_faces.max()} out of range (F={F})"
    assert patch_faces.min() >= 0, \
        f"Negative face index: {patch_faces.min()}"

    # Verify inclusive criterion: every returned face has at least one
    # vertex in the patch
    patch_set = set(patches[0])
    for fi in patch_faces:
        face_verts = set(mesh.faces[fi])
        assert len(face_verts & patch_set) > 0, \
            f"Face {fi} has no vertices in patch -- inclusive criterion violated"

    return True, f"{len(patch_faces)} faces found, all satisfy inclusive criterion"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ("build_graph",              test_build_graph),
    ("geodesic_distances",       test_geodesic_distances),
    ("estimate_n_seeds",         test_estimate_n_seeds),
    ("farthest_point_sampling",  test_farthest_point_sampling),
    ("extract_patches_radius",   test_extract_patches_radius),
    ("extract_patches_voronoi",  test_extract_patches_voronoi),
    ("find_adjacent_patches",    test_find_adjacent_patches),
    ("get_patch_faces",          test_get_patch_faces),
]

def run_all():
    print("\n=== segmentation tests ===\n")
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