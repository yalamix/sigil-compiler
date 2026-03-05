# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
import numpy as np
import logging
import trimesh

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s'
)

# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(mesh):
    """
    Build sparse adjacency graph from mesh edges. Call once, reuse everywhere.
    
    mesh.edges_unique        shape (E, 2)  — vertex index pairs
    mesh.edges_unique_length shape (E,)   — 3D distance per edge
    
    We add both directions (v0->v1 and v1->v0) because the graph is undirected.
    """
    V = len(mesh.vertices)
    v0 = mesh.edges_unique[:, 0]   # shape (E,) — all source vertex indices
    v1 = mesh.edges_unique[:, 1]   # shape (E,) — all target vertex indices
    lengths = mesh.edges_unique_length  # shape (E,)

    # Both directions
    row  = np.concatenate([v0, v1])  # shape (2E,)
    col  = np.concatenate([v1, v0])  # shape (2E,)
    data = np.concatenate([lengths, lengths])  # shape (2E,)

    return csr_matrix((data, (row, col)), shape=(V, V))


# ---------------------------------------------------------------------------
# Geodesic distance
# ---------------------------------------------------------------------------

def compute_geodesic_distances(graph, source_vertex):
    """
    Dijkstra from one source to all vertices.
    
    graph:         csr_matrix (V, V) — from build_graph()
    source_vertex: int — index of starting vertex
    
    returns: (V,) array of distances
    """
    distances = shortest_path(graph, method='D',
                               directed=False,
                               indices=source_vertex)
    return distances  # shape (V,)


# ---------------------------------------------------------------------------
# Seed count estimation
# ---------------------------------------------------------------------------

def _empirical_patch_capacity(mesh, target_degree: int = 4,
                               error_tolerance: float = 1e-3) -> float:
    """
    Placeholder capacity constant: how much curvature one patch can handle.

    This will be replaced by an empirically calibrated lookup table once
    the full pipeline is running and we can measure actual SR error on
    reference meshes (sphere, bunny, face scan).

    Current approach: one patch should cover ~1/20th of surface area,
    scaled by degree and tolerance.
    """
    patch_fraction = 1.0 / 20.0
    return (mesh.area * patch_fraction) * target_degree * error_tolerance


def _estimate_fast(mesh, target_degree: int = 4,
                   error_tolerance: float = 1e-3) -> int:
    """
    O(edges) curvature estimate using normal variation between adjacent faces.
    Runs in milliseconds on any mesh size. Good enough for seed estimation.
    """
    edges = mesh.face_adjacency
    n1 = mesh.face_normals[edges[:, 0]]
    n2 = mesh.face_normals[edges[:, 1]]
    curvature_per_edge = 1.0 - np.einsum('ij,ij->i', n1, n2)

    total = np.sum(curvature_per_edge)
    median_c = np.median(curvature_per_edge)

    capacity_per_patch = _empirical_patch_capacity(mesh, target_degree, error_tolerance)
    edge_capacity = capacity_per_patch * len(edges) / mesh.area

    n = int(np.ceil(total / edge_capacity))

    logging.info(f"Fast curvature estimate: median={median_c:.6f}, total={total:.4f}")
    logging.info(f"Edge capacity per patch: {edge_capacity:.6f}")

    return n


def _estimate_accurate(mesh, target_degree: int = 4,
                        error_tolerance: float = 1e-3) -> int:
    """
    O(N) per-vertex Gaussian + mean curvature via angle defect.
    More principled than the fast estimator but takes ~3 min on 70k faces.
    Will be fast enough once we move to compiled code.
    """
    gaussian_curvature = trimesh.curvature.discrete_gaussian_curvature_measure(
        mesh, mesh.vertices, radius=1e-8
    )
    mean_curvature = trimesh.curvature.discrete_mean_curvature_measure(
        mesh, mesh.vertices, radius=1e-8
    )

    H = mean_curvature
    K = gaussian_curvature
    discriminant = np.maximum(H**2 - K, 0)
    k1 = H + np.sqrt(discriminant)
    k2 = H - np.sqrt(discriminant)
    curvedness = np.sqrt((k1**2 + k2**2) / 2)

    capacity_per_patch = _empirical_patch_capacity(mesh, target_degree, error_tolerance)
    total_curvature_budget = np.median(curvedness) * mesh.area

    n = int(np.ceil(total_curvature_budget / capacity_per_patch))

    logging.info(f"Accurate curvature estimate:")
    logging.info(f"  Mesh scale (max extent): {np.max(mesh.extents):.4f}")
    logging.info(f"  Mesh area: {mesh.area:.4f}")
    logging.info(f"  Curvedness: median={np.median(curvedness):.6f}, "
                 f"mean={np.mean(curvedness):.6f}, "
                 f"p95={np.percentile(curvedness, 95):.6f}")
    logging.info(f"  Total curvature budget: {total_curvature_budget:.6f}")
    logging.info(f"  Capacity per patch: {capacity_per_patch:.6f}")

    return n


def estimate_n_seeds(mesh, target_degree: int = 4,
                     error_tolerance: float = 1e-3,
                     fast: bool = True) -> int:
    """
    Estimate the number of patches needed to represent mesh with degree-d polynomials.

    Args:
        mesh:            trimesh.Trimesh
        target_degree:   max polynomial degree per patch (default 4)
        error_tolerance: acceptable SR fit error (default 1e-3)
        fast:            use O(edges) normal-variation estimator (default True)
                         False uses O(N) Gaussian curvature (slower, more accurate)

    Returns:
        N: even integer >= 2
    """
    n = _estimate_fast(mesh, target_degree, error_tolerance) if fast \
        else _estimate_accurate(mesh, target_degree, error_tolerance)

    n = max(2, n)
    n = n + (n % 2)  # enforce even for binary tree merging

    logging.info(f"Estimated N seeds: {n}")
    return n



# ---------------------------------------------------------------------------
# Curvature weights
# ---------------------------------------------------------------------------

def _compute_curvature_weights(mesh, alpha=5.0):
    """
    Per-vertex curvature weight for FPS.
    w(v) = 1 + alpha * normalised_curvedness(v)

    Higher weight at v means seeds are attracted to v,
    producing smaller patches in high-curvature regions.

    Uses face normal variation as a fast O(edges) curvature proxy.

    mesh.face_adjacency       shape (A, 2) — pairs of adjacent face indices
    mesh.face_adjacency_edges shape (A, 2) — the shared edge [v0, v1] per pair
    mesh.face_normals         shape (F, 3) — one normal per face

    returns: (V,) float array, values >= 1.0
    """
    V = len(mesh.vertices)
    vertex_curvature = np.zeros(V)   # (V,) accumulator
    vertex_counts = np.zeros(V)      # (V,) how many edges contributed

    # Normal variation per adjacent face pair
    edges = mesh.face_adjacency                    # (A, 2)
    n1 = mesh.face_normals[edges[:, 0]]            # (A, 3)
    n2 = mesh.face_normals[edges[:, 1]]            # (A, 3)
    edge_curvature = 1.0 - np.einsum('ij,ij->i', n1, n2)  # (A,)
    # einsum 'ij,ij->i': multiply elementwise then sum across j (the xyz axis)
    # equivalent to: np.sum(n1 * n2, axis=1) but faster

    # Scatter edge curvature back to the two vertices of each shared edge
    shared_edges = mesh.face_adjacency_edges       # (A, 2) vertex index pairs
    v0 = shared_edges[:, 0]                        # (A,)
    v1 = shared_edges[:, 1]                        # (A,)

    np.add.at(vertex_curvature, v0, edge_curvature)
    np.add.at(vertex_curvature, v1, edge_curvature)
    np.add.at(vertex_counts, v0, 1)
    np.add.at(vertex_counts, v1, 1)
    # np.add.at is unbuffered in-place addition — handles duplicate indices
    # correctly, unlike vertex_curvature[v0] += edge_curvature which silently
    # drops duplicate index contributions

    vertex_counts = np.maximum(vertex_counts, 1)   # avoid divide by zero
    vertex_curvature /= vertex_counts              # (V,) average per vertex

    # Normalise to [0, 1]
    vmax = vertex_curvature.max()
    if vmax > 0:
        vertex_curvature /= vmax

    weights = 1.0 + alpha * vertex_curvature       # (V,) values in [1, 1+alpha]
    return weights


# ---------------------------------------------------------------------------
# Farthest point sampling
# ---------------------------------------------------------------------------

def farthest_point_sampling(mesh, graph, n_seeds, curvature_weighted=False):
    """
    Place n_seeds seeds on the mesh via farthest-point sampling.

    mesh:               trimesh.Trimesh
    graph:              csr_matrix (V, V) from build_graph() — passed in, not rebuilt
    n_seeds:            int
    curvature_weighted: bool

    returns:
        seed_vertices:   (n_seeds,)      int array of vertex indices
        distance_matrix: (n_seeds, V)    float array — row i = distances from seed i
    """
    V = len(mesh.vertices)
    distance_matrix = np.zeros((n_seeds, V))      # (n_seeds, V) — fill as we go
    min_distances = np.full(V, np.inf)             # (V,) — closest seed so far

    if curvature_weighted:
        weights = _compute_curvature_weights(mesh) # (V,) — one weight per vertex
    else:
        weights = np.ones(V)                       # (V,) — uniform

    # Seed 0: vertex closest to mesh centroid
    centroid = mesh.centroid                       # (3,) — [x, y, z]
    dists_to_centroid = np.linalg.norm(
        mesh.vertices - centroid, axis=1           # (V, 3) - (3,) broadcasts to (V, 3)
    )                                              # result: (V,)
    first_seed = int(np.argmin(dists_to_centroid))

    seeds = [first_seed]

    # Dijkstra from seed 0
    d = compute_geodesic_distances(graph, first_seed)  # (V,)
    distance_matrix[0] = d
    min_distances = d / (weights + 1e-8)               # (V,) weighted

    logging.info(f"FPS: seed 1/{n_seeds} at vertex {first_seed}")

    for i in range(1, n_seeds):
        # Exclude already-chosen seeds from selection
        min_distances[seeds] = -np.inf

        next_seed = int(np.argmax(min_distances))      # scalar
        seeds.append(next_seed)

        # Dijkstra from new seed
        d = compute_geodesic_distances(graph, next_seed)  # (V,)
        distance_matrix[i] = d

        # Update running minimum with weighted distances
        weighted_d = d / (weights + 1e-8)              # (V,)
        min_distances = np.minimum(min_distances, weighted_d)  # (V,)

        logging.info(f"FPS: seed {i+1}/{n_seeds} at vertex {next_seed}")

    return np.array(seeds, dtype=np.int64), distance_matrix


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def estimate_patch_radius(mesh, n_seeds, overlap_fraction=0.25):
    """
    Estimate geodesic radius so that patches cover the mesh with desired overlap.

    If patches were perfect non-overlapping equal-area discs:
        area_per_patch = total_area / n_seeds
        radius = sqrt(area_per_patch / pi)

    Scale up by (1 + overlap_fraction) to get the overlap buffer.

    mesh:             trimesh.Trimesh
    n_seeds:          int
    overlap_fraction: float — 0.25 means ~25% overlap between adjacent patches

    returns: float
    """
    area_per_patch = mesh.area / n_seeds           # scalar
    base_radius = np.sqrt(area_per_patch / np.pi)  # scalar
    radius = base_radius * (1.0 + overlap_fraction)

    logging.info(f"Patch radius: {radius:.6f} "
                 f"(area/patch={area_per_patch:.6f}, "
                 f"base_r={base_radius:.6f})")
    return radius


def extract_patches_radius(distance_matrix, radius):
    """
    distance_matrix: (n_seeds, V)
    radius:          float

    returns: list of n_seeds arrays, each shape (n_patch_vertices,)
    """
    patches = []
    for i in range(len(distance_matrix)):
        patch = np.where(distance_matrix[i] <= radius)[0]  # (n_patch_vertices,)
        patches.append(patch)
        logging.info(f"Patch {i}: {len(patch)} vertices")
    return patches


def extract_patches_voronoi(distance_matrix):
    """
    distance_matrix: (n_seeds, V)

    returns: list of n_seeds arrays, each shape (n_patch_vertices,)
    """
    # For each vertex, which seed is closest?
    assignments = np.argmin(distance_matrix, axis=0)  # (V,) values in [0, n_seeds)

    patches = []
    for i in range(len(distance_matrix)):
        patch = np.where(assignments == i)[0]          # (n_patch_vertices,)
        patches.append(patch)
        logging.info(f"Voronoi cell {i}: {len(patch)} vertices")
    return patches


# ---------------------------------------------------------------------------
# Adjacency
# ---------------------------------------------------------------------------

def find_adjacent_patches(patches):
    """
    Find pairs of patches that share vertices (overlap region).

    patches: list of (n_patch_vertices,) arrays — from extract_patches_radius

    returns: dict mapping (i, j) -> (n_overlap_vertices,) array
             where i < j and patches i,j share at least one vertex
    """
    n = len(patches)
    # Convert each patch to a set for O(1) lookup
    patch_sets = [set(p) for p in patches]  # list of n sets

    adjacency = {}
    for i in range(n):
        for j in range(i + 1, n):
            overlap = patch_sets[i] & patch_sets[j]  # set intersection
            if len(overlap) > 0:
                adjacency[(i, j)] = np.array(sorted(overlap), dtype=np.int64)

    logging.info(f"Found {len(adjacency)} adjacent patch pairs")
    return adjacency


# ---------------------------------------------------------------------------
# Face utilities
# ---------------------------------------------------------------------------

def get_patch_faces(mesh, patch_vertices):
    """
    Find all faces that have at least one vertex in this patch.
    Inclusive criterion: face belongs to patch if ANY vertex is in patch.
    This avoids gaps at boundaries where a face straddles two patches.

    mesh:           trimesh.Trimesh
    patch_vertices: (n_patch_vertices,) int array of vertex indices

    returns: (n_patch_faces,) int array of face indices

    mesh.faces is (F, 3) — each row is [v0, v1, v2]
    np.isin(mesh.faces, patch_vertices) is (F, 3) bool
        True wherever a face vertex is in the patch
    np.any(..., axis=1) collapses (F, 3) -> (F,) bool
        True for faces where at least one vertex matched
    """
    mask = np.any(np.isin(mesh.faces, patch_vertices), axis=1)  # (F,) bool
    return np.where(mask)[0]                                      # (n_patch_faces,)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def segment_mesh(mesh, n_seeds=None, strategy='radius',
                 curvature_weighted=True, overlap_fraction=0.25,
                 target_degree=4, error_tolerance=1e-3):
    """
    Full segmentation pipeline. This is the only function callers need.

    mesh:               trimesh.Trimesh
    n_seeds:            int or None (None = auto-estimate)
    strategy:           'radius' or 'voronoi'
    curvature_weighted: bool — weight FPS toward high-curvature regions
    overlap_fraction:   float — for radius strategy (default 0.25 = 25% overlap)
    target_degree:      int — for seed count estimation
    error_tolerance:    float — for seed count estimation

    returns dict:
        'n_seeds':        int
        'seed_vertices':  (n_seeds,) int array
        'distance_matrix':(n_seeds, V) float array — kept for scalar_field.py
        'patches':        list of (n_patch_vertices,) arrays
        'adjacency':      dict (i,j) -> (n_overlap,) array  [radius only]
        'radius':         float  [radius only]
        'strategy':       str
    """
    if n_seeds is None:
        n_seeds = estimate_n_seeds(mesh, target_degree, error_tolerance)

    logging.info(f"Segmenting: {n_seeds} seeds, strategy='{strategy}'")

    # Build graph once — reused by all Dijkstra calls inside FPS
    graph = build_graph(mesh)

    # FPS gives us seeds AND the distance matrix as a byproduct
    seed_vertices, distance_matrix = farthest_point_sampling(
        mesh, graph, n_seeds, curvature_weighted
    )
    # seed_vertices:   (n_seeds,)
    # distance_matrix: (n_seeds, V)

    result = {
        'n_seeds':         n_seeds,
        'seed_vertices':   seed_vertices,
        'distance_matrix': distance_matrix,
        'strategy':        strategy,
    }

    if strategy == 'radius':
        radius = estimate_patch_radius(mesh, n_seeds, overlap_fraction)
        patches = extract_patches_radius(distance_matrix, radius)
        adjacency = find_adjacent_patches(patches)
        result['patches']   = patches
        result['adjacency'] = adjacency
        result['radius']    = radius

    elif strategy == 'voronoi':
        patches = extract_patches_voronoi(distance_matrix)
        result['patches']   = patches
        result['adjacency'] = {}

    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'radius' or 'voronoi'.")

    patch_sizes = [len(p) for p in patches]
    logging.info(f"Segmentation complete:")
    logging.info(f"  Patches:            {len(patches)}")
    logging.info(f"  Vertices per patch: "
                 f"min={min(patch_sizes)}, "
                 f"max={max(patch_sizes)}, "
                 f"mean={np.mean(patch_sizes):.0f}")

    return result