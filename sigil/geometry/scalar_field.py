# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import numpy as np
import logging
import trimesh

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s'
)

# ---------------------------------------------------------------------------
# Epsilon estimation
# ---------------------------------------------------------------------------

def estimate_epsilon(patch_vertices, mesh):
    """
    Compute epsilon for this patch: the offset distance for off-surface points
    and the label value assigned to them.

    patch_vertices: (N_patch,) int array of vertex indices
    mesh:           trimesh.Trimesh

    returns: float

    The bounding box diagonal scales epsilon to the patch's physical size.
    0.005 = 0.5% of diagonal -- small enough to stay near surface,
    large enough for GPR to distinguish 0 from +-epsilon numerically.
    """
    positions = mesh.vertices[patch_vertices]  # (N_patch, 3)
    bbox_min = positions.min(axis=0)           # (3,) -- min x, min y, min z
    bbox_max = positions.max(axis=0)           # (3,)
    diagonal = np.linalg.norm(bbox_max - bbox_min)  # scalar
    return 0.005 * diagonal


# ---------------------------------------------------------------------------
# Sample count
# ---------------------------------------------------------------------------

def get_sample_count(patch_faces, mesh,
                     samples_per_unit_area=5000,
                     min_samples=30,
                     max_samples=600):
    """
    How many on-surface points to generate for this patch.

    patch_faces:          (N_faces,) int array of face indices
    mesh:                 trimesh.Trimesh
    samples_per_unit_area: density target
    min_samples:          floor -- tiny patches still get enough points
    max_samples:          ceiling -- before subsampling, not GPR budget

    returns: int

    max_samples is generous (600) because we subsample afterward.
    We generate densely first for good spatial coverage, then subsample.
    """
    patch_area = mesh.area_faces[patch_faces].sum()  # scalar
    n = int(patch_area * samples_per_unit_area)
    return int(np.clip(n, min_samples, max_samples))


# ---------------------------------------------------------------------------
# Surface sampling
# ---------------------------------------------------------------------------

def sample_mesh_sdf(mesh, n_surface=10000, epsilon=0.01, n_volume=10000):
    """
    Sample signed distance field from mesh geometry.

    Three layers:
      - Surface points:       f = 0          (on mesh surface)
      - Near-surface outside: f = +epsilon   (just outside)
      - Near-surface inside:  f = -epsilon   (just inside)
      - Volume points:        f = true SDF   (random points in bounding box,
                                              clamped to [-5e, +5e])

    The volume points are critical -- without them the polynomial has no
    constraint away from the surface and produces spurious zero level sets
    in the interior and exterior.
    """
    # --- Surface and near-surface samples ---
    pts_surface, face_idx = trimesh.sample.sample_surface(mesh, n_surface)
    face_normals          = mesh.face_normals[face_idx]

    pts_outside = pts_surface + epsilon * face_normals
    pts_inside  = pts_surface - epsilon * face_normals

    # --- Volume samples ---
    bounds_min = mesh.bounds[0] * 1.2
    bounds_max = mesh.bounds[1] * 1.2
    pts_volume = np.random.uniform(bounds_min, bounds_max, (n_volume, 3))

    from trimesh.proximity import signed_distance
    sd_volume = -signed_distance(mesh, pts_volume)

    # Clamp to [-5*epsilon, +5*epsilon] so volume points don't dominate
    # the loss over surface points
    y_volume = np.clip(sd_volume, -5 * epsilon, 5 * epsilon)

    # --- Combine ---
    X = np.vstack([
        pts_surface,
        pts_outside,
        pts_inside,
        pts_volume,
    ]).astype(np.float64)

    y = np.concatenate([
        np.zeros(n_surface),
        np.full(n_surface,  epsilon),
        np.full(n_surface, -epsilon),
        y_volume,
    ])

    return X, y


def sample_surface(patch_faces, mesh, n_samples, use_gpu=False):
    """
    Sample n_samples points uniformly by area across patch faces.

    patch_faces: (N_faces,) int array of face indices
    mesh:        trimesh.Trimesh
    n_samples:   int
    use_gpu:     bool -- use CuPy if available

    returns:
        points:  (n_samples, 3) float -- 3D positions on surface
        normals: (n_samples, 3) float -- unit normals at each point

    Each point is drawn from a face chosen with probability proportional
    to face area. Within the face, the point is drawn uniformly using
    the sqrt(r1) barycentric formula.
    """
    if use_gpu:
        return _sample_surface_gpu(patch_faces, mesh, n_samples)
    return _sample_surface_cpu(patch_faces, mesh, n_samples)


def _sample_surface_cpu(patch_faces, mesh, n_samples):
    # Face corners -- shape (N_faces, 3, 3)
    # mesh.faces[patch_faces] is (N_faces, 3) -- vertex indices per face
    # mesh.vertices[...] then maps each index to its xyz
    face_indices = mesh.faces[patch_faces]          # (N_faces, 3)
    A = mesh.vertices[face_indices[:, 0]]           # (N_faces, 3)
    B = mesh.vertices[face_indices[:, 1]]           # (N_faces, 3)
    C = mesh.vertices[face_indices[:, 2]]           # (N_faces, 3)

    # Vertex normals at each corner
    NA = mesh.vertex_normals[face_indices[:, 0]]    # (N_faces, 3)
    NB = mesh.vertex_normals[face_indices[:, 1]]    # (N_faces, 3)
    NC = mesh.vertex_normals[face_indices[:, 2]]    # (N_faces, 3)

    # Choose faces weighted by area
    areas = mesh.area_faces[patch_faces]            # (N_faces,)
    probs = areas / areas.sum()                     # (N_faces,) sums to 1
    chosen = np.random.choice(len(patch_faces),
                              size=n_samples,
                              p=probs)              # (n_samples,)

    # Barycentric coordinates
    r1 = np.random.uniform(0, 1, n_samples)         # (n_samples,)
    r2 = np.random.uniform(0, 1, n_samples)         # (n_samples,)
    sqrt_r1 = np.sqrt(r1)                           # (n_samples,)

    # Weights -- shape (n_samples, 1) for broadcasting against (n_samples, 3)
    w0 = (1 - sqrt_r1)[:, None]                    # (n_samples, 1)
    w1 = (sqrt_r1 * (1 - r2))[:, None]             # (n_samples, 1)
    w2 = (sqrt_r1 * r2)[:, None]                   # (n_samples, 1)

    # Interpolate positions
    points = w0 * A[chosen] + w1 * B[chosen] + w2 * C[chosen]   # (n_samples, 3)

    # Interpolate normals
    normals = w0 * NA[chosen] + w1 * NB[chosen] + w2 * NC[chosen]  # (n_samples, 3)

    # Renormalize -- interpolated normals aren't unit length
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)     # (n_samples, 1)
    normals = normals / (lengths + 1e-8)                         # (n_samples, 3)

    return points, normals


def _sample_surface_gpu(patch_faces, mesh, n_samples):
    try:
        import cupy as cp
    except ImportError:
        logging.warning("CuPy not available, falling back to CPU sampling")
        return _sample_surface_cpu(patch_faces, mesh, n_samples)

    face_indices = mesh.faces[patch_faces]

    # Move data to GPU
    A  = cp.asarray(mesh.vertices[face_indices[:, 0]])
    B  = cp.asarray(mesh.vertices[face_indices[:, 1]])
    C  = cp.asarray(mesh.vertices[face_indices[:, 2]])
    NA = cp.asarray(mesh.vertex_normals[face_indices[:, 0]])
    NB = cp.asarray(mesh.vertex_normals[face_indices[:, 1]])
    NC = cp.asarray(mesh.vertex_normals[face_indices[:, 2]])

    areas = cp.asarray(mesh.area_faces[patch_faces])
    probs = areas / areas.sum()
    chosen = cp.random.choice(len(patch_faces), size=n_samples, p=probs)

    r1 = cp.random.uniform(0, 1, n_samples)
    r2 = cp.random.uniform(0, 1, n_samples)
    sqrt_r1 = cp.sqrt(r1)

    w0 = (1 - sqrt_r1)[:, None]
    w1 = (sqrt_r1 * (1 - r2))[:, None]
    w2 = (sqrt_r1 * r2)[:, None]

    points  = w0 * A[chosen] + w1 * B[chosen] + w2 * C[chosen]
    normals = w0 * NA[chosen] + w1 * NB[chosen] + w2 * NC[chosen]
    lengths = cp.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / (lengths + 1e-8)

    # Back to CPU -- mesh is never touched again after this
    return cp.asnumpy(points), cp.asnumpy(normals)


# ---------------------------------------------------------------------------
# Training data construction
# ---------------------------------------------------------------------------

def build_training_data(points, normals, epsilon,
                        patch_vertices, mesh,
                        include_vertices):
    """
    From surface points and normals, build the full X, y arrays.
    Optionally prepend actual mesh vertices as guaranteed ground truth.

    points:          (N, 3) -- sampled surface points
    normals:         (N, 3) -- unit normals at each point
    epsilon:         float  -- offset distance and label magnitude
    patch_vertices:  (N_patch,) -- vertex indices, used if include_vertices=True
    mesh:            trimesh.Trimesh
    include_vertices: bool

    returns:
        X: (M, 3) where M = 3*N or 3*N + N_patch depending on include_vertices
        y: (M,)
    """
    on_surface  = points                        # (N, 3)  label 0
    outside     = points + epsilon * normals    # (N, 3)  label +epsilon
    inside      = points - epsilon * normals    # (N, 3)  label -epsilon

    if include_vertices:
        verts = mesh.vertices[patch_vertices]   # (N_patch, 3)  label 0
        on_surface = np.vstack([verts,
                                on_surface])    # (N_patch + N, 3)

    X = np.vstack([on_surface, outside, inside])    # (M, 3)

    n_on  = len(on_surface)
    n_off = len(points)      # outside and inside always = N, not N + N_patch
    y = np.concatenate([
        np.zeros(n_on),                 # (n_on,)
        np.full(n_off, +epsilon),       # (n_off,)
        np.full(n_off, -epsilon),       # (n_off,)
    ])                                  # (M,)

    return X, y


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------

def subsample(X, y, epsilon, max_gpr_points=300):
    """
    Stratified subsample to GPR budget.
    Takes equal numbers from each of the three label categories.

    X:              (N, 3)
    y:              (N,)
    epsilon:        float -- used to identify label categories
    max_gpr_points: int   -- total budget, split equally across 3 categories

    returns:
        X_sub: (M, 3) where M <= max_gpr_points
        y_sub: (M,)
    """
    per_category = max_gpr_points // 3

    # Identify the three categories by label value
    # Use tolerance for float comparison
    tol = epsilon * 0.01
    idx_on      = np.where(np.abs(y) < tol)[0]
    idx_outside = np.where(y > epsilon - tol)[0]
    idx_inside  = np.where(y < -epsilon + tol)[0]

    def sample_indices(indices, n):
        if len(indices) <= n:
            return indices
        return np.random.choice(indices, size=n, replace=False)

    keep = np.concatenate([
        sample_indices(idx_on,      per_category),
        sample_indices(idx_outside, per_category),
        sample_indices(idx_inside,  per_category),
    ])

    # Shuffle so GPR doesn't see all on-surface points first
    np.random.shuffle(keep)

    return X[keep], y[keep]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def sample_scalar_field(patch_vertices, mesh,
                        samples_per_unit_area=5000,
                        max_gpr_points=300,
                        use_gpu=False):
    """
    Full pipeline: patch vertices + mesh -> X, y ready for GPR.

    patch_vertices:       (N_patch,) int array from segmentation
    mesh:                 trimesh.Trimesh
    samples_per_unit_area: sampling density before subsampling
    max_gpr_points:       GPR budget (total across all three categories)
    use_gpu:              use CuPy for face sampling if available

    returns:
        X: (M, 3)  training points
        y: (M,)    scalar labels
    """
    from sigil.geometry.segmentation import get_patch_faces

    epsilon = estimate_epsilon(patch_vertices, mesh)

    patch_faces = get_patch_faces(mesh, patch_vertices)  # (N_faces,)

    # Auto-toggle: include vertices explicitly if they fit in the budget
    n_verts = len(patch_vertices)
    include_vertices = n_verts <= max_gpr_points // 3

    n_samples = get_sample_count(patch_faces, mesh,
                                 samples_per_unit_area)

    points, normals = sample_surface(patch_faces, mesh,
                                     n_samples, use_gpu)

    X, y = build_training_data(points, normals, epsilon,
                               patch_vertices, mesh,
                               include_vertices)

    X_sub, y_sub = subsample(X, y, epsilon, max_gpr_points)

    logging.debug(f"scalar_field: patch {len(patch_vertices)} verts, "
                  f"{len(patch_faces)} faces, epsilon={epsilon:.6f}, "
                  f"include_vertices={include_vertices}, "
                  f"sampled {n_samples} -> {len(X_sub)} after subsample")

    return X_sub, y_sub
