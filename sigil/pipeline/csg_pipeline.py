# csg_pipeline.py
# Step 1: Medial axis approximation via interior distance field local maxima.
#
# Approach:
#   1. Load and repair mesh so inside/outside tests are reliable.
#   2. Sample random points inside the mesh interior.
#   3. For each interior point compute its distance to the nearest surface point.
#      This is the "inscribed sphere radius" at that location.
#   4. Keep only local maxima of that distance field — points whose radius is
#      larger than all their neighbors. These are the medial axis points.
#   5. Visualize: transparent mesh + opaque spheres, same coordinate space.
#
# Why not Voronoi?
#   scipy Voronoi on 3000 points produces ~20k vertices and huge intermediate
#   data structures — too much RAM for a 16GB machine. The distance field
#   approach is O(n) in interior samples with a fixed KD-tree query cost.

import logging
import numpy as np
import scipy.spatial
import trimesh
import trimesh.repair
import plotly.graph_objects as go
from trimesh.ray.ray_util import contains_points as _ray_contains_points

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s'
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MESH_PATH = r'C:\Users\yalam\Documents\sigil-compiler\assets\meshes\fandisk.obj'

N_INTERIOR      = 50000   # random interior candidate points to sample
                           # more = better medial axis coverage, costs more RAM
                           # but much cheaper than Voronoi at equivalent quality

N_SURFACE_KD    = 10000   # surface points used to build the KD-tree for
                           # nearest-surface-point distance queries
                           # does not need to be huge — 10k covers the bunny well

MIN_RADIUS      = 0.01    # discard medial spheres smaller than this
                           # (noise, mesh discretization artifacts near thin regions)

LOCAL_MAX_K     = 20      # number of neighbors to check when finding local maxima
                           # a point is a local max if its radius > all k neighbors

MAX_SPHERES     = 80      # cap for visualization — show only the largest N spheres

# ---------------------------------------------------------------------------
# Step 1a: Load and repair mesh
# ---------------------------------------------------------------------------

logging.info("Loading mesh...")
# mesh = trimesh.load(MESH_PATH)
# mesh = trimesh.creation.icosphere(subdivisions=3)
mesh = trimesh.creation.box(extents=[1.0, 0.6, 0.4])
logging.info(f"Mesh loaded: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

# Repair before anything else — signed distance and inside tests depend on
# the mesh being as close to watertight as possible
logging.info("Repairing mesh...")
trimesh.repair.fix_winding(mesh)        # ensure consistent face winding
trimesh.repair.fix_normals(mesh)        # normals must point outward
trimesh.repair.fill_holes(mesh)         # close small holes

if mesh.is_watertight:
    logging.info("Mesh is watertight after repair — signed distance is reliable")
else:
    logging.info("Mesh still not watertight after repair — will use ray casting fallback")

# Normalize to unit bounding box so all thresholds are scale-independent
# regardless of the original mesh units
mesh.apply_translation(-mesh.bounds.mean(axis=0))
scale = mesh.bounds[1].max() - mesh.bounds[0].min()
mesh.apply_scale(1.0 / scale)
logging.info(f"Normalized bounds: {mesh.bounds}")

# ---------------------------------------------------------------------------
# Step 1b: Inside/outside test — signed distance with ray casting fallback
# ---------------------------------------------------------------------------

def points_inside_mesh(mesh, points):
    """
    Return boolean mask: True where points are inside the mesh.

    Primary method: trimesh signed_distance (fast, exact on watertight meshes).
    Fallback: ray casting with majority vote over 3 axis-aligned rays.
              Works even on meshes with small holes because a hole must
              be exactly on the ray direction to cause a wrong answer,
              and we take a majority vote over 3 independent rays.

    points: (N, 3) float array
    returns: (N,) bool array
    """
    logging.info("Using ray casting for inside test...")
    try:
        ray_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    except Exception:
        ray_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

    directions = [
        [0.4395064455, 0.617598629942, 0.652231566745],  # trimesh default
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0.577, 0.577, 0.577],
    ]
    votes = np.zeros(len(points), dtype=int)
    for d in directions:
        votes += _ray_contains_points(
            ray_intersector, points, check_direction=d
        ).astype(int)

    return votes >= 3  # majority of 5

# ---------------------------------------------------------------------------
# Step 1c: Sample interior points
# ---------------------------------------------------------------------------

logging.info(f"Sampling {N_INTERIOR} interior candidate points...")

# Sample uniformly inside the bounding box, then filter to mesh interior.
# We oversample because many points will be outside the mesh.
# Bounding box is [-0.5, 0.5]^3 after normalization, expand slightly.
bb_min = mesh.bounds[0] * 1.05
bb_max = mesh.bounds[1] * 1.05

rng = np.random.default_rng(42)
candidates = rng.uniform(bb_min, bb_max, (N_INTERIOR, 3))

inside_mask     = points_inside_mesh(mesh, candidates)
interior_points = candidates[inside_mask]
logging.info(f"Interior points: {len(interior_points)} / {N_INTERIOR} "
             f"({100*len(interior_points)/N_INTERIOR:.1f}% fill rate)")

if len(interior_points) < 100:
    raise RuntimeError(
        "Too few interior points found. Check mesh repair or increase N_INTERIOR."
    )

# ---------------------------------------------------------------------------
# Step 1d: Build KD-tree on surface points for distance queries
# ---------------------------------------------------------------------------

logging.info(f"Sampling {N_SURFACE_KD} surface points for KD-tree...")
surface_pts, _ = trimesh.sample.sample_surface(mesh, N_SURFACE_KD)

logging.info("Building KD-tree on surface points...")
kd_tree = scipy.spatial.cKDTree(surface_pts)

# For each interior point, query nearest surface point distance.
# This is the radius of the largest inscribed sphere centered at that point.
logging.info("Querying nearest surface distances for all interior points...")
radii, _ = kd_tree.query(interior_points, k=1, workers=-1)
# workers=-1 uses all CPU cores — this is the expensive step

logging.info(f"Distance range: {radii.min():.4f} — {radii.max():.4f}")

# ---------------------------------------------------------------------------
# Step 1e: Find local maxima of the distance field
# ---------------------------------------------------------------------------
# A point is a local maximum if its inscribed sphere radius is strictly
# greater than all its k nearest neighbors' radii.
# These are the medial axis points — centers of maximal inscribed spheres.

logging.info(f"Finding local maxima (k={LOCAL_MAX_K} neighbors)...")

# Build KD-tree on interior points themselves for neighbor queries
interior_kd = scipy.spatial.cKDTree(interior_points)

# Query k+1 neighbors (first result is the point itself)
neighbor_dists, neighbor_idx = interior_kd.query(
    interior_points, k=LOCAL_MAX_K + 1, workers=-1
)

# For each point, check if its radius exceeds all neighbors' radii
# neighbor_idx[:, 0] is always the point itself, skip it
neighbor_radii = radii[neighbor_idx[:, 1:]]   # (N_interior, LOCAL_MAX_K)
is_local_max   = np.all(radii[:, None] > neighbor_radii, axis=1)

medial_centers = interior_points[is_local_max]
medial_radii   = radii[is_local_max]
logging.info(f"Local maxima found: {len(medial_centers)}")

# ---------------------------------------------------------------------------
# Step 1f: Filter and sort
# ---------------------------------------------------------------------------

# Remove tiny spheres — noise from mesh discretization in thin regions
valid = medial_radii >= MIN_RADIUS
medial_centers = medial_centers[valid]
medial_radii   = medial_radii[valid]
logging.info(f"After radius filter (>= {MIN_RADIUS}): {len(medial_centers)} spheres")

# Sort by radius descending — largest (most important) first
order          = np.argsort(medial_radii)[::-1]
medial_centers = medial_centers[order]
medial_radii   = medial_radii[order]

# Cap for visualization
if len(medial_centers) > MAX_SPHERES:
    logging.info(f"Capping to {MAX_SPHERES} largest spheres for visualization")
    medial_centers = medial_centers[:MAX_SPHERES]
    medial_radii   = medial_radii[:MAX_SPHERES]

logging.info(f"Final sphere count: {len(medial_centers)}")
logging.info(f"Largest sphere: center={medial_centers[0]}, "
             f"radius={medial_radii[0]:.4f}")

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

# 1. All sphere centers must be inside the mesh
inside_check = points_inside_mesh(mesh, medial_centers)
n_outside = (~inside_check).sum()
if n_outside > 0:
    logging.warning(f"{n_outside} sphere centers are outside the mesh — "
                    f"filtering them out")
    medial_centers = medial_centers[inside_check]
    medial_radii   = medial_radii[inside_check]
else:
    logging.info("Sanity check passed: all sphere centers are inside the mesh")

# 2. Largest sphere should be near the geometric center of the mesh
mesh_center      = mesh.bounds.mean(axis=0)
dist_to_center   = np.linalg.norm(medial_centers[0] - mesh_center)
logging.info(f"Largest sphere dist to mesh center: {dist_to_center:.4f} "
             f"(small = good for convex shapes, may be larger for bunnies)")

# 3. No sphere should have radius > half the bounding box diagonal
# (would mean it extends outside the mesh)
max_sensible_radius = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]) / 2
if np.any(medial_radii > max_sensible_radius):
    logging.warning("Some spheres have implausibly large radii — check mesh repair")
else:
    logging.info("Sanity check passed: all radii are within sensible range")

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
# Mesh: high transparency (opacity=0.08) so interior is visible.
# Spheres: lower transparency (opacity=0.6), colored warm→cool by radius.
# Same coordinate space — not side by side.

logging.info("Building interactive visualization...")

fig = go.Figure()

# --- Transparent mesh ---
mesh_verts = np.array(mesh.vertices)
mesh_faces = np.array(mesh.faces)

fig.add_trace(go.Mesh3d(
    x         = mesh_verts[:, 0],
    y         = mesh_verts[:, 1],
    z         = mesh_verts[:, 2],
    i         = mesh_faces[:, 0],
    j         = mesh_faces[:, 1],
    k         = mesh_faces[:, 2],
    color     = 'red',
    opacity   = 0.08,
    name      = 'mesh',
    showlegend= True,
))

# --- Spheres ---
# Plotly has no native sphere primitive so we tessellate with a UV grid.
# 20x20 resolution is enough to see the shape clearly without killing the browser.
u  = np.linspace(0, 2 * np.pi, 20)
v  = np.linspace(0, np.pi, 20)
uu, vv = np.meshgrid(u, v)
unit_x = np.cos(uu) * np.sin(vv)   # unit sphere x
unit_y = np.sin(uu) * np.sin(vv)   # unit sphere y
unit_z = np.cos(vv)                 # unit sphere z

# Precompute triangle indices for the UV grid — same for every sphere
rows, cols = unit_x.shape
tri_i, tri_j, tri_k = [], [], []
for row in range(rows - 1):
    for col in range(cols - 1):
        a  = row * cols + col
        b  = row * cols + col + 1
        c_ = (row + 1) * cols + col
        d  = (row + 1) * cols + col + 1
        tri_i += [a,  a]
        tri_j += [b,  c_]
        tri_k += [c_, d]

# Normalize radii to [0,1] for coloring
r_norm = (medial_radii - medial_radii.min()) / (
          medial_radii.max() - medial_radii.min() + 1e-8)

for i, (c, r, rn) in enumerate(zip(medial_centers, medial_radii, r_norm)):
    # Scale and translate unit sphere to this primitive
    sx = c[0] + r * unit_x
    sy = c[1] + r * unit_y
    sz = c[2] + r * unit_z

    # Warm (orange-red) for large spheres, cool (blue) for small
    red   = int(220 * rn + 35)
    blue  = int(220 * (1 - rn) + 35)
    color = f'rgb({red}, 80, {blue})'

    fig.add_trace(go.Mesh3d(
        x          = sx.ravel(),
        y          = sy.ravel(),
        z          = sz.ravel(),
        i          = tri_i,
        j          = tri_j,
        k          = tri_k,
        color      = color,
        opacity    = 0.55,
        name       = f'sphere_{i} r={r:.3f}',
        showlegend = (i < 8),   # only first 8 in legend to avoid clutter
    ))

fig.update_layout(
    title  = f'Medial axis: {len(medial_centers)} spheres inside mesh',
    scene  = dict(aspectmode='data'),
    width  = 1100,
    height = 850,
)

out_path = 'medial_axis_step1.html'
fig.write_html(out_path)
logging.info(f"Saved: {out_path}")
# logging.info("What to look for:")
# logging.info("  - Large warm sphere should sit in the bunny body/butt")
# logging.info("  - Medium spheres should appear in head, chest regions")
# logging.info("  - Small cool spheres should trace ears and fine detail")
# logging.info("  - No sphere should visibly protrude outside the red mesh")

# ---------------------------------------------------------------------------
# Step 2: Primitive type promotion via curvature measures
# ---------------------------------------------------------------------------
# For each medial sphere, query the discrete Gaussian and mean curvature
# measures from trimesh at the sphere center using the sphere radius.
#
# These measures integrate curvature over the entire sphere neighborhood,
# so a sphere touching mixed geometry (flat base + curved wall) gets a
# weighted average — it naturally picks the dominant surface type.
#
# After normalizing by ball volume, we get true curvature values:
#   K_norm = Gaussian curvature ≈ κ1 * κ2
#   H_norm = Mean curvature    ≈ (κ1 + κ2) / 2
#
# Classification:
#   K ≈ 0, H ≈ 0  -> both principal curvatures flat -> box
#   K ≈ 0, H large -> one flat, one curved           -> capsule
#   K large        -> both curved                    -> sphere
#
# All thresholds are on normalized curvature values, which are
# scale-independent because we normalized the mesh to unit bounding box.

logging.info("=" * 60)
logging.info("Step 2: Primitive type promotion")

# Thresholds on normalized curvature.
# These are tuned for a mesh normalized to unit bounding box.
# K_SPHERE_THRESHOLD:  below this, Gaussian curvature is "flat" in at least one direction
# H_CAPSULE_THRESHOLD: above this, mean curvature is "curved enough" for a capsule
K_SPHERE_THRESHOLD  = 0.01   # if K_norm < this -> not a sphere (box or capsule)
H_CAPSULE_THRESHOLD = 0.05   # if H_norm > this (and K low) -> capsule, else box

# Minimum neighborhood fraction — skip promotion if sphere touches too little surface
# (avoids classifying primitives in sparse regions with noisy curvature estimates)
MIN_NEIGHBOR_FRACTION = 0.003   # at least 0.3% of surface points in neighborhood
MIN_NEIGHBORS         = int(MIN_NEIGHBOR_FRACTION * N_SURFACE_KD)

# Maximum neighborhood fraction — skip promotion if sphere touches too much surface
# (sphere is so large its neighborhood is the whole mesh, not a local region)
MAX_NEIGHBOR_FRACTION = 0.15
MAX_NEIGHBORS         = int(MAX_NEIGHBOR_FRACTION * N_SURFACE_KD)

# Ball volume normalization factor: 4/3 * pi * r^3
# trimesh returns raw integral over ball, we divide to get curvature density
def _ball_volume(r):
    return (4.0 / 3.0) * np.pi * r ** 3

# Build surface KD-tree for neighbor counting
# (reuse surface_pts from step 1)
surface_kd = scipy.spatial.cKDTree(surface_pts)

primitives = []

for i, (center, radius) in enumerate(zip(medial_centers, medial_radii)):

    # Count neighbors to decide if this sphere's neighborhood is meaningful
    neighbor_idx = surface_kd.query_ball_point(center, radius * 2.0)
    n_neighbors  = len(neighbor_idx)

    if n_neighbors < MIN_NEIGHBORS or n_neighbors > MAX_NEIGHBORS:
        # Too few: noisy estimate. Too many: not a local region.
        # Default to sphere — safe fallback, will be refined in optimization.
        primitives.append({
            'type':        'sphere',
            'center':      center,
            'radius':      radius,
            'axis':        None,
            'half_height': None,
            'axes':        None,
        })
        logging.info(f"Primitive {i}: r={radius:.3f} neighbors={n_neighbors} "
                     f"-> sphere (out of range, skipping curvature)")
        continue

    # Query curvature measures at this sphere center with this sphere radius
    ball_vol = _ball_volume(radius)

    K_raw = trimesh.curvature.discrete_gaussian_curvature_measure(
        mesh, [center], radius
    )[0]
    H_raw = trimesh.curvature.discrete_mean_curvature_measure(
        mesh, [center], radius
    )[0]

    # Normalize by ball volume to get curvature density
    K_norm = K_raw / (ball_vol + 1e-10)
    H_norm = H_raw / (ball_vol + 1e-10)

    logging.info(f"Primitive {i}: r={radius:.3f} neighbors={n_neighbors} "
                 f"K={K_norm:.4f} H={H_norm:.4f}", )

    if abs(K_norm) < K_SPHERE_THRESHOLD:
        # Gaussian curvature near zero — at least one principal curvature is flat

        if abs(H_norm) > H_CAPSULE_THRESHOLD:
            # Mean curvature significant — one direction curved, one flat -> capsule
            # Find the capsule axis via PCA on local surface points
            # (PCA is reliable here because we already know it's elongated)
            local_pts   = surface_pts[neighbor_idx]
            local_mean  = local_pts.mean(axis=0)
            centered    = local_pts - local_mean
            _, S, Vt    = np.linalg.svd(centered, full_matrices=False)

            axis        = Vt[0]   # direction of most variance = capsule long axis
            projections = centered @ axis
            half_height = float(projections.max() - projections.min()) / 2.0

            primitives.append({
                'type':        'capsule',
                'center':      center,
                'radius':      radius,
                'axis':        axis,
                'half_height': half_height,
                'axes':        Vt,
            })
            logging.info(f"  -> capsule (axis={axis.round(2)}, "
                         f"half_height={half_height:.3f})")

        else:
            # Both curvatures near zero -> flat region -> box
            # Box orientation from PCA — normal is axis of least variance
            local_pts   = surface_pts[neighbor_idx]
            local_mean  = local_pts.mean(axis=0)
            centered    = local_pts - local_mean
            _, S, Vt    = np.linalg.svd(centered, full_matrices=False)

            # Vt[2] = least variance direction = normal to the flat face
            normal      = Vt[2]
            projections = centered @ normal
            half_height = float(projections.max() - projections.min()) / 2.0

            # Planar half-extents from the two larger PCA axes
            proj0       = centered @ Vt[0]
            proj1       = centered @ Vt[1]
            half_ext0   = float(proj0.max() - proj0.min()) / 2.0
            half_ext1   = float(proj1.max() - proj1.min()) / 2.0

            primitives.append({
                'type':        'box',
                'center':      center,
                'radius':      max(half_ext0, half_ext1),  # larger planar extent
                'axis':        normal,
                'half_height': half_height,
                'axes':        Vt,
                'half_extents': np.array([half_ext0, half_ext1, half_height]),
            })
            logging.info(f"  -> box (normal={normal.round(2)}, "
                         f"half_extents={[round(half_ext0,3), round(half_ext1,3), round(half_height,3)]})")

    else:
        # Gaussian curvature significant -> both directions curved -> sphere
        primitives.append({
            'type':        'sphere',
            'center':      center,
            'radius':      radius,
            'axis':        None,
            'half_height': None,
            'axes':        None,
        })
        logging.info(f"  -> sphere")

# Count types for logging
type_counts = {}
for p in primitives:
    type_counts[p['type']] = type_counts.get(p['type'], 0) + 1
logging.info(f"Primitive types after promotion: {type_counts}")

# ---------------------------------------------------------------------------
# Step 2 sanity checks
# ---------------------------------------------------------------------------

# All capsule axes should be unit vectors
for i, p in enumerate(primitives):
    if p['axis'] is not None:
        norm = np.linalg.norm(p['axis'])
        assert abs(norm - 1.0) < 1e-5, \
            f"Primitive {i} axis is not unit length: {norm}"

logging.info("Sanity check passed: all axes are unit vectors")

# Half heights should be positive
for i, p in enumerate(primitives):
    if p['half_height'] is not None:
        assert p['half_height'] > 0, \
            f"Primitive {i} has non-positive half_height: {p['half_height']}"

logging.info("Sanity check passed: all half_heights are positive")

# ---------------------------------------------------------------------------
# Step 2 visualization
# ---------------------------------------------------------------------------
# Same as step 1 but now capsules are drawn as cylinders + two end caps,
# boxes as rectangular prisms, spheres as UV spheres.
# Mesh stays transparent red. Primitives colored by type:
#   sphere  -> blue
#   capsule -> green
#   box     -> orange

logging.info("Building step 2 visualization...")

fig2 = go.Figure()

# --- Transparent mesh ---
fig2.add_trace(go.Mesh3d(
    x         = mesh_verts[:, 0],
    y         = mesh_verts[:, 1],
    z         = mesh_verts[:, 2],
    i         = mesh_faces[:, 0],
    j         = mesh_faces[:, 1],
    k         = mesh_faces[:, 2],
    color     = 'red',
    opacity   = 0.08,
    name      = 'mesh',
    showlegend= True,
))

def _add_sphere(fig, center, radius, color, name, showlegend=False):
    """Add a UV-tessellated sphere to a plotly figure."""
    u  = np.linspace(0, 2 * np.pi, 16)
    v  = np.linspace(0, np.pi, 16)
    uu, vv = np.meshgrid(u, v)
    sx = center[0] + radius * np.cos(uu) * np.sin(vv)
    sy = center[1] + radius * np.sin(uu) * np.sin(vv)
    sz = center[2] + radius * np.cos(vv)

    rows, cols = uu.shape
    ti, tj, tk = [], [], []
    for row in range(rows - 1):
        for col in range(cols - 1):
            a  = row * cols + col
            b  = row * cols + col + 1
            c_ = (row + 1) * cols + col
            d  = (row + 1) * cols + col + 1
            ti += [a, a];  tj += [b, c_];  tk += [c_, d]

    fig.add_trace(go.Mesh3d(
        x=sx.ravel(), y=sy.ravel(), z=sz.ravel(),
        i=ti, j=tj, k=tk,
        color=color, opacity=0.6,
        name=name, showlegend=showlegend,
    ))


def _add_capsule(fig, center, radius, axis, half_height, color, name,
                 showlegend=False):
    """
    Add a capsule to a plotly figure.
    A capsule = cylinder of given radius and half_height,
    capped with two hemispheres at each end.
    axis is the unit vector along the capsule's long direction.
    """
    # Build a local coordinate frame aligned to axis
    # We need two vectors perpendicular to axis
    axis = axis / (np.linalg.norm(axis) + 1e-10)
    # Pick an arbitrary vector not parallel to axis
    up   = np.array([0, 0, 1]) if abs(axis[2]) < 0.9 else np.array([0, 1, 0])
    perp1 = np.cross(axis, up);  perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(axis, perp1)

    # Cylinder body
    n_phi = 16
    phi   = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    # Two rings: one at +half_height, one at -half_height along axis
    top_center    = center + half_height * axis
    bottom_center = center - half_height * axis
    top_ring    = top_center    + radius * (np.outer(np.cos(phi), perp1)
                                          + np.outer(np.sin(phi), perp2))
    bottom_ring = bottom_center + radius * (np.outer(np.cos(phi), perp1)
                                          + np.outer(np.sin(phi), perp2))
    # Combine into triangles for the cylinder wall
    cyl_verts = np.vstack([top_ring, bottom_ring])  # (2*n_phi, 3)
    ti, tj, tk = [], [], []
    for j in range(n_phi):
        a = j;              b = (j+1) % n_phi
        c_ = j + n_phi;    d = (j+1) % n_phi + n_phi
        ti += [a, a];  tj += [b, c_];  tk += [c_, d]

    fig.add_trace(go.Mesh3d(
        x=cyl_verts[:,0], y=cyl_verts[:,1], z=cyl_verts[:,2],
        i=ti, j=tj, k=tk,
        color=color, opacity=0.6,
        name=name, showlegend=showlegend,
    ))

    # Two hemispherical end caps
    for cap_center, sign in [(top_center, 1), (bottom_center, -1)]:
        u  = np.linspace(0, 2 * np.pi, 16)
        v  = np.linspace(0, np.pi / 2, 10)   # only half sphere
        uu, vv = np.meshgrid(u, v)
        # Hemisphere in local frame, oriented along +/- axis
        hx = (cap_center[0]
              + radius * np.sin(vv) * (np.cos(uu) * perp1[0]
                                      + np.sin(uu) * perp2[0])
              + radius * np.cos(vv) * sign * axis[0])
        hy = (cap_center[1]
              + radius * np.sin(vv) * (np.cos(uu) * perp1[1]
                                      + np.sin(uu) * perp2[1])
              + radius * np.cos(vv) * sign * axis[1])
        hz = (cap_center[2]
              + radius * np.sin(vv) * (np.cos(uu) * perp1[2]
                                      + np.sin(uu) * perp2[2])
              + radius * np.cos(vv) * sign * axis[2])

        rows2, cols2 = uu.shape
        ti2, tj2, tk2 = [], [], []
        for row in range(rows2 - 1):
            for col in range(cols2 - 1):
                a  = row * cols2 + col
                b  = row * cols2 + col + 1
                c_ = (row + 1) * cols2 + col
                d  = (row + 1) * cols2 + col + 1
                ti2 += [a, a];  tj2 += [b, c_];  tk2 += [c_, d]

        fig.add_trace(go.Mesh3d(
            x=hx.ravel(), y=hy.ravel(), z=hz.ravel(),
            i=ti2, j=tj2, k=tk2,
            color=color, opacity=0.6,
            name=name, showlegend=False,
        ))


def _add_box(fig, center, radius, axis, half_height, axes_mat, color, name,
             showlegend=False):
    """
    Add an oriented box to a plotly figure.
    axes_mat rows are the three principal axes (from PCA).
    radius = half-extent in the two planar directions.
    half_height = half-extent in the normal direction.
    """
    # 8 corners of the box in local frame, then rotate to world
    a0 = axes_mat[0] * radius        # half-extent along first axis
    a1 = axes_mat[1] * radius        # half-extent along second axis
    a2 = axes_mat[2] * half_height   # half-extent along normal

    corners = np.array([
        center + s0*a0 + s1*a1 + s2*a2
        for s0 in [-1, 1]
        for s1 in [-1, 1]
        for s2 in [-1, 1]
    ])   # (8, 3)

    # 6 faces of the box as pairs of triangles
    # corners are ordered: s0 in {-1,1}, s1 in {-1,1}, s2 in {-1,1}
    # indices: 0=---, 1=--+, 2=-+-, 3=-++, 4=+--, 5=+-+, 6=++-, 7=+++
    face_quads = [
        [0,1,3,2], [4,5,7,6],   # s2 = -1 and s2 = +1 faces
        [0,1,5,4], [2,3,7,6],   # s1 = -1 and s1 = +1 faces
        [0,2,6,4], [1,3,7,5],   # s0 = -1 and s0 = +1 faces
    ]
    ti, tj, tk = [], [], []
    for q in face_quads:
        ti += [q[0], q[0]]
        tj += [q[1], q[2]]
        tk += [q[2], q[3]]

    fig.add_trace(go.Mesh3d(
        x=corners[:,0], y=corners[:,1], z=corners[:,2],
        i=ti, j=tj, k=tk,
        color=color, opacity=0.6,
        name=name, showlegend=showlegend,
    ))


# --- Draw all primitives ---
type_colors = {'sphere': 'blue', 'capsule': 'green', 'box': 'orange'}
type_seen   = set()

for i, p in enumerate(primitives):
    ptype  = p['type']
    color  = type_colors[ptype]
    name   = f"{ptype}_{i} r={p['radius']:.3f}"
    show   = ptype not in type_seen   # show first of each type in legend
    type_seen.add(ptype)

    if ptype == 'sphere':
        _add_sphere(fig2, p['center'], p['radius'], color, name, show)

    elif ptype == 'capsule':
        _add_capsule(fig2, p['center'], p['radius'],
                     p['axis'], p['half_height'], color, name, show)

    elif ptype == 'box':
        _add_box(fig2, p['center'], p['radius'],
                 p['axis'], p['half_height'], p['axes'], color, name, show)

fig2.update_layout(
    title  = (f"Step 2: Primitive promotion — "
              f"{type_counts.get('sphere',0)} spheres, "
              f"{type_counts.get('capsule',0)} capsules, "
              f"{type_counts.get('box',0)} boxes"),
    scene  = dict(aspectmode='data'),
    width  = 1100,
    height = 850,
)

out_path2 = 'medial_axis_step2.html'
fig2.write_html(out_path2)
logging.info(f"Saved: {out_path2}")
logging.info("What to look for:")
logging.info("  - Ear regions on bunny should show green capsules")
logging.info("  - Torso/limb columns on Lucy should show green capsules")
logging.info("  - Flat regions (robes, base) may show orange boxes")
logging.info("  - Round body regions should stay blue spheres")