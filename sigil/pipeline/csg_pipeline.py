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

objs = [
    'bunny-low',
    'bunny-high',
    'lucy',
    'fandisk'
]

created = [
    'icosphere',
    'box',
    'torus'
]

mesh_option = "fandisk"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MESH_PATH = r'C:\Users\yalam\Documents\sigil-compiler\assets\meshes\mesh_option.obj'.replace('mesh_option', f'{mesh_option if mesh_option in objs else objs[0]}')

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

logging.info(f"Loading mesh: {mesh_option}")

if mesh_option in objs:
    mesh = trimesh.load(MESH_PATH)
elif mesh_option == 'icosphere':
    mesh = trimesh.creation.icosphere(subdivisions=3)
elif mesh_option == 'box':
    mesh = trimesh.creation.box(extents=[1.0, 0.6, 0.4])
elif mesh_option == 'torus':
    mesh = trimesh.creation.torus(
        major_radius=1.0, 
        minor_radius=0.3,
        major_sections=128,  # default is 32
        minor_sections=64,   # default is 16
    )

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

# ===========================================================================
# Step 0: Slab pre-pass — detect coplanar face regions, place exact-fit
#         extruded polygon primitives, mark their surface points as explained.
#
# This runs BEFORE the medial axis so that:
#   (a) step 1 interior sampling ignores already-explained regions
#   (b) step 2 skeleton analysis never places redundant boxes on flat faces
#
# Algorithm:
#   1. Build face adjacency graph; region-grow by face normal similarity
#   2. Filter regions below area threshold (relative to total mesh area)
#   3. For each qualifying region:
#      a. Project face vertices onto the region's mean plane
#      b. Compute 2D convex hull -> polygon vertices
#      c. Raycast inward from polygon boundary points to find thickness
#      d. Emit extruded_polygon primitive with exact surface fit
# ===========================================================================

logging.info("=" * 60)
logging.info("Step 0: Slab pre-pass")

# --- Configuration ---
SLAB_NORMAL_DOT_THRESHOLD = 0.998   # cos(angle) threshold for coplanar faces
                                     # 0.998 ≈ 3.6° tolerance
SLAB_MIN_AREA_FRACTION    = 0.01    # region must be >= 1% of total mesh area
                                     # to qualify for a slab primitive
SLAB_THICKNESS_N_RAYS     = 16      # number of boundary points to raycast for
                                     # thickness estimation
SLAB_EPSILON              = 1e-4    # small offset to avoid self-intersection
                                     # when raycasting inward

total_mesh_area = float(mesh.area)
logging.info(f"Total mesh area: {total_mesh_area:.4f}")
logging.info(f"Slab area threshold: {SLAB_MIN_AREA_FRACTION * total_mesh_area:.4f} "
             f"({SLAB_MIN_AREA_FRACTION*100:.1f}% of total)")

# ---------------------------------------------------------------------------
# 0a: Build face normal adjacency and region-grow coplanar groups
# ---------------------------------------------------------------------------

face_normals  = mesh.face_normals          # (F, 3) unit normals
face_areas    = mesh.area_faces            # (F,) per-face area
face_adj      = mesh.face_adjacency        # (E, 2) pairs of adjacent face indices
n_faces       = len(face_normals)

# For each face, find its adjacent faces and their normal dot product
logging.info(f"Region-growing {n_faces} faces by normal similarity...")

# Build adjacency list
adj_list = [[] for _ in range(n_faces)]
for fi, fj in face_adj:
    adj_list[fi].append(fj)
    adj_list[fj].append(fi)

# BFS region growing — seed from each unvisited face
visited     = np.zeros(n_faces, dtype=bool)
slab_regions = []   # list of np.array of face indices

for seed_face in range(n_faces):
    if visited[seed_face]:
        continue
    
    region      = []
    queue       = [seed_face]
    seed_normal = face_normals[seed_face]
    visited[seed_face] = True
    
    while queue:
        fi = queue.pop()
        region.append(fi)
        for fj in adj_list[fi]:
            if visited[fj]:
                continue
            # Accept neighbor if normal is nearly parallel to seed normal
            dot = abs(float(np.dot(face_normals[fj], seed_normal)))
            if dot >= SLAB_NORMAL_DOT_THRESHOLD:
                visited[fj] = True
                queue.append(fj)
    
    slab_regions.append(np.array(region))

logging.info(f"Found {len(slab_regions)} coplanar regions")

# ---------------------------------------------------------------------------
# 0b: Filter regions by area and build extruded polygon primitives
# ---------------------------------------------------------------------------

slab_primitives     = []
slab_face_mask      = np.zeros(n_faces, dtype=bool)   # faces claimed by slabs

min_area = SLAB_MIN_AREA_FRACTION * total_mesh_area

qualifying = [(r, face_areas[r].sum()) for r in slab_regions
              if face_areas[r].sum() >= min_area]
qualifying.sort(key=lambda x: -x[1])   # largest first

logging.info(f"Qualifying regions (>= {min_area:.4f} area): {len(qualifying)}")

for region_faces, region_area in qualifying:
    
    # Mean normal for this region (weighted by face area for stability)
    weights      = face_areas[region_faces]
    mean_normal  = (face_normals[region_faces] * weights[:, None]).sum(axis=0)
    mean_normal /= np.linalg.norm(mean_normal) + 1e-10

    # Collect all unique vertices in this region
    region_verts_idx = np.unique(mesh.faces[region_faces].ravel())
    region_verts     = mesh.vertices[region_verts_idx]
    region_centroid  = region_verts.mean(axis=0)

    # Build an orthonormal basis for the plane: (u, v, normal)
    # Find a vector not parallel to normal
    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, mean_normal)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    u_axis = np.cross(mean_normal, helper)
    u_axis /= np.linalg.norm(u_axis) + 1e-10
    v_axis  = np.cross(mean_normal, u_axis)
    v_axis /= np.linalg.norm(v_axis) + 1e-10

    # Project vertices onto the plane (2D coordinates)
    delta    = region_verts - region_centroid
    coords2d = np.stack([delta @ u_axis, delta @ v_axis], axis=1)  # (N, 2)

    # Concave hull via alpha shapes
    # Alpha controls tightness: smaller = more concave detail
    # We use the mean edge length of the region as a natural scale
    region_edge_lengths = np.linalg.norm(
        mesh.vertices[mesh.faces[region_faces][:, 1]] -
        mesh.vertices[mesh.faces[region_faces][:, 0]], axis=1
    )
    mean_edge = float(region_edge_lengths.mean())
    alpha = 1.0 / (mean_edge * 2.0)   # tighter than convex hull

    try:
        import alphashape
        shape2d = alphashape.alphashape(coords2d, alpha)
        if shape2d.is_empty or shape2d.geom_type not in ('Polygon', 'MultiPolygon'):
            raise ValueError("degenerate alpha shape")
        if shape2d.geom_type == 'MultiPolygon':
            # Take largest polygon
            shape2d = max(shape2d.geoms, key=lambda g: g.area)
        hull_pts2d = np.array(shape2d.exterior.coords[:-1])  # drop closing repeat
    except Exception as e:
        logging.warning(f"  Alpha shape failed ({e}), skipping.")
        continue
    # -----------------------------------------------------------------------
    # 0c: Raycast inward from centroid to find slab thickness
    # -----------------------------------------------------------------------
    # One ray from centroid along -normal, one along +normal.
    # The two hit distances sum to total thickness; we want the smaller
    # (distance to the opposite face), which is the slab thickness.

    SELF_HIT_MIN_DIST = 0.005   # ignore hits closer than this (self-intersection)
    NORMAL_OFFSET     = 0.01    # offset from surface before shooting

    thickness = None
    for origin_sign, direction in [(+1, -mean_normal), (-1, mean_normal)]:
        origin = region_centroid + origin_sign * NORMAL_OFFSET * mean_normal
        try:
            locs, _, _ = mesh.ray.intersects_location(
                ray_origins    = [origin],
                ray_directions = [direction],
                multiple_hits  = True,
            )
            if len(locs) == 0:
                continue
            dists = np.linalg.norm(locs - origin, axis=1)
            dists = dists[dists > SELF_HIT_MIN_DIST]
            if len(dists) == 0:
                continue
            dist = float(dists.min()) + NORMAL_OFFSET  # add back the offset
            if thickness is None or dist < thickness:
                thickness = dist
        except Exception:
            pass

    if thickness is None or thickness < 1e-4:
        logging.warning(f"  Thickness raycast failed or too small, skipping")
        continue

    # -----------------------------------------------------------------------
    # 0d: Emit extruded_polygon primitive
    # -----------------------------------------------------------------------
    slab_prim = {
        'type':        'extruded_polygon',
        'center':      region_centroid - mean_normal * (thickness / 2.0),
        'normal':      mean_normal,          # extrusion axis
        'u_axis':      u_axis,
        'v_axis':      v_axis,
        'hull_pts2d':  hull_pts2d,           # 2D polygon in (u,v) space
        'thickness':   thickness,
        # Fields required by generic dispatch (set to None or defaults)
        'radius':      thickness / 2.0,
        'axis':        mean_normal,
        'half_height': thickness / 2.0,
        'axes':        None,
    }
    slab_primitives.append(slab_prim)
    slab_face_mask[region_faces] = True

    area_pct = 100.0 * region_area / total_mesh_area
    logging.info(f"  Slab: area={region_area:.4f} ({area_pct:.1f}%), "
                 f"thickness={thickness:.4f}, "
                 f"hull_verts={len(hull_pts2d)}, "
                 f"normal={mean_normal.round(2)}")

logging.info(f"Step 0 complete: {len(slab_primitives)} slab primitives, "
             f"covering {slab_face_mask.sum()} / {n_faces} faces "
             f"({100*slab_face_mask.mean():.1f}%)")

# ---------------------------------------------------------------------------
# 0e: SDF for extruded_polygon
# ---------------------------------------------------------------------------
# Inigo Quilez pattern: signed distance = max(2D_polygon_sdf, |along_normal| - half_thickness)
# The 2D polygon SDF is the min signed distance to all edges (negative inside).

def _sdf_polygon_2d(p2d, hull_pts2d):
    """2D SDF for arbitrary (possibly concave) polygon. Negative inside."""
    from shapely.geometry import Point, Polygon
    poly = Polygon(hull_pts2d)
    # Vectorized via distance to boundary - inside test
    result = np.array([
        poly.boundary.distance(Point(p)) * (-1 if poly.contains(Point(p)) else 1)
        for p in p2d
    ])
    return result


def _sdf_extruded_polygon(points, center, normal, u_axis, v_axis,
                           hull_pts2d, thickness, **kwargs):
    """
    Exact SDF for a flat extruded convex polygon.
    points: (N, 3)
    Returns: (N,) signed distance
    """
    delta   = points - center                        # (N, 3)
    along   = delta @ normal                         # (N,) projection along normal
    pu      = delta @ u_axis                         # (N,) u coordinate in plane
    pv      = delta @ v_axis                         # (N,) v coordinate in plane
    p2d     = np.stack([pu, pv], axis=1)             # (N, 2)

    # 2D polygon SDF (convex hull, CCW winding assumed)
    n_verts = len(hull_pts2d)
    # For each edge, compute signed distance from p2d
    # Signed distance to concave polygon
    poly_sdf = _sdf_polygon_2d(p2d, hull_pts2d)

    # Height SDF: |along| - half_thickness
    half_t   = thickness / 2.0
    height_sdf = np.abs(along) - half_t

    # Extruded SDF: IQ's formula for extruded shape
    # d = (max(poly_sdf, 0), max(height_sdf, 0)) -> length - min(max(poly,height),0)
    w = np.stack([
        np.maximum(poly_sdf,   0.0),
        np.maximum(height_sdf, 0.0),
    ], axis=1)
    sdf_outside = np.linalg.norm(w, axis=1)
    sdf_inside  = np.minimum(np.maximum(poly_sdf, height_sdf), 0.0)
    return sdf_outside + sdf_inside


# ---------------------------------------------------------------------------
# 0f: Visualization
# ---------------------------------------------------------------------------

def _visualize_slabs(mesh, slab_primitives):
    fig = go.Figure()
    fig.add_trace(go.Mesh3d(
        x=mesh.vertices[:,0], y=mesh.vertices[:,1], z=mesh.vertices[:,2],
        i=mesh.faces[:,0], j=mesh.faces[:,1], k=mesh.faces[:,2],
        color='lightgray', opacity=0.25, name='mesh', showlegend=True,
    ))
    colors = ['blue','green','red','orange','purple','cyan','magenta']
    for si, slab in enumerate(slab_primitives):
        c      = slab['center']
        n      = slab['normal']
        u      = slab['u_axis']
        v      = slab['v_axis']
        pts2d  = slab['hull_pts2d']
        thick  = slab['thickness']
        color  = colors[si % len(colors)]

        # Draw top and bottom polygon faces + side edges
        for sign in [+1, -1]:
            offset = c + sign * (thick / 2.0) * n
            pts3d  = offset + pts2d[:,0:1]*u + pts2d[:,1:2]*v
            # Close the polygon
            xs = np.append(pts3d[:,0], pts3d[0,0])
            ys = np.append(pts3d[:,1], pts3d[0,1])
            zs = np.append(pts3d[:,2], pts3d[0,2])
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode='lines',
                line=dict(color=color, width=4),
                name=f'slab_{si}', showlegend=(sign==1),
            ))
        # Side edges
        n_h = len(pts2d)
        for k in range(n_h):
            pt2  = pts2d[k]
            top  = c + (thick/2.0)*n + pt2[0]*u + pt2[1]*v
            bot  = c - (thick/2.0)*n + pt2[0]*u + pt2[1]*v
            fig.add_trace(go.Scatter3d(
                x=[top[0],bot[0]], y=[top[1],bot[1]], z=[top[2],bot[2]],
                mode='lines', line=dict(color=color, width=2),
                showlegend=False,
            ))

    fig.update_layout(
        title=f"Step 0: {len(slab_primitives)} slab primitives",
        scene=dict(aspectmode='data'), width=1100, height=850,
    )
    fig.write_html('slab_prepass_step0.html')
    logging.info("Saved: slab_prepass_step0.html")

_visualize_slabs(mesh, slab_primitives)
quit()

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
surface_pts, _ = trimesh.sample.sample_surface(mesh, N_SURFACE_KD, seed=42)

# Mask surface points already explained by slabs (exact fit, no epsilon needed)
# We use a small epsilon only to account for floating point in the SDF
slab_explained_mask = np.zeros(len(surface_pts), dtype=bool)
for slab in slab_primitives:
    sdf = _sdf_extruded_polygon(surface_pts, **slab)
    slab_explained_mask |= (np.abs(sdf) <= 0.005)

logging.info(f"Slab pre-pass explains {slab_explained_mask.sum()} / "
             f"{len(surface_pts)} surface points "
             f"({100*slab_explained_mask.mean():.1f}%)")

# Interior points near slab regions are also excluded from medial axis search
# by removing them from the candidate pool before step 1e (local maxima finding)

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

# Exclude interior points whose nearest surface point is slab-explained
# (those regions are already handled — no medial sphere needed there)
nearest_surf_idx = kd_tree.query(interior_points, k=1, workers=-1)[1]
not_in_slab      = ~slab_explained_mask[nearest_surf_idx]
interior_points  = interior_points[not_in_slab]
radii            = radii[not_in_slab]
logging.info(f"After slab exclusion: {len(interior_points)} interior points remain")

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
# Step 2 (revised): Skeleton graph analysis
# ---------------------------------------------------------------------------
# Build an overlap graph where medial spheres are nodes and edges connect
# spheres whose volumes overlap. Then analyze connected components and their
# topology to determine primitive types.
#
# Two spheres overlap if the distance between their centers is less than
# the sum of their radii — i.e. they share interior volume.
# We use a slightly generous factor so near-touching spheres are also connected.

logging.info("=" * 60)
logging.info("Step 2 (revised): Skeleton graph analysis")

surface_kd = scipy.spatial.cKDTree(surface_pts)

OVERLAP_FACTOR = 1.1   # two spheres are "connected" if
                        # dist(c1, c2) < OVERLAP_FACTOR * (r1 + r2)
                        # > 1.0 catches near-touching spheres too

# ---------------------------------------------------------------------------
# Build adjacency list
# ---------------------------------------------------------------------------

n_spheres = len(medial_centers)

# For each sphere, find all others it overlaps with
adjacency = [[] for _ in range(n_spheres)]

for i in range(n_spheres):
    for j in range(i + 1, n_spheres):
        dist = np.linalg.norm(medial_centers[i] - medial_centers[j])
        if dist < OVERLAP_FACTOR * (medial_radii[i] + medial_radii[j]):
            adjacency[i].append(j)
            adjacency[j].append(i)

# Log connectivity for debugging
degrees = [len(adj) for adj in adjacency]
logging.info(f"Overlap graph: {n_spheres} nodes, "
             f"{sum(degrees)//2} edges, "
             f"mean degree={np.mean(degrees):.1f}, "
             f"max degree={max(degrees)}")

# ---------------------------------------------------------------------------
# Find connected components via BFS
# ---------------------------------------------------------------------------

visited    = np.zeros(n_spheres, dtype=bool)
components = []   # list of lists of sphere indices

for start in range(n_spheres):
    if visited[start]:
        continue
    # BFS from this node
    component = []
    queue     = [start]
    visited[start] = True
    while queue:
        node = queue.pop(0)
        component.append(node)
        for neighbor in adjacency[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
    components.append(component)

logging.info(f"Connected components: {len(components)}")
for ci, comp in enumerate(components):
    logging.info(f"  Component {ci}: {len(comp)} spheres, "
                 f"indices={sorted(comp)}")

# ---------------------------------------------------------------------------
# Analyze each component to determine primitive type
# ---------------------------------------------------------------------------
# Pattern recognition on the graph topology and geometry of each component.
#
# Patterns (in order of priority):
#   1. Single node                    -> sphere or ellipsoid
#   2. Linear chain (high R²)         -> capsule / cylinder / cone
#   3. Ring (centers fit a circle)    -> torus / capped torus
#   4. Planar cluster (centers flat)  -> box / disc
#   5. Branching / complex            -> keep as spheres, beam search handles it

# Thresholds — all dimensionless ratios, scale-independent
LINE_R2_THRESHOLD    = 0.92   # R² for line fit to be "linear enough"
CIRCLE_R2_THRESHOLD  = 0.85   # R² for circle fit to be "ring-shaped enough"
PLANE_FLAT_THRESHOLD = 0.10   # smallest_eigenvalue / largest for "flat enough"
CONE_SLOPE_THRESHOLD = 0.05   # slope of radius vs position to distinguish
                               # cone (high slope) from capsule/cylinder (low)
TAPER_THRESHOLD      = 0.15   # std(radii)/mean(radii) — high = tapered = capsule
                               #                          low  = uniform = cylinder


def _fit_line_r2(points):
    """
    Fit a line to 3D points via PCA (first principal component).
    Returns (axis, r2) where axis is the unit direction and r2 is the
    fraction of variance explained by the first component.
    R² close to 1.0 means points are nearly collinear.
    """
    centered = points - points.mean(axis=0)
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # R² = variance explained by first component / total variance
    r2 = S[0]**2 / (np.sum(S**2) + 1e-10)
    return Vt[0], float(r2)


def _fit_plane_flatness(points):
    """
    Fit a plane to 3D points via PCA.
    Returns (normal, flatness) where flatness = S[2]/S[0].
    Flatness close to 0 means points are nearly coplanar.
    """
    centered = points - points.mean(axis=0)
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    flatness = float(S[2] / (S[0] + 1e-10))
    return Vt[2], flatness   # Vt[2] = normal to best-fit plane


def _fit_circle_r2(points, normal):
    """
    Fit a circle to 3D points that lie approximately in a plane
    with the given normal. Projects to 2D, fits circle, returns r2.

    Circle fit via algebraic method (Pratt):
    Finds center and radius of best-fit circle in 2D.
    R² = 1 - residual_variance / total_variance
    """
    # Project points onto plane perpendicular to normal
    mean   = points.mean(axis=0)
    up     = np.array([0,0,1]) if abs(normal[2]) < 0.9 else np.array([0,1,0])
    u_axis = np.cross(normal, up);  u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(normal, u_axis)

    pts2d = np.column_stack([
        (points - mean) @ u_axis,
        (points - mean) @ v_axis,
    ])   # (n, 2)

    if len(pts2d) < 3:
        return 0.0, 0.0, mean

    # Algebraic circle fit: minimize ||(x-cx)^2 + (y-cy)^2 - r^2||
    # Rewrite as linear system: 2cx*x + 2cy*y + (r^2 - cx^2 - cy^2) = x^2+y^2
    A = np.column_stack([
        2 * pts2d[:, 0],
        2 * pts2d[:, 1],
        np.ones(len(pts2d)),
    ])
    b = pts2d[:, 0]**2 + pts2d[:, 1]**2

    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy          = result[0], result[1]
    r_fit           = np.sqrt(result[2] + cx**2 + cy**2)

    # R² of the circle fit
    residuals = np.sqrt((pts2d[:,0]-cx)**2 + (pts2d[:,1]-cy)**2) - r_fit
    ss_res    = np.sum(residuals**2)
    ss_tot    = np.sum((np.linalg.norm(pts2d, axis=1)
                        - np.mean(np.linalg.norm(pts2d, axis=1)))**2)
    r2        = float(1.0 - ss_res / (ss_tot + 1e-10))

    # Center in 3D
    center_3d = mean + cx * u_axis + cy * v_axis

    return r2, float(r_fit), center_3d


# Store promoted primitives — may replace multiple medial spheres with one
promoted_primitives = []
# Track which medial spheres have been consumed by a promoted primitive
consumed = np.zeros(n_spheres, dtype=bool)

for ci, comp in enumerate(components):
    centers_c = medial_centers[comp]   # (k, 3) centers of this component
    radii_c   = medial_radii[comp]     # (k,)   radii of this component
    k         = len(comp)

    logging.info(f"Analyzing component {ci} ({k} spheres)...")

    # --- Pattern 0: degenerate fully-connected component -> single primitive ---
    # If nearly every sphere overlaps every other, the medial axis is a plateau
    # (flat distance field in the interior). This happens for boxes and ellipsoids.
    # In this case, fit a box directly to the surface points rather than sphere centers,
    # since the sphere centers are spread across the interior and carry no shape info.
    mean_degree_c = sum(len(adjacency[idx]) for idx in comp) / k
    if mean_degree_c / k > 0.5 and k > 5:
        # Collect surface points near this component's bounding region
        comp_center = centers_c.mean(axis=0)
        comp_radius = np.linalg.norm(centers_c - comp_center, axis=1).max() \
                      + radii_c.max()
        nearby_surf = surface_kd.query_ball_point(comp_center, comp_radius)
        local_pts   = surface_pts[nearby_surf]

        _, S, Vt    = np.linalg.svd(local_pts - local_pts.mean(axis=0),
                                     full_matrices=False)
        center      = local_pts.mean(axis=0)
        proj0       = (local_pts - center) @ Vt[0]
        proj1       = (local_pts - center) @ Vt[1]
        proj2       = (local_pts - center) @ Vt[2]
        he0         = float(proj0.max() - proj0.min()) / 2.0
        he1         = float(proj1.max() - proj1.min()) / 2.0
        he2         = float(proj2.max() - proj2.min()) / 2.0

        logging.info(f"  mean_degree/k={mean_degree_c/k:.2f} -> "
                     f"degenerate, fitting box to surface points")
        logging.info(f"  -> box (half_extents=[{he0:.3f}, {he1:.3f}, {he2:.3f}])")
        promoted_primitives.append({
            'type':         'box',
            'center':       center,
            'radius':       max(he0, he1),
            'axis':         Vt[2],
            'half_height':  he2,
            'axes':         Vt,
            'half_extents': np.array([he0, he1, he2]),
            'source':       comp,
        })
        for idx in comp:
            consumed[idx] = True
        continue

    # --- Pattern 1: single node ---
    if k == 1:
        idx = comp[0]
        promoted_primitives.append({
            'type':        'sphere',
            'center':      medial_centers[idx],
            'radius':      medial_radii[idx],
            'axis':        None,
            'half_height': None,
            'axes':        None,
            'source':      [idx],
        })
        consumed[idx] = True
        logging.info(f"  -> sphere (single node)")
        continue

    # --- Pattern 2: linear chain ---
    axis, line_r2 = _fit_line_r2(centers_c)
    logging.info(f"  Line R²={line_r2:.3f}")

    if line_r2 >= LINE_R2_THRESHOLD:
        # Project centers onto fitted axis to find extent
        mean_c      = centers_c.mean(axis=0)
        projections = (centers_c - mean_c) @ axis
        half_height = float(projections.max() - projections.min()) / 2.0
        center      = mean_c + axis * (projections.max() + projections.min()) / 2.0

        # Mean radius for the primitive
        mean_radius = float(radii_c.mean())

        # Distinguish capsule / cylinder / cone by radius variation
        # Sort spheres along axis and check radius trend
        order       = np.argsort(projections)
        radii_sorted = radii_c[order]
        pos_sorted   = projections[order]

        # Fit line to radius vs position — slope indicates cone taper
        if len(pos_sorted) >= 2:
            slope = np.polyfit(pos_sorted, radii_sorted, 1)[0]
        else:
            slope = 0.0

        radius_cv = float(np.std(radii_c) / (np.mean(radii_c) + 1e-10))

        if abs(slope) > CONE_SLOPE_THRESHOLD and radius_cv > TAPER_THRESHOLD:
            # Radii change monotonically along axis -> cone
            ptype    = 'cone'
            r_bottom = float(radii_sorted[-1])   # larger end
            r_top    = float(radii_sorted[0])    # smaller end
            if slope < 0:
                r_bottom, r_top = r_top, r_bottom
            logging.info(f"  -> cone (r_top={r_top:.3f}, r_bottom={r_bottom:.3f}, "
                         f"slope={slope:.3f})")
            promoted_primitives.append({
                'type':        'cone',
                'center':      center,
                'radius':      mean_radius,
                'r_top':       r_top,
                'r_bottom':    r_bottom,
                'axis':        axis,
                'half_height': half_height,
                'axes':        None,
                'source':      comp,
            })

        elif radius_cv < TAPER_THRESHOLD:
            # Uniform radii -> cylinder
            logging.info(f"  -> cylinder (r={mean_radius:.3f}, "
                         f"half_height={half_height:.3f})")
            promoted_primitives.append({
                'type':        'cylinder',
                'center':      center,
                'radius':      mean_radius,
                'axis':        axis,
                'half_height': half_height,
                'axes':        None,
                'source':      comp,
            })

        else:
            # Tapered ends -> capsule
            logging.info(f"  -> capsule (r={mean_radius:.3f}, "
                         f"half_height={half_height:.3f})")
            promoted_primitives.append({
                'type':        'capsule',
                'center':      center,
                'radius':      mean_radius,
                'axis':        axis,
                'half_height': half_height,
                'axes':        None,
                'source':      comp,
            })

        for idx in comp:
            consumed[idx] = True
        continue

    # --- Pattern 3: ring (torus) ---
    # First check if centers are roughly coplanar
    normal, flatness = _fit_plane_flatness(centers_c)
    logging.info(f"  Plane flatness={flatness:.3f}")

    if flatness < 0.3 and k >= 4:
        # Coplanar enough — try circle fit
        circle_r2, major_radius, circle_center = _fit_circle_r2(centers_c, normal)
        logging.info(f"  Circle R²={circle_r2:.3f}, major_r={major_radius:.3f}")

        if circle_r2 >= CIRCLE_R2_THRESHOLD:
            minor_radius = float(radii_c.mean())

            # Check if it's a full ring or partial (capped torus)
            # Compute angular span of centers around circle center
            mean_c   = centers_c.mean(axis=0)
            up       = np.array([0,0,1]) if abs(normal[2]) < 0.9 else np.array([0,1,0])
            u_axis   = np.cross(normal, up);  u_axis /= np.linalg.norm(u_axis)
            v_axis   = np.cross(normal, u_axis)
            vecs     = centers_c - circle_center
            angles   = np.arctan2(vecs @ v_axis, vecs @ u_axis)
            angles   = np.sort(angles)
            # Angular span: largest gap between consecutive angles
            gaps     = np.diff(angles)
            gaps     = np.append(gaps, angles[0] + 2*np.pi - angles[-1])
            max_gap  = float(gaps.max())
            is_full_ring = max_gap < np.pi / 2   # gap < 90° -> full ring

            ptype = 'torus' if is_full_ring else 'capped_torus'
            logging.info(f"  -> {ptype} (major_r={major_radius:.3f}, "
                         f"minor_r={minor_radius:.3f}, "
                         f"max_gap={np.degrees(max_gap):.1f}°)")
            promoted_primitives.append({
                'type':         ptype,
                'center':       circle_center,
                'radius':       minor_radius,
                'major_radius': major_radius,
                'axis':         normal,   # torus symmetry axis
                'half_height':  None,
                'axes':         None,
                'source':       comp,
            })
            for idx in comp:
                consumed[idx] = True
            continue

    # --- Pattern 4: planar cluster (box) ---
    if flatness < PLANE_FLAT_THRESHOLD and k >= 6:
        _, S, Vt = np.linalg.svd(centers_c - centers_c.mean(axis=0),
                                  full_matrices=False)
        # Box half-extents from point cloud extent along each axis
        center   = centers_c.mean(axis=0)
        proj0    = (centers_c - center) @ Vt[0]
        proj1    = (centers_c - center) @ Vt[1]
        proj2    = (centers_c - center) @ Vt[2]
        he0      = float(proj0.max() - proj0.min()) / 2.0 + float(radii_c.mean())
        he1      = float(proj1.max() - proj1.min()) / 2.0 + float(radii_c.mean())
        he2      = float(proj2.max() - proj2.min()) / 2.0 + float(radii_c.mean())

        logging.info(f"  -> box (half_extents=[{he0:.3f}, {he1:.3f}, {he2:.3f}])")
        promoted_primitives.append({
            'type':         'box',
            'center':       center,
            'radius':       max(he0, he1),
            'axis':         Vt[2],
            'half_height':  he2,
            'axes':         Vt,
            'half_extents': np.array([he0, he1, he2]),
            'source':       comp,
        })
        for idx in comp:
            consumed[idx] = True
        continue

    # --- Pattern 5: complex / branching -> keep as individual spheres ---
    logging.info(f"  -> complex component, keeping as {k} individual spheres")
    for idx in comp:
        promoted_primitives.append({
            'type':        'sphere',
            'center':      medial_centers[idx],
            'radius':      medial_radii[idx],
            'axis':        None,
            'half_height': None,
            'axes':        None,
            'source':      [idx],
        })
        consumed[idx] = True

# Sanity check: all medial spheres must be consumed
assert np.all(consumed), "Some medial spheres were not assigned to a primitive"

type_counts = {}
for p in promoted_primitives:
    type_counts[p['type']] = type_counts.get(p['type'], 0) + 1
logging.info(f"Promoted primitives: {type_counts}")
logging.info(f"Total: {len(promoted_primitives)} primitives "
             f"from {n_spheres} medial spheres")

# logging.info("Skipping visualization for now. End.")
# quit()

# ---------------------------------------------------------------------------
# Step 2 sanity checks
# ---------------------------------------------------------------------------

# All capsule axes should be unit vectors
for i, p in enumerate(promoted_primitives):
    if p['axis'] is not None:
        norm = np.linalg.norm(p['axis'])
        assert abs(norm - 1.0) < 1e-5, \
            f"Primitive {i} axis is not unit length: {norm}"

logging.info("Sanity check passed: all axes are unit vectors")

# Half heights should be positive
for i, p in enumerate(promoted_primitives):
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
             showlegend=False, half_extents=None):
    """
    Add an oriented box to a plotly figure.
    axes_mat rows are the three principal axes (from PCA).
    radius = half-extent in the two planar directions.
    half_height = half-extent in the normal direction.
    """
    # Use per-axis half extents if available, otherwise fall back to radius
    if half_extents is not None:
        e0, e1, e2 = half_extents
    else:
        e0 = e1 = radius
        e2 = half_height    
    # 8 corners of the box in local frame, then rotate to world
    a0 = axes_mat[0] * e0        # half-extent along first axis
    a1 = axes_mat[1] * e1        # half-extent along second axis
    a2 = axes_mat[2] * e2        # half-extent along normal

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


def _add_torus(fig, center, major_radius, minor_radius, axis, color, name,
               showlegend=False):
    """
    Tessellate a torus and add to plotly figure.
    major_radius: distance from torus center to tube center
    minor_radius: radius of the tube
    axis: symmetry axis (normal to the torus plane)
    """
    # Build local frame perpendicular to axis
    axis  = axis / (np.linalg.norm(axis) + 1e-10)
    up    = np.array([0,0,1]) if abs(axis[2]) < 0.9 else np.array([0,1,0])
    u_ax  = np.cross(axis, up);  u_ax /= np.linalg.norm(u_ax)
    v_ax  = np.cross(axis, u_ax)

    # Two angular parameters:
    # phi: angle around the torus ring (major circle)
    # theta: angle around the tube (minor circle)
    n_phi   = 32
    n_theta = 16
    phi     = np.linspace(0, 2*np.pi, n_phi,   endpoint=False)
    theta   = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    pp, tt  = np.meshgrid(phi, theta)   # (n_theta, n_phi)

    # Torus parametric equation in local frame:
    # point = center
    #       + (major_r + minor_r*cos(theta)) * (cos(phi)*u_ax + sin(phi)*v_ax)
    #       + minor_r*sin(theta) * axis
    r       = major_radius + minor_radius * np.cos(tt)
    x       = center[0] + r*np.cos(pp)*u_ax[0] + r*np.sin(pp)*v_ax[0] \
              + minor_radius*np.sin(tt)*axis[0]
    y       = center[1] + r*np.cos(pp)*u_ax[1] + r*np.sin(pp)*v_ax[1] \
              + minor_radius*np.sin(tt)*axis[1]
    z       = center[2] + r*np.cos(pp)*u_ax[2] + r*np.sin(pp)*v_ax[2] \
              + minor_radius*np.sin(tt)*axis[2]

    # Build triangle indices for the (n_theta, n_phi) grid
    # wraps around in both directions
    ti, tj, tk = [], [], []
    for i in range(n_theta):
        for j in range(n_phi):
            a  = i * n_phi + j
            b  = i * n_phi + (j+1) % n_phi
            c_ = ((i+1) % n_theta) * n_phi + j
            d  = ((i+1) % n_theta) * n_phi + (j+1) % n_phi
            ti += [a,  a]
            tj += [b,  c_]
            tk += [c_, d]

    fig.add_trace(go.Mesh3d(
        x=x.ravel(), y=y.ravel(), z=z.ravel(),
        i=ti, j=tj, k=tk,
        color=color, opacity=0.6,
        name=name, showlegend=showlegend,
    ))


# --- Draw all primitives ---
type_colors = {
    'sphere':   'blue',
    'capsule':  'green',
    'cylinder': 'cyan',
    'cone':     'purple',
    'box':      'orange',
    'torus':    'magenta',
    'capped_torus': 'pink',
}
type_seen   = set()

for i, p in enumerate(promoted_primitives):
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
                p['axis'], p['half_height'], p['axes'], color, name, show,
                half_extents=p.get('half_extents'))

    elif ptype in ('torus', 'capped_torus'):
        _add_torus(fig2, p['center'], p['major_radius'], p['radius'],
                p['axis'], color, name, show)

    elif ptype == 'cylinder':
        _add_capsule(fig2, p['center'], p['radius'],
                    p['axis'], p['half_height'], color, name, show)

    elif ptype == 'cone':
        _add_capsule(fig2, p['center'], p['radius'],
                    p['axis'], p['half_height'], color, name, show)        

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
# logging.info("What to look for:")
# logging.info("  - Ear regions on bunny should show green capsules")
# logging.info("  - Torso/limb columns on Lucy should show green capsules")
# logging.info("  - Flat regions (robes, base) may show orange boxes")
# logging.info("  - Round body regions should stay blue spheres")

# ---------------------------------------------------------------------------
# SDF evaluators for each primitive type
# ---------------------------------------------------------------------------
# All primitives are in normalized mesh coordinates (unit bounding box).
# Each function takes (points: (n,3)) and returns (n,) signed distances.
# We only need the zero set to be correct — not true SDFs everywhere.

def _sdf_sphere(points, center, radius, **kwargs):
    return np.linalg.norm(points - center, axis=1) - radius


def _sdf_capsule(points, center, radius, axis, half_height, **kwargs):
    # Project onto axis, clamp to segment, measure distance to nearest point
    axis     = axis / (np.linalg.norm(axis) + 1e-10)
    pa       = points - (center - axis * half_height)
    ba       = axis * 2.0 * half_height
    t        = np.clip(np.dot(pa, ba) / (np.dot(ba, ba) + 1e-10), 0.0, 1.0)
    nearest  = (center - axis * half_height) + t[:, None] * ba
    return np.linalg.norm(points - nearest, axis=1) - radius


def _sdf_cylinder(points, center, radius, axis, half_height, **kwargs):
    axis     = axis / (np.linalg.norm(axis) + 1e-10)
    delta    = points - center
    along    = np.dot(delta, axis)
    radial   = np.linalg.norm(delta - along[:, None] * axis, axis=1)
    d_radial = radial - radius
    d_axial  = np.abs(along) - half_height
    # Inigo Quilez cylinder SDF
    return (np.sqrt(np.maximum(d_radial, 0)**2 + np.maximum(d_axial, 0)**2)
            + np.minimum(np.maximum(d_radial, d_axial), 0))


def _sdf_box(points, center, axes, half_extents, **kwargs):
    # Transform points into box local frame
    delta  = points - center
    # Project onto each box axis
    q      = np.column_stack([
        np.abs(delta @ axes[0]) - half_extents[0],
        np.abs(delta @ axes[1]) - half_extents[1],
        np.abs(delta @ axes[2]) - half_extents[2],
    ])
    return (np.linalg.norm(np.maximum(q, 0), axis=1)
            + np.minimum(np.max(q, axis=1), 0))


def _sdf_torus(points, center, major_radius, radius, axis, **kwargs):
    axis     = axis / (np.linalg.norm(axis) + 1e-10)
    delta    = points - center
    along    = np.dot(delta, axis)
    radial   = np.linalg.norm(delta - along[:, None] * axis, axis=1)
    # Distance to the torus tube center circle
    return np.sqrt((radial - major_radius)**2 + along**2) - radius


def _sdf_cone(points, center, axis, half_height, r_bottom, r_top, **kwargs):
    # Capped cone — Inigo Quilez formulation
    axis     = axis / (np.linalg.norm(axis) + 1e-10)
    a        = center - axis * half_height   # bottom center
    b        = center + axis * half_height   # top center
    pa       = points - a
    ba       = b - a
    baba     = np.dot(ba, ba)
    paba     = np.dot(pa, ba) / (baba + 1e-10)
    t        = np.clip(paba, 0.0, 1.0)
    # Radius at each projected point along axis
    r        = r_bottom + (r_top - r_bottom) * t
    radial   = np.linalg.norm(pa - t[:, None] * ba, axis=1)
    return radial - r


# Dispatch table
_SDF_DISPATCH = {
    'sphere':       _sdf_sphere,
    'capsule':      _sdf_capsule,
    'cylinder':     _sdf_cylinder,
    'box':          _sdf_box,
    'torus':        _sdf_torus,
    'capped_torus': _sdf_torus,   # same evaluator, uses major_radius
    'cone':         _sdf_cone,
}


def eval_primitive(points, primitive):
    """Evaluate SDF of a single primitive at given points. Returns (n,) array."""
    fn = _SDF_DISPATCH.get(primitive['type'])
    if fn is None:
        raise ValueError(f"Unknown primitive type: {primitive['type']}")
    return fn(points, **primitive)


def eval_union(points, primitives, k_smooth=32.0):
    """
    Evaluate smooth union (smin) of all primitives at given points.
    k_smooth: smoothness parameter — higher = sharper joins.
    Returns (n,) array of SDF values.
    """
    if not primitives:
        return np.full(len(points), np.inf)
    # Start with first primitive
    result = eval_primitive(points, primitives[0])
    for prim in primitives[1:]:
        d2  = eval_primitive(points, prim)
        # Inigo Quilez polynomial smooth minimum
        h   = np.clip(0.5 + 0.5 * (d2 - result) / (1.0 / k_smooth), 0.0, 1.0)
        result = d2 * (1.0 - h) + result * h - (1.0 / k_smooth) * h * (1.0 - h)
    return result


def coverage_mask(points, primitives, epsilon=0.02):
    """
    Returns boolean mask: True where smin_union(x) <= epsilon.
    These are the 'explained' surface points.
    """
    if not primitives:
        return np.zeros(len(points), dtype=bool)
    sdf = eval_union(points, primitives)
    return np.abs(sdf) <= epsilon


def coverage_fraction(points, primitives, epsilon=0.02):
    """Fraction of surface points explained by the current primitive set."""
    return float(coverage_mask(points, primitives, epsilon).mean())

# ---------------------------------------------------------------------------
# Step 2 coverage report
# ---------------------------------------------------------------------------
EPSILON = 0.02

cov = coverage_fraction(surface_pts, promoted_primitives, EPSILON)
mask = coverage_mask(surface_pts, promoted_primitives, EPSILON)
logging.info(f"Step 2 coverage: {cov*100:.1f}% of surface points explained "
             f"({mask.sum()} / {len(surface_pts)})")

# ---------------------------------------------------------------------------
# Step 3: GPU-accelerated voxel beam search
# ---------------------------------------------------------------------------
# Instead of clustering residual surface points (which gives no inside/outside
# guarantee), we voxelize the mesh interior and search for primitives that
# explain unexplained interior voxels.
#
# All heavy computation runs on GPU via PyTorch.
# Surface point coverage (ground truth) still uses the existing numpy evaluators.

import torch

logging.info("=" * 60)
logging.info("Step 3: GPU voxel beam search")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Device: {DEVICE}")

# --- Configuration ---
VOXEL_RES            = 128      # voxel grid resolution per axis (64^3 = 262k voxels)
                                # increase to 128 or 256 for higher fidelity
BEAM_COVERAGE_TARGET = 0.95    # stop when this fraction of surface pts explained
BEAM_MAX_PRIMITIVES  = 60      # hard cap on primitives added by beam search
BEAM_MIN_GAIN        = 0.001   # stop if best candidate explains < 0.3% new surface pts
BEAM_EPSILON         = EPSILON # same epsilon as coverage (0.02)
MIN_DEPTH            = EPSILON / 2  # ignore voxels shallower than this from surface
                                    # filters gap voxels between medial spheres and mesh

# ---------------------------------------------------------------------------
# 3a: Build voxel grid
# ---------------------------------------------------------------------------
# Generate a uniform grid of points inside the mesh bounding box,
# filter to interior, compute surface distance for each.

logging.info(f"Building {VOXEL_RES}^3 voxel grid...")

# Grid coordinates in normalized mesh space
bb_min_v = mesh.bounds[0]
bb_max_v = mesh.bounds[1]

xs = np.linspace(bb_min_v[0], bb_max_v[0], VOXEL_RES)
ys = np.linspace(bb_min_v[1], bb_max_v[1], VOXEL_RES)
zs = np.linspace(bb_min_v[2], bb_max_v[2], VOXEL_RES)

gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
grid_pts   = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)  # (N, 3)

logging.info(f"Grid points: {len(grid_pts)} total, testing interior...")

# Inside test — reuse existing ray caster
voxel_inside = points_inside_mesh(mesh, grid_pts)
interior_voxels = grid_pts[voxel_inside]   # (M, 3) only interior points
logging.info(f"Interior voxels: {len(interior_voxels)} "
             f"({100*len(interior_voxels)/len(grid_pts):.1f}% fill rate)")

# Surface distance for each interior voxel
# Reuse the existing surface KD-tree from step 1
voxel_surf_dist, _ = kd_tree.query(interior_voxels, k=1, workers=-1)
logging.info(f"Voxel surface distance range: "
             f"{voxel_surf_dist.min():.4f} — {voxel_surf_dist.max():.4f}")

# Move everything to GPU
voxels_gpu     = torch.tensor(interior_voxels, dtype=torch.float32, device=DEVICE)
surf_dist_gpu  = torch.tensor(voxel_surf_dist, dtype=torch.float32, device=DEVICE)
surface_pts_gpu = torch.tensor(surface_pts,    dtype=torch.float32, device=DEVICE)

logging.info(f"Tensors on {DEVICE}: "
             f"voxels={voxels_gpu.shape}, surface={surface_pts_gpu.shape}")

# ---------------------------------------------------------------------------
# 3b: GPU SDF evaluators
# ---------------------------------------------------------------------------
# Mirror the numpy SDFs in PyTorch. Same math, tensor ops throughout.
# All functions take (points: Tensor[N,3]) and return Tensor[N].

def _sdf_sphere_gpu(points, center, radius, **kwargs):
    c = torch.tensor(center, dtype=torch.float32, device=DEVICE)
    return torch.norm(points - c, dim=1) - radius


def _sdf_capsule_gpu(points, center, radius, axis, half_height, **kwargs):
    ax  = torch.tensor(axis / (np.linalg.norm(axis) + 1e-10),
                       dtype=torch.float32, device=DEVICE)
    c   = torch.tensor(center, dtype=torch.float32, device=DEVICE)
    a   = c - ax * half_height
    b_  = ax * 2.0 * half_height
    pa  = points - a
    t   = torch.clamp((pa @ b_) / (b_ @ b_ + 1e-10), 0.0, 1.0)
    nearest = a + t.unsqueeze(1) * b_
    return torch.norm(points - nearest, dim=1) - radius


def _sdf_cylinder_gpu(points, center, radius, axis, half_height, **kwargs):
    ax    = torch.tensor(axis / (np.linalg.norm(axis) + 1e-10),
                         dtype=torch.float32, device=DEVICE)
    c     = torch.tensor(center, dtype=torch.float32, device=DEVICE)
    delta = points - c
    along = delta @ ax
    radial = torch.norm(delta - along.unsqueeze(1) * ax, dim=1)
    d_rad  = radial - radius
    d_ax   = torch.abs(along) - half_height
    return (torch.sqrt(torch.clamp(d_rad, 0)**2 + torch.clamp(d_ax, 0)**2)
            + torch.min(torch.max(d_rad, d_ax),
                        torch.zeros_like(d_rad)))


def _sdf_box_gpu(points, center, axes, half_extents, **kwargs):
    c   = torch.tensor(center,      dtype=torch.float32, device=DEVICE)
    ax  = torch.tensor(axes,        dtype=torch.float32, device=DEVICE)  # (3,3)
    he  = torch.tensor(half_extents,dtype=torch.float32, device=DEVICE)  # (3,)
    delta = points - c
    q   = torch.stack([
        torch.abs(delta @ ax[0]) - he[0],
        torch.abs(delta @ ax[1]) - he[1],
        torch.abs(delta @ ax[2]) - he[2],
    ], dim=1)
    return (torch.norm(torch.clamp(q, min=0), dim=1)
            + torch.clamp(torch.max(q, dim=1).values, max=0))


def _sdf_torus_gpu(points, center, major_radius, radius, axis, **kwargs):
    ax    = torch.tensor(axis / (np.linalg.norm(axis) + 1e-10),
                         dtype=torch.float32, device=DEVICE)
    c     = torch.tensor(center, dtype=torch.float32, device=DEVICE)
    delta = points - c
    along = delta @ ax
    radial = torch.norm(delta - along.unsqueeze(1) * ax, dim=1)
    return torch.sqrt((radial - major_radius)**2 + along**2) - radius


def _sdf_cone_gpu(points, center, axis, half_height, r_bottom, r_top, **kwargs):
    ax    = torch.tensor(axis / (np.linalg.norm(axis) + 1e-10),
                         dtype=torch.float32, device=DEVICE)
    c     = torch.tensor(center, dtype=torch.float32, device=DEVICE)
    a     = c - ax * half_height
    b_    = ax * 2.0 * half_height
    pa    = points - a
    baba  = b_ @ b_
    t     = torch.clamp((pa @ b_) / (baba + 1e-10), 0.0, 1.0)
    r     = r_bottom + (r_top - r_bottom) * t
    radial = torch.norm(pa - t.unsqueeze(1) * b_, dim=1)
    return radial - r


_SDF_GPU_DISPATCH = {
    'sphere':       _sdf_sphere_gpu,
    'capsule':      _sdf_capsule_gpu,
    'cylinder':     _sdf_cylinder_gpu,
    'box':          _sdf_box_gpu,
    'torus':        _sdf_torus_gpu,
    'capped_torus': _sdf_torus_gpu,
    'cone':         _sdf_cone_gpu,
}


def eval_primitive_gpu(points, primitive):
    """Evaluate SDF of a single primitive on GPU. Returns Tensor[N]."""
    fn = _SDF_GPU_DISPATCH.get(primitive['type'])
    if fn is None:
        raise ValueError(f"Unknown primitive type: {primitive['type']}")
    return fn(points, **primitive)


def eval_union_gpu(points, primitives, k_smooth=32.0):
    """Smooth union of all primitives on GPU. Returns Tensor[N]."""
    if not primitives:
        return torch.full((len(points),), float('inf'), device=DEVICE)
    result = eval_primitive_gpu(points, primitives[0])
    for prim in primitives[1:]:
        d2 = eval_primitive_gpu(points, prim)
        h  = torch.clamp(0.5 + 0.5 * (d2 - result) * k_smooth, 0.0, 1.0)
        result = d2 * (1.0 - h) + result * h - h * (1.0 - h) / k_smooth
    return result


def coverage_mask_gpu(points, primitives, epsilon):
    """Boolean mask of explained points on GPU."""
    if not primitives:
        return torch.zeros(len(points), dtype=torch.bool, device=DEVICE)
    return torch.abs(eval_union_gpu(points, primitives)) <= epsilon


# ---------------------------------------------------------------------------
# 3c: Initialize coverage from step 2 primitives
# ---------------------------------------------------------------------------

logging.info("Computing initial GPU coverage from step 2 primitives...")

current_primitives  = list(promoted_primitives)
voxel_mask_gpu      = coverage_mask_gpu(voxels_gpu, current_primitives, BEAM_EPSILON)
surface_mask_gpu    = coverage_mask_gpu(surface_pts_gpu, current_primitives, BEAM_EPSILON)
current_coverage    = float(surface_mask_gpu.float().mean().item())

logging.info(f"Starting voxel coverage:   "
             f"{float(voxel_mask_gpu.float().mean().item())*100:.1f}%")
logging.info(f"Starting surface coverage: {current_coverage*100:.1f}%")
logging.info(f"Target:                    {BEAM_COVERAGE_TARGET*100:.1f}%")

# ---------------------------------------------------------------------------
# 3d: Greedy voxel beam search loop
# ---------------------------------------------------------------------------

def _propose_capsule(center, radius, cand_pts, cand_dist, seed_kd, neighborhood_k=30):
    """
    Given a seed center, find neighboring unexplained voxels and fit a capsule.
    Returns capsule primitive dict or None if neighborhood is too small.
    """
    # radius here is the seed's surface distance — the safe inscribed sphere radius
    nbr_idx = seed_kd.query_ball_point(center, r=radius * 1.5)
    if len(nbr_idx) < 4:
        return None

    nbr_pts = cand_pts[nbr_idx]

    # PCA axis only — center stays at seed
    centered = nbr_pts - center
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    axis = Vt[0]

    # Project neighbors onto axis relative to seed center
    projections = centered @ axis
    proj_min = projections.min()
    proj_max = projections.max()

    # Half-height capped at seed surface distance — same guarantee as sphere radius
    half_height = float(np.clip((proj_max - proj_min) / 2.0, 0.0, radius * 0.9))
    cap_radius  = float(np.clip(radius * 0.85, 0.01, MAX_BEAM_RADIUS))

    if half_height < cap_radius * 1.1:
        return None  # not elongated enough to beat a sphere

    return {
        'type':        'capsule',
        'center':      center,       # ALWAYS the seed point
        'radius':      cap_radius,
        'axis':        axis,
        'half_height': half_height,
        'axes':        None,
    }


n_added = 0
MAX_BEAM_RADIUS = float(max(medial_radii))

while current_coverage < BEAM_COVERAGE_TARGET and n_added < BEAM_MAX_PRIMITIVES:

    # --- Find candidate seed voxels ---
    # Unexplained + deep enough from surface to avoid gap microprimitives
    deep_enough   = surf_dist_gpu > MIN_DEPTH
    unexplained   = ~voxel_mask_gpu
    candidate_mask = unexplained & deep_enough

    n_candidates = int(candidate_mask.sum().item())
    logging.info(f"  Iteration {n_added+1}: "
                 f"{int(unexplained.sum().item())} unexplained voxels, "
                 f"{n_candidates} deep candidates, "
                 f"surface coverage {current_coverage*100:.1f}%")

    if n_candidates < 5:
        logging.info("  Too few candidate voxels, stopping.")
        break

    # Pull candidate voxels and their surface distances to CPU for seed finding
    cand_pts_gpu  = voxels_gpu[candidate_mask]
    cand_dist_gpu = surf_dist_gpu[candidate_mask]

    cand_pts  = cand_pts_gpu.cpu().numpy()
    cand_dist = cand_dist_gpu.cpu().numpy()

    # --- Find local maxima of surface distance among candidate voxels ---
    # Same logic as step 1: a voxel is a seed if its distance > all k neighbors
    LOCAL_SEED_K = min(20, len(cand_pts) - 1)
    if LOCAL_SEED_K < 1:
        logging.info("  Not enough candidates for neighbor search, stopping.")
        break

    seed_kd   = scipy.spatial.cKDTree(cand_pts)
    _, nbr_idx = seed_kd.query(cand_pts, k=LOCAL_SEED_K + 1, workers=-1)
    nbr_dist  = cand_dist[nbr_idx[:, 1:]]
    is_seed   = np.all(cand_dist[:, None] > nbr_dist, axis=1)

    seed_pts  = cand_pts[is_seed]
    seed_dist = cand_dist[is_seed]

    logging.info(f"  Seeds found: {len(seed_pts)}")

    if len(seed_pts) == 0:
        logging.info("  No seeds found, stopping.")
        break

    # --- For each seed, propose a sphere candidate ---
    # Sphere first: center=seed, radius=surface_distance (like step 1).
    # We score all seeds on GPU in one batched pass, pick the best.
    seed_pts_gpu  = torch.tensor(seed_pts,  dtype=torch.float32, device=DEVICE)
    seed_dist_gpu = torch.tensor(seed_dist, dtype=torch.float32, device=DEVICE)

    # Clamp radius to MAX_BEAM_RADIUS
    seed_radii_gpu = torch.clamp(seed_dist_gpu * 0.9, max=MAX_BEAM_RADIUS)

    # Batched sphere SDF on surface points: (N_seeds, N_surface)
    # For each seed, compute how many new surface points it would explain
    # surface_pts_gpu: (N_surf, 3), seed_pts_gpu: (N_seeds, 3)
    # dist[i,j] = ||surface_pts[j] - seed[i]||
    # Do in chunks if needed to avoid OOM
    CHUNK = 256  # slightly smaller to leave room for capsule scoring

    best_gain      = 0.0
    best_candidate = None

    current_surf_mask_gpu = coverage_mask_gpu(
        surface_pts_gpu, current_primitives, BEAM_EPSILON
    )
    unexplained_surf_gpu = ~current_surf_mask_gpu  # (N_surf,) bool

    # --- Score sphere candidates in batched GPU pass ---
    for chunk_start in range(0, len(seed_pts_gpu), CHUNK):
        chunk_end   = min(chunk_start + CHUNK, len(seed_pts_gpu))
        seeds_chunk = seed_pts_gpu[chunk_start:chunk_end]
        radii_chunk = seed_radii_gpu[chunk_start:chunk_end]

        diff    = surface_pts_gpu.unsqueeze(0) - seeds_chunk.unsqueeze(1)
        dists   = torch.norm(diff, dim=2)
        sdf     = dists - radii_chunk.unsqueeze(1)
        explains = torch.abs(sdf) <= BEAM_EPSILON
        new_exp  = explains & unexplained_surf_gpu.unsqueeze(0)
        gains    = new_exp.float().sum(dim=1) / len(surface_pts)

        best_in_chunk     = int(torch.argmax(gains).item())
        best_gain_chunk   = float(gains[best_in_chunk].item())

        if best_gain_chunk > best_gain:
            best_gain = best_gain_chunk
            global_idx = chunk_start + best_in_chunk
            best_candidate = {
                'type':        'sphere',
                'center':      seed_pts[global_idx],
                'radius':      float(seed_radii_gpu[global_idx].item()),
                'axis':        None,
                'half_height': None,
                'axes':        None,
            }

    # --- Score capsule candidates (CPU fit, GPU score) ---
    for i, (center, radius) in enumerate(zip(seed_pts, seed_dist)):
        cap = _propose_capsule(center, radius, cand_pts, cand_dist, seed_kd)
        if cap is None:
            continue
        
        cap_sdf  = eval_primitive_gpu(surface_pts_gpu, cap)
        explains = torch.abs(cap_sdf) <= BEAM_EPSILON
        new_exp  = explains & unexplained_surf_gpu
        gain     = float(new_exp.float().sum().item()) / len(surface_pts)
        
        if gain > best_gain:
            best_gain      = gain
            best_candidate = cap

    if best_candidate is None or best_gain < BEAM_MIN_GAIN:
        logging.info(f"  Best gain {best_gain*100:.2f}% below threshold, stopping.")
        break

    # Dedup: skip if best candidate is nearly identical to last committed
    if len(current_primitives) > 0 and best_candidate is not None:
        last = current_primitives[-1]
        if (last['type'] == best_candidate['type'] == 'capsule' and
            np.linalg.norm(np.array(last['center']) - np.array(best_candidate['center'])) < BEAM_EPSILON * 2):
            logging.info("  Dedup: identical capsule, stopping.")
            break

    # --- Commit best candidate ---
    current_primitives.append(best_candidate)

    # Update masks on GPU
    new_sdf_vox  = eval_primitive_gpu(voxels_gpu,      best_candidate)
    new_sdf_surf = eval_primitive_gpu(surface_pts_gpu, best_candidate)
    voxel_mask_gpu   = voxel_mask_gpu   | (torch.abs(new_sdf_vox)  <= BEAM_EPSILON)
    surface_mask_gpu = surface_mask_gpu | (torch.abs(new_sdf_surf) <= BEAM_EPSILON)
    current_coverage = float(surface_mask_gpu.float().mean().item())

    logging.info(f"  Committed {best_candidate['type']} r={best_candidate['radius']:.3f} "
                 f"(gain={best_gain*100:.1f}%, "
                 f"coverage now {current_coverage*100:.1f}%)")
    n_added += 1

logging.info(f"Beam search done: added {n_added} primitives, "
             f"final coverage {current_coverage*100:.1f}%")
logging.info(f"Total primitives: {len(current_primitives)}")

# ---------------------------------------------------------------------------
# Step 3 visualization
# ---------------------------------------------------------------------------

def _visualize_beam_search(residual_pts, cluster_labels, mesh, step_num, primitives):
    fig = go.Figure()

    # Mesh
    fig.add_trace(go.Mesh3d(
        x=mesh.vertices[:,0], y=mesh.vertices[:,1], z=mesh.vertices[:,2],
        i=mesh.faces[:,0], j=mesh.faces[:,1], k=mesh.faces[:,2],
        color='pink', opacity=0.15, name='mesh', showlegend=True,
    ))

    # Residual points colored by cluster
    cluster_colors = [
        '#e6194b','#3cb44b','#ffe119','#4363d8','#f58231',
        '#911eb4','#42d4f4','#f032e6','#bfef45','#fabed4',
        '#469990','#dcbeff','#9A6324','#fffac8','#800000',
        '#aaffc3','#808000','#ffd8b1','#000075','#a9a9a9',
    ]
    for label in sorted(set(cluster_labels)):
        mask  = cluster_labels == label
        pts   = residual_pts[mask]
        color = '#888888' if label == -1 else cluster_colors[label % len(cluster_colors)]
        name  = 'noise' if label == -1 else f'cluster_{label}'
        fig.add_trace(go.Scatter3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            mode='markers',
            marker=dict(size=2, color=color, opacity=0.6),
            name=name, showlegend=True,
        ))

    # Primitives
    type_colors = {
        'sphere': 'blue', 'capsule': 'green', 'cylinder': 'cyan',
        'cone': 'purple', 'box': 'orange', 'torus': 'magenta',
        'capped_torus': 'pink',
    }
    type_seen = set()
    for pi, p in enumerate(primitives):
        ptype = p['type']
        color = type_colors.get(ptype, 'gray')
        name  = f"{ptype}_{pi}"
        show  = ptype not in type_seen
        type_seen.add(ptype)
        if ptype == 'sphere':
            _add_sphere(fig, p['center'], p['radius'], color, name, show)
        elif ptype in ('capsule', 'cylinder', 'cone'):
            _add_capsule(fig, p['center'], p['radius'],
                         p['axis'], p['half_height'], color, name, show)
        elif ptype == 'box':
            _add_box(fig, p['center'], p['radius'], p['axis'],
                     p['half_height'], p['axes'], color, name, show,
                     half_extents=p.get('half_extents'))
        elif ptype in ('torus', 'capped_torus'):
            _add_torus(fig, p['center'], p['major_radius'], p['radius'],
                       p['axis'], color, name, show)

    n_types = {}
    for p in primitives:
        n_types[p['type']] = n_types.get(p['type'], 0) + 1
    title = (f"Step 3 iteration {step_num}: "
             + ", ".join(f"{v} {k}s" for k, v in n_types.items())
             + f" | coverage {current_coverage*100:.1f}%")

    fig.update_layout(title=title, scene=dict(aspectmode='data'),
                      width=1100, height=850)
    fname = 'beam_search_final.html'
    fig.write_html(fname)
    logging.info(f"Saved: {fname}")


# Residual surface points for visualization
residual_surf     = surface_pts[~surface_mask_gpu.cpu().numpy()]
# Simple distance-based labeling for visualization — no DBSCAN needed
# Just show the raw residual points, all same label
residual_labels   = np.zeros(len(residual_surf), dtype=int)

_visualize_beam_search(residual_surf, residual_labels,
                       mesh, n_added, current_primitives)