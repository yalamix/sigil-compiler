# sigil/pipeline/geometry_pipeline.py

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import trimesh

from sigil.geometry.segmentation import (
    segment_mesh,
    find_adjacent_patches,
    get_patch_faces,
)
from sigil.geometry.scalar_field import sample_scalar_field
from sigil.geometry.gpr import fit_gpr, predict, generate_query_points
from sigil.geometry.sr.base import Equation
from sigil.geometry.sr.sparse_regression import SparseRegressionBackend
from sigil.geometry.sr.pysr_backend import PySRBackend
from sigil.geometry.merge import blend_polynomial, residual_correction_merge, blend_partition_of_unity


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    # Segmentation
    n_seeds:               Optional[int]  = None
    use_voronoi:           bool           = False

    # Scalar field sampling
    samples_per_unit_area: int            = 5000
    max_gpr_points:        int            = 300
    gpr_resolution:        int            = 20

    # SR -- leaf patches
    sr_degree:             int            = 4
    sr_refine_steps:       int            = 500
    sr_refine_lr:          float          = 1e-3

    # SR -- merge patches
    pysr_niterations:      int            = 25
    pysr_populations:      int            = 15
    pysr_root_multiplier:  int            = 4

    # Merge
    merge_strategy:        str            = 'partition_of_unity'
    smin_k:                float          = 0.1
    use_pysr_at_leaves:    bool           = True   # sparse regression (already done)
    use_pysr_at_merges:    bool           = True   # sparse regression for intermediate nodes
    use_pysr_at_root:      bool           = True    # PySR with root_multiplier * niterations    
    blend_sigma: Optional[float]          = None

    # Post-merge refinement (mandatory)
    merge_refine_steps:    int            = 1000
    merge_refine_lr:       float          = 1e-3

    # Parallelism
    n_workers:             Optional[int]  = None   # None = os.cpu_count()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LeafResult:
    patch_idx:      int
    patch_vertices: np.ndarray
    equation:       Equation
    X_query:        np.ndarray   # (N, 3) query points for this patch
    y_query:        np.ndarray   # (N,)   GPR values for this patch
    seed_position:  np.ndarray


@dataclass
class MergeNode:
    node_id:       int
    patch_indices: frozenset     # original patch indices contained
    equation:      Equation
    X_data:        np.ndarray   # (N, 3) -- all query points accumulated
    y_data:        np.ndarray   # (N,)   -- all scalar field values accumulated
    centroid:      np.ndarray


# ---------------------------------------------------------------------------
# Overlap data utilities
# ---------------------------------------------------------------------------

def _get_overlap_data(node_a, node_b, original_overlap_data):
    """
    Retrieve combined overlap (X, y) between two MergeNodes by looking up
    all original patch pair overlaps between their constituent indices.

    original_overlap_data: dict {(i,j): (X_overlap, y_overlap)}
                           canonical key: i < j always

    returns: (X_overlap, y_overlap) or (None, None) if not adjacent
    """
    X_parts = []
    y_parts = []

    for i in node_a.patch_indices:
        for j in node_b.patch_indices:
            key = (min(i, j), max(i, j))
            if key in original_overlap_data:
                X_ov, y_ov = original_overlap_data[key]
                X_parts.append(X_ov)
                y_parts.append(y_ov)

    if not X_parts:
        return None, None

    return np.vstack(X_parts), np.concatenate(y_parts)


# ---------------------------------------------------------------------------
# Leaf pass -- single patch (module-level for pickling)
# ---------------------------------------------------------------------------

def _process_leaf_patch(args):
    """
    Process a single leaf patch end-to-end.
    Module-level function so ProcessPoolExecutor can pickle it.

    args: tuple (patch_idx, patch_vertices, neighbor_indices,
                 adjacency, mesh_vertices, mesh_faces,
                 mesh_vertex_normals, mesh_face_normals,
                 mesh_area_faces, config)

    We pass mesh arrays separately rather than the trimesh object
    because trimesh is large and we want minimal pickle overhead.
    We reconstruct the trimesh object inside the worker.

    returns: (LeafResult, overlap_data_contributions)
             overlap_data_contributions: dict {(i,j): (X_ov, y_ov)}
             only contains pairs where i == patch_idx (lower index owns it)
    """
    (patch_idx, patch_vertices, neighbor_indices,
     adjacency, mesh_vertices, mesh_faces,
     mesh_vertex_normals, mesh_face_normals,
     mesh_area_faces, seed_position, config) = args

    # Reconstruct trimesh inside worker
    mesh = trimesh.Trimesh(
        vertices=mesh_vertices,
        faces=mesh_faces,
        vertex_normals=mesh_vertex_normals,
        process=False
    )
    # mesh._cache['face_normals']  = mesh_face_normals
    # mesh._cache['area_faces']    = mesh_area_faces

    logging.info(f"Leaf {patch_idx}: sampling scalar field")

    # Step 1-4: sample -> GPR -> query grid -> evaluate
    X_train, y_train = sample_scalar_field(
        patch_vertices, mesh,
        samples_per_unit_area = config.samples_per_unit_area,
        max_gpr_points        = config.max_gpr_points,
    )
    gpr_model = fit_gpr(X_train, y_train)
    X_query   = generate_query_points(
        patch_vertices, mesh,
        resolution = config.gpr_resolution
    )
    y_query   = predict(gpr_model, X_query)

    # Step 5: sparse regression
    logging.info(f"Leaf {patch_idx}: fitting sparse regression")
    backend = SparseRegressionBackend(
        degree        = config.sr_degree,
        n_refine_steps= config.sr_refine_steps,
        lr            = config.sr_refine_lr,
    )
    equation = backend.fit(X_query, y_query)

    logging.info(
        f"Leaf {patch_idx}: equation fitted, "
        f"rmse={equation.rmse:.6f}, "
        f"expr={equation.sympy_expr}"
    )

    leaf = LeafResult(
        patch_idx      = patch_idx,
        patch_vertices = patch_vertices,
        equation       = equation,
        X_query        = X_query,
        y_query        = y_query,
        seed_position  = seed_position
    )

    # Step 6: pre-compute overlap GPR for all neighbors where
    # patch_idx < neighbor_idx (lower index owns the overlap)
    overlap_contributions = {}

    for neighbor_idx in neighbor_indices:
        if patch_idx >= neighbor_idx:
            continue   # higher index defers to lower index

        key             = (patch_idx, neighbor_idx)
        overlap_verts   = adjacency[key]

        if len(overlap_verts) < 5:
            logging.debug(
                f"Overlap ({patch_idx},{neighbor_idx}) too small "
                f"({len(overlap_verts)} verts) -- skipping"
            )
            continue

        logging.info(
            f"Leaf {patch_idx}: computing overlap GPR "
            f"with patch {neighbor_idx} "
            f"({len(overlap_verts)} verts)"
        )

        X_ov_train, y_ov_train = sample_scalar_field(
            overlap_verts, mesh,
            samples_per_unit_area = config.samples_per_unit_area,
            max_gpr_points        = config.max_gpr_points,
        )
        overlap_gpr   = fit_gpr(X_ov_train, y_ov_train)
        X_ov_query    = generate_query_points(
            overlap_verts, mesh,
            resolution = config.gpr_resolution
        )
        y_ov_query    = predict(overlap_gpr, X_ov_query)

        overlap_contributions[key] = (X_ov_query, y_ov_query)

    return leaf, overlap_contributions


def _process_all_leaves(patches, adjacency, seed_positions, mesh, config):
    """
    Process all leaf patches in parallel via ProcessPoolExecutor.

    returns:
        leaf_results:         list[LeafResult] ordered by patch_idx
        original_overlap_data: dict {(i,j): (X_overlap, y_overlap)}
    """
    n_workers = config.n_workers or os.cpu_count()

    # Pre-extract mesh arrays for pickling
    mesh_vertices      = mesh.vertices
    mesh_faces         = mesh.faces
    mesh_vertex_normals= mesh.vertex_normals
    mesh_face_normals  = mesh.face_normals
    mesh_area_faces    = mesh.area_faces

    # Build neighbor index lists
    neighbor_map = {}
    for i in range(len(patches)):
        neighbor_map[i] = []
    for (i, j) in adjacency.keys():
        neighbor_map[i].append(j)
        neighbor_map[j].append(i)

    # Build args list
    args_list = []
    for patch_idx, patch_vertices in enumerate(patches):
        args_list.append((
            patch_idx,
            patch_vertices,
            neighbor_map[patch_idx],
            adjacency,
            mesh_vertices,
            mesh_faces,
            mesh_vertex_normals,
            mesh_face_normals,
            mesh_area_faces,
            seed_positions[patch_idx],  
            config,
        ))

    leaf_results          = [None] * len(patches)
    original_overlap_data = {}

    logging.info(
        f"Processing {len(patches)} leaf patches "
        f"with {n_workers} workers"
    )

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_process_leaf_patch, args): args[0]
            for args in args_list
        }
        for future in as_completed(futures):
            patch_idx = futures[future]
            try:
                leaf, overlap_contributions = future.result()
                leaf_results[patch_idx]     = leaf
                original_overlap_data.update(overlap_contributions)
                logging.info(f"Leaf {patch_idx} complete")
            except Exception as e:
                logging.error(
                    f"Leaf {patch_idx} failed: "
                    f"{type(e).__name__}: {e}"
                )
                raise

    return leaf_results, original_overlap_data


# ---------------------------------------------------------------------------
# Merge tree construction
# ---------------------------------------------------------------------------

def _build_merge_pairs(nodes, original_overlap_data):
    """
    Greedy pairing by overlap size for one merge level.

    nodes: list[MergeNode]

    returns:
        pairs:    list of (node_a, node_b, X_overlap, y_overlap)
        unpaired: list[MergeNode] -- carry forward to next level
    """
    candidates = []

    for i, na in enumerate(nodes):
        for j, nb in enumerate(nodes):
            if j <= i:
                continue
            X_ov, y_ov = _get_overlap_data(na, nb, original_overlap_data)
            if X_ov is not None:
                candidates.append((len(X_ov), i, j, X_ov, y_ov))

    # Sort by overlap size descending -- largest overlap paired first
    candidates.sort(reverse=True)

    paired   = set()
    pairs    = []
    unpaired = []

    for size, i, j, X_ov, y_ov in candidates:
        if i in paired or j in paired:
            continue
        pairs.append((nodes[i], nodes[j], X_ov, y_ov))
        paired.add(i)
        paired.add(j)

    unpaired = [n for idx, n in enumerate(nodes) if idx not in paired]

    logging.info(
        f"Merge level: {len(pairs)} pairs, "
        f"{len(unpaired)} unpaired (carry forward)"
    )

    return pairs, unpaired


# ---------------------------------------------------------------------------
# Merge pass -- single pair (module-level for pickling)
# ---------------------------------------------------------------------------

def _merge_pair(args):
    """
    Merge two MergeNodes into one.
    Module-level for ProcessPoolExecutor pickling.

    args: tuple (node_a, node_b, X_overlap, y_overlap,
                 new_node_id, is_root, config)

    returns: MergeNode
    """
    import sys
    sys.setrecursionlimit(100000)    
    node_a, node_b, X_overlap, y_overlap, new_node_id, is_root, config = args

    logging.info(
        f"Merging nodes {node_a.node_id} and {node_b.node_id} "
        f"(patches {set(node_a.patch_indices)} + "
        f"{set(node_b.patch_indices)}, "
        f"is_root={is_root})"
    )

    # Combined dataset: both nodes' data plus overlap
    X_combined = np.vstack([node_a.X_data, node_b.X_data, X_overlap])
    y_combined = np.concatenate([node_a.y_data, node_b.y_data, y_overlap])

    # Blend
    if config.merge_strategy == 'polynomial':
        eq_blend = blend_polynomial(
            node_a.equation, node_b.equation,
            X_overlap, y_overlap,
            degree = 4,
        )
    elif config.merge_strategy == 'partition_of_unity':    
        eq_blend = blend_partition_of_unity(node_a.equation, node_b.equation, 
                                            node_a.centroid, node_b.centroid, sigma=config.blend_sigma)

    # PySR backend -- 4x iterations at root
    niterations = config.pysr_niterations
    if is_root:
        niterations = niterations * config.pysr_root_multiplier
        logging.info(f"Root merge: PySR iterations = {niterations}")

    if is_root and config.use_pysr_at_root:
        backend = PySRBackend(
            niterations  = niterations,
            populations  = config.pysr_populations,
            batch_size   = 2000
        )
    elif not is_root and config.use_pysr_at_merges:
        backend = PySRBackend(
            niterations  = niterations,
            populations  = config.pysr_populations,
        )
    else:
        backend = SparseRegressionBackend(
            degree         = config.sr_degree,
            n_refine_steps = config.merge_refine_steps,
            lr             = config.merge_refine_lr,
        )               

    # Residual correction + mandatory refinement
    eq_merged = residual_correction_merge(
        eq_blend, X_combined, y_combined,
        backend,
        is_root      = is_root,
        refine_steps = config.merge_refine_steps,
        refine_lr    = config.merge_refine_lr,
    )

    logging.info(
        f"Merge {node_a.node_id}+{node_b.node_id} complete: "
        f"rmse={eq_merged.rmse:.6f}"
    )

    centroid = (node_a.centroid + node_b.centroid) / 2.0

    return MergeNode(
        node_id       = new_node_id,
        patch_indices = node_a.patch_indices | node_b.patch_indices,
        equation      = eq_merged,
        X_data        = X_combined,
        y_data        = y_combined,
        centroid      = centroid
    )


def _merge_level(pairs, unpaired, next_node_id, original_overlap_data,
                 config, is_root_level):
    """
    Execute one level of merges in parallel.

    pairs:          list of (node_a, node_b, X_ov, y_ov)
    unpaired:       list[MergeNode] -- carry forward unchanged
    next_node_id:   int -- starting ID for new nodes
    is_root_level:  bool -- True if this produces the final single node

    returns: list[MergeNode] -- merged nodes + unpaired nodes
    """
    n_workers = config.n_workers or os.cpu_count()

    args_list = []
    for i, (node_a, node_b, X_ov, y_ov) in enumerate(pairs):
        is_root = is_root_level and (len(pairs) == 1) and (len(unpaired) == 0)
        args_list.append((
            node_a, node_b, X_ov, y_ov,
            next_node_id + i,
            is_root,
            config,
        ))

    merged_nodes = [None] * len(pairs)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_merge_pair, args): i
            for i, args in enumerate(args_list)
        }
        for future in as_completed(futures):
            i = futures[future]
            try:
                merged_nodes[i] = future.result()
            except Exception as e:
                logging.error(
                    f"Merge pair {i} failed: "
                    f"{type(e).__name__}: {e}"
                )
                raise

    return merged_nodes + unpaired


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compile_mesh(mesh, config=None):
    """
    Full pipeline: trimesh.Trimesh -> Equation

    mesh:   trimesh.Trimesh
    config: PipelineConfig or None (uses defaults)

    returns: Equation -- single implicit equation covering the full mesh.
             The zero level set f(x,y,z) = 0 approximates the mesh surface.

    Pipeline stages:
        1. Segment mesh into patches (FPS + radius extraction)
        2. Leaf pass (parallel): sample -> GPR -> SR per patch
           + pre-compute overlap GPR for all adjacent patch pairs
        3. Merge tree (level-parallel): blend + PySR residual correction
           + mandatory coefficient refinement at each merge step
        4. Return root equation
    """
    if config is None:
        config = PipelineConfig()

    logging.info("compile_mesh: starting pipeline")
    logging.info(f"Config: {config}")

    # Stage 1: segmentation
    logging.info("Stage 1: segmentation")
    result     = segment_mesh(mesh, n_seeds=config.n_seeds, strategy = 'voronoi' if config.use_voronoi else 'radius')
    patches    = result['patches']
    adjacency  = result['adjacency']

    logging.info(
        f"Segmentation: {len(patches)} patches, "
        f"{len(adjacency)} adjacent pairs"
    )

    seed_positions = mesh.vertices[result['seed_vertices']]  # (n_seeds, 3)

    if config.blend_sigma is None:
        config.blend_sigma = result.get('radius', 1.0)

    # Stage 2: leaf pass
    logging.info("Stage 2: leaf pass")
    leaf_results, original_overlap_data = _process_all_leaves(
        patches, adjacency, seed_positions, mesh, config
    )

    logging.info(
        f"Leaf pass complete: {len(leaf_results)} equations, "
        f"{len(original_overlap_data)} overlap regions computed"
    )

    # Convert leaf results to MergeNodes
    nodes = []
    for leaf in leaf_results:
        nodes.append(MergeNode(
            node_id       = leaf.patch_idx,
            patch_indices = frozenset([leaf.patch_idx]),
            equation      = leaf.equation,
            X_data        = leaf.X_query,
            y_data        = leaf.y_query,
            centroid      = leaf.seed_position
        ))

    # Stage 3: merge tree
    logging.info("Stage 3: merge tree")
    next_node_id = len(patches)
    level        = 0

    while len(nodes) > 1:
        logging.info(
            f"Merge level {level}: {len(nodes)} nodes remaining"
        )

        pairs, unpaired = _build_merge_pairs(nodes, original_overlap_data)

        if not pairs:
            # No adjacent pairs found -- this should not happen on a
            # connected mesh but handle gracefully
            logging.error(
                f"No adjacent pairs at merge level {level}. "
                f"Remaining nodes: {[n.node_id for n in nodes]}. "
                f"Check segmentation connectivity."
            )
            break

        is_root_level = (
            len(pairs) == 1 and
            len(unpaired) == 0
        )

        nodes = _merge_level(
            pairs, unpaired,
            next_node_id,
            original_overlap_data,
            config,
            is_root_level,
        )

        next_node_id += len(pairs)
        level        += 1

        logging.info(
            f"Merge level {level-1} complete: "
            f"{len(nodes)} nodes remaining"
        )

    root_equation = nodes[0].equation

    logging.info(
        f"Pipeline complete: "
        f"rmse={root_equation.rmse:.6f}, "
        f"expr={root_equation.sympy_expr}"
    )

    return root_equation