# sigil/pipeline/balloon_pipeline.py

import logging
import sys
import numpy as np
import sympy
import trimesh

from sigil.geometry.scalar_field import sample_mesh_sdf
from sigil.geometry.sr.base  import build_feature_matrix
from sigil.pipeline.balloon_eikonal import refine_eikonal
from sigil.geometry.sr.sparse_regression import (
    _lasso_fit,
    _refine_torch,
    _compute_rmse,
    _normalize_X,
    _denormalize_expr,
    alphas_to_sympy,
)
from sigil.geometry.sr.base import Equation
from dataclasses import dataclass, field
from typing import Optional

PRUNE_THRESHOLD = 1e-4

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BalloonConfig:
    # Sampling
    n_surface:        int   = 10000   # points sampled on surface (f=0)
    epsilon:          float = 0.01    # offset distance for ±ε samples

    # Degree schedule
    start_degree:     int   = 2       # always start with sphere
    max_degree:       int   = 20      # hard ceiling
    degree_step:      int   = 2       # increment when plateau

    # Gradient descent per degree level
    gd_steps:         int   = 2000    # steps per degree level
    gd_lr:            float = 1e-3    # learning rate

    # Lasso regularization
    lasso_lambda:     float = 1e-4    # sparsity penalty

    # Convergence
    rmse_threshold:   float = 1e-3    # stop if rmse below this
    plateau_patience: int   = 3       # degree increases before giving up
                                      # (if rmse not improving)

    # Optional PySR correction on residuals
    use_pysr_correction:  bool = False
    pysr_niterations:     int  = 50

    visualize_progress: bool = False 
    min_degree_improvement: float = 1e-4

    lambda_eikonal:  float = 0.01
    lambda_curv:     float = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sphere_alphas(r, n_features):
    """
    Initialize alpha vector for bounding sphere x0²+x1²+x2²-r²=0.

    Feature order from sklearn PolynomialFeatures degree=2:
    ['1', 'x0', 'x1', 'x2', 'x0^2', 'x0 x1', 'x0 x2',
     'x1^2', 'x1 x2', 'x2^2']
    indices: 0      1     2     3     4        5        6
             7       8      9

    We want: -r² * 1  +  1 * x0²  +  1 * x1²  +  1 * x2²
    """
    alphas = np.zeros(n_features)
    _, names = build_feature_matrix(np.zeros((1, 3)), degree=2)

    for i, name in enumerate(names):
        if name == '1':
            alphas[i] = -r ** 2
        elif name == 'x0^2':
            alphas[i] = 1.0
        elif name == 'x1^2':
            alphas[i] = 1.0
        elif name == 'x2^2':
            alphas[i] = 1.0

    return alphas


def _expand_alphas(alphas, old_degree, new_degree):
    """
    Pad alpha vector from old_degree feature space to new_degree feature space.
    Sklearn guarantees old features are a prefix of new features.
    New terms initialized to zero -- balloon keeps current shape.
    """
    _, new_features = build_feature_matrix(np.zeros((1, 3)), new_degree)
    new_alphas = np.zeros(len(new_features))
    new_alphas[:len(alphas)] = alphas
    return new_alphas


def _compute_rmse_direct(X, y, alphas, degree):
    """Evaluate polynomial on X and compute RMSE against y."""
    Phi, _ = build_feature_matrix(X, degree)
    y_pred  = Phi @ alphas
    return float(np.sqrt(np.mean((y_pred - y) ** 2)))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def compile_mesh_balloon(mesh, config=None):
    """
    Balloon pipeline: single global polynomial implicit surface.

    Starts with the bounding sphere and deforms it toward the mesh
    surface via gradient descent, increasing polynomial degree when
    the fit plateaus.

    mesh:   trimesh.Trimesh (should be centered and normalized)
    config: BalloonConfig or None (uses defaults)

    returns: Equation -- single polynomial implicit equation.
             Zero level set f(x,y,z) = 0 approximates the mesh surface.
    """
    if config is None:
        config = BalloonConfig()

    logging.info("compile_mesh_balloon: starting balloon pipeline")
    logging.info(f"Config: {config}")

    # Stage 1: sample mesh SDF directly -- no GPR
    logging.info(
        f"Stage 1: sampling mesh SDF "
        f"({config.n_surface} surface points, "
        f"epsilon={config.epsilon})"
    )
    X, y = sample_mesh_sdf(
        mesh,
        n_surface = config.n_surface,
        epsilon   = config.epsilon,
    )
    logging.info(f"Dataset: {len(X)} points total")

    # Normalize X to [-1,1]^3 for numerical stability
    # (reuse existing normalization)
    X_norm, center, scale = _normalize_X(X)

    from sigil.pipeline.balloon_eikonal import compute_mesh_curvature
    curvature_target = compute_mesh_curvature(mesh, X[:config.n_surface])

    # Stage 2: initialize bounding sphere
    logging.info("Stage 2: initializing bounding sphere")
    r_world = float(np.max(np.linalg.norm(X, axis=1))) * 1.1
    r_norm  = r_world / scale   # sphere radius in normalized coords

    _, init_features = build_feature_matrix(
        np.zeros((1, 3)), config.start_degree
    )
    alphas  = _sphere_alphas(r_norm, len(init_features))
    degree  = config.start_degree

    rmse = _compute_rmse_direct(X_norm, y, alphas, degree)
    logging.info(
        f"Bounding sphere: degree={degree}, rmse={rmse:.6f}, "
        f"r_norm={r_norm:.4f}"
    )

    # Stage 3: optimization loop
    logging.info("Stage 3: balloon optimization loop")

    best_rmse     = rmse
    plateau_count = 0

    while True:
        rmse_at_degree_start = rmse
        # Build feature matrix for current degree
        Phi, feature_names = build_feature_matrix(X_norm, degree)

        # Gradient descent
        # alphas = _refine_torch(Phi, y, alphas,
        #                     n_steps=config.gd_steps,
        #                     lr=config.gd_lr)

        alphas = refine_eikonal(
            X_norm, y, alphas, feature_names, config.n_surface,
            curvature_target,
            degree,
            n_steps        = config.gd_steps,
            lr             = config.gd_lr,
            lambda_eikonal = config.lambda_eikonal,
            lambda_curv    = config.lambda_curv,
        )        

        rmse = _compute_rmse_direct(X_norm, y, alphas, degree)
        logging.info(f"Degree {degree}: rmse={rmse:.6f}")

        coeff_norm = np.linalg.norm(alphas)
        logging.info(f"  coeff_norm={coeff_norm:.3f}")        
        # Check convergence
        if rmse < config.rmse_threshold:
            logging.info(f"Converged at degree {degree}, rmse={rmse:.6f}")
            break

        if degree >= config.max_degree:
            logging.info(f"Reached max degree {config.max_degree}, rmse={rmse:.6f}")
            break

        # improvement = rmse_at_degree_start - rmse
        # logging.info(f"Degree {degree}: improvement={improvement:.6f}")
        # if improvement < config.min_degree_improvement:
        #     logging.info("Degree no longer helping -- stopping early")
        #     break

        # Check plateau
        if rmse < best_rmse * 0.99:
            best_rmse     = rmse
            plateau_count = 0
        else:
            plateau_count += 1
            logging.info(f"Plateau detected ({plateau_count}/{config.plateau_patience})")

        if plateau_count >= config.plateau_patience:            
            new_degree  = degree + config.degree_step
            logging.info(f"Increasing degree {degree} -> {new_degree}")
            old_n       = len(alphas)
            alphas      = _expand_alphas(alphas, degree, new_degree)
            
            # Break zero-gradient deadlock for new terms
            new_n       = len(alphas)
            noise_scale = float(np.std(alphas[:old_n])) * 0.01
            alphas[old_n:] = np.random.randn(new_n - old_n) * noise_scale
            
            degree        = new_degree
            plateau_count = 0
            best_rmse     = rmse

        if config.visualize_progress:
            try:
                import skimage.measure
                logging.info("Saving visualization...")
                res  = 64
                lin  = np.linspace(-1.5, 1.5, res)
                xx, yy, zz = np.meshgrid(lin, lin, lin, indexing='ij')
                X_grid = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
                
                Phi_grid, _ = build_feature_matrix(X_grid, degree)
                f_grid      = (Phi_grid @ alphas).reshape(res, res, res)
                
                verts, faces, _, _ = skimage.measure.marching_cubes(f_grid, level=0.0,
                                                                    spacing=(3.0/res,)*3)
                verts -= 1.5
                
                
                mesh_approx          = trimesh.Trimesh(verts, faces)
                mesh_approx.visual.face_colors = [100, 200, 100, 180]  # green, semi-transparent
                mesh.visual.face_colors        = [200, 100, 100, 180]  # red, semi-transparent
                
                mesh_display = mesh.copy()
                mesh_display.apply_translation([2.0, 0, 0])  # shift original to the right
                mesh_display.visual.face_colors = [200, 100, 100, 220]  # red

                mesh_approx.visual.face_colors = [100, 200, 100, 220]   # green

                scene = trimesh.Scene([mesh_approx, mesh_display])
                png   = scene.save_image(resolution=(800, 600))

                mesh_approx.export(
                    f'C:\\Users\\yalam\\Documents\\sigil-compiler\\outputs\\balloon_progress\\balloon_degree_{degree}_rmse_{rmse:.4f}.obj'
                )
                
                out_path = f'C:\\Users\\yalam\\Documents\\sigil-compiler\\outputs\\balloon_progress\\balloon_degree_{degree}_rmse_{rmse:.4f}.png'
                with open(out_path, 'wb') as f:
                    f.write(png)
                logging.info(f"Saved visualization: {out_path}")
            except Exception as e:
                logging.info(f"Visualization save failed: {e}")

    # Phi, feature_names = build_feature_matrix(X_norm, degree)
    # alphas = _lasso_fit(Phi, y, alphas)
    # alphas = _refine_torch(Phi, y, alphas, n_steps=500, lr=config.gd_lr)

    # Stage 4: optional PySR correction on residuals
    if config.use_pysr_correction and rmse > config.rmse_threshold:
        logging.info("Stage 4: PySR residual correction")
        from sigil.geometry.sr.pysr_backend import PySRBackend

        Phi, _      = build_feature_matrix(X_norm, degree)
        y_pred      = Phi @ alphas
        residuals   = y - y_pred

        backend     = PySRBackend(niterations=config.pysr_niterations)

        # Denormalize X_norm back to world for PySR
        X_world     = X_norm * scale + center
        eq_correction = backend.fit(X_world, residuals)

        logging.info(
            f"PySR correction rmse={eq_correction.rmse:.6f}, "
            f"expr={eq_correction.sympy_expr}"
        )
    else:
        eq_correction = None

    # Stage 5: build final equation in world coordinates
    logging.info("Stage 5: building final equation")

    alphas_pruned = np.where(np.abs(alphas) > PRUNE_THRESHOLD, alphas, 0.0)

    sympy_expr = alphas_to_sympy(alphas_pruned, feature_names)
    sympy_expr = _denormalize_expr(sympy_expr, center, scale)
    sympy_expr = sympy.expand(sympy_expr)

    if eq_correction is not None:
        sympy_expr = sympy_expr + eq_correction.sympy_expr
        sympy_expr = sympy.expand(sympy_expr)

    # Final RMSE in world coordinates
    # (denormalized expression, evaluated on original X)
    x0, x1, x2    = sympy.symbols('x0 x1 x2')
    compiled       = sympy.lambdify(
        [x0, x1, x2], sympy_expr, modules='numpy'
    )
    y_final        = compiled(X[:, 0], X[:, 1], X[:, 2])
    final_rmse     = float(np.sqrt(np.mean((y_final - y) ** 2)))

    logging.info(
        f"Balloon pipeline complete: "
        f"degree={degree}, rmse={final_rmse:.6f}, "
        f"n_terms={int(np.sum(alphas != 0))}, "
        f"final_eq={sympy_expr}"
    )

    return Equation(
        sympy_expr    = sympy_expr,
        rmse          = final_rmse,
        degree        = degree,
        alphas        = alphas,
        feature_names = feature_names,
    )

def compile_mesh_pysr(mesh, config=None):
    if config is None:
        config = BalloonConfig()

    # Stage 1: same sampling as balloon
    X, y = sample_mesh_sdf(
        mesh,
        n_surface = config.n_surface,
        epsilon   = config.epsilon,
    )
    logging.info(f"Dataset: {len(X)} points, y range [{y.min():.4f}, {y.max():.4f}]")

    # Stage 2: PySR directly on world coordinates
    # (PySR backend handles normalization internally)
    from sigil.geometry.sr.pysr_backend import PySRBackend

    backend = PySRBackend(
        niterations      = 1000,      # overnight
        populations      = 30,        # more diverse search
        batching         = True,
        batch_size       = 2000,
    )

    equation = backend.fit(X, y)
    logging.info(f"PySR result: rmse={equation.rmse:.6f}, expr={equation.sympy_expr}")
    return equation