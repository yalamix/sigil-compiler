# sigil/pipeline/balloon_pipeline.py

import logging
import sys
import numpy as np
import sympy
import trimesh

from sigil.geometry.scalar_field import sample_mesh_sdf
from sigil.geometry.sr.base  import build_feature_matrix
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

        logging.info(
            f"Degree {degree}: running gradient descent "
            f"({config.gd_steps} steps, lr={config.gd_lr})"
        )

        Phi, feature_names = build_feature_matrix(X_norm, degree)

        # Gradient descent (torch) -- warm start from current alphas
        alphas = _refine_torch(
            Phi, y, alphas,
            n_steps = config.gd_steps,
            lr      = config.gd_lr,
        )

        rmse = _compute_rmse_direct(X_norm, y, alphas, degree)
        logging.info(f"Degree {degree}: rmse={rmse:.6f}")

        # Check convergence
        if rmse < config.rmse_threshold:
            logging.info(
                f"Converged at degree {degree}, rmse={rmse:.6f}"
            )
            break

        # Check degree ceiling
        if degree >= config.max_degree:
            logging.info(
                f"Reached max degree {config.max_degree}, "
                f"rmse={rmse:.6f}"
            )
            break

        # Check plateau
        if rmse < best_rmse * 0.99:   # at least 1% improvement
            best_rmse     = rmse
            plateau_count = 0
        else:
            plateau_count += 1
            logging.info(
                f"Plateau detected ({plateau_count}/"
                f"{config.plateau_patience})"
            )

        if plateau_count >= config.plateau_patience:
            if degree >= config.max_degree:
                logging.info("Max degree reached at plateau -- stopping")
                break
            # Increase degree
            new_degree = degree + config.degree_step
            logging.info(
                f"Increasing degree {degree} -> {new_degree}"
            )
            alphas        = _expand_alphas(alphas, degree, new_degree)
            degree        = new_degree
            plateau_count = 0

        # Sparsify with Lasso before next degree level
        # (keeps equation lean as degree grows)
        alphas = _lasso_fit(Phi, y, alphas)
        alphas = _refine_torch(Phi, y, alphas,
                               n_steps=200, lr=config.gd_lr)

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

    sympy_expr = alphas_to_sympy(alphas, feature_names)
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