# sigil/pipeline/balloon_sklearn.py

import logging
import numpy as np
import trimesh
import skimage.measure
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from sigil.geometry.scalar_field import sample_mesh_sdf
from sigil.geometry.sr.base import Equation
from sigil.geometry.sr.sparse_regression import _normalize_X
import sympy

OUTPUT_DIR = Path(r'C:\Users\yalam\Documents\sigil-compiler\outputs\balloon_progress')

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SKLearnConfig:
    # Sampling
    n_surface:      int   = 20000
    epsilon:        float = 0.05

    # Model — any sklearn-compatible regressor
    # Default: LightGBM
    model_name:     str   = 'svr'   # 'lgbm', 'rf', 'gbm', 'svr', 'knn'
    model_params:   dict  = field(default_factory=dict)

    # Output
    visualize_progress: bool = True


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _make_model(name, params):
    if name == 'lgbm':
        from lightgbm import LGBMRegressor
        defaults = dict(n_estimators=500, num_leaves=63, learning_rate=0.05,
                        n_jobs=-1, verbose=-1)
        defaults.update(params)
        return LGBMRegressor(**defaults)

    elif name == 'rf':
        from sklearn.ensemble import RandomForestRegressor
        defaults = dict(n_estimators=200, n_jobs=-1)
        defaults.update(params)
        return RandomForestRegressor(**defaults)

    elif name == 'gbm':
        from sklearn.ensemble import GradientBoostingRegressor
        defaults = dict(n_estimators=300, max_depth=5, learning_rate=0.05)
        defaults.update(params)
        return GradientBoostingRegressor(**defaults)

    elif name == 'svr':
        from sklearn.svm import SVR
        defaults = dict(kernel='rbf', C=10.0, epsilon=0.01)
        defaults.update(params)
        return SVR(**defaults)

    elif name == 'knn':
        from sklearn.neighbors import KNeighborsRegressor
        defaults = dict(n_neighbors=10, weights='distance', n_jobs=-1)
        defaults.update(params)
        return KNeighborsRegressor(**defaults)

    else:
        raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _visualize_sklearn(model, mesh, rmse, model_name):
    try:
        logging.info("Saving sklearn visualization...")
        res = 64
        lin = np.linspace(-1.5, 1.5, res)
        xx, yy, zz = np.meshgrid(lin, lin, lin, indexing='ij')
        X_grid  = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        f_grid = model.predict(X_grid).reshape(res, res, res)

        verts, faces, _, _ = skimage.measure.marching_cubes(
            f_grid, level=0.0, spacing=(3.0/res,)*3
        )
        verts -= 1.5

        mesh_approx = trimesh.Trimesh(verts, faces)
        mesh_approx.visual.face_colors = [100, 200, 100, 220]
        mesh_copy = mesh.copy()
        mesh_copy.apply_translation([2.0, 0, 0])
        mesh_copy.visual.face_colors = [200, 100, 100, 220]

        scene = trimesh.Scene([mesh_approx, mesh_copy])
        png   = scene.save_image(resolution=(800, 600))

        out_path = OUTPUT_DIR / f'sklearn_{model_name}_rmse_{rmse:.4f}.png'
        with open(out_path, 'wb') as f:
            f.write(png)
        logging.info(f"Saved: {out_path}")
    except Exception as e:
        logging.info(f"Visualization failed: {e}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def compile_mesh_sklearn(mesh, config=None):
    if config is None:
        config = SKLearnConfig()

    logging.info("compile_mesh_sklearn: starting")
    logging.info(f"Config: {config}")

    # Stage 1: sample
    logging.info(f"Stage 1: sampling ({config.n_surface} surface pts)")
    X, y = sample_mesh_sdf(mesh, n_surface=config.n_surface, epsilon=config.epsilon)
    logging.info(f"Dataset: {len(X)} points")

    X_norm, center, scale = _normalize_X(X)

    # Stage 2: fit
    logging.info(f"Stage 2: fitting {config.model_name}")
    model = _make_model(config.model_name, config.model_params)
    model.fit(X_norm, y)
    logging.info("Fit complete")

    # Stage 3: evaluate
    y_pred     = model.predict(X_norm)
    final_rmse = float(np.sqrt(np.mean((y_pred - y) ** 2)))
    logging.info(f"RMSE: {final_rmse:.6f}")

    # Stage 4: visualize
    if config.visualize_progress:
        _visualize_sklearn(model, mesh, final_rmse, config.model_name)

    # Stage 5: model size
    import pickle
    model_bytes = len(pickle.dumps(model))
    logging.info(f"Model size: {model_bytes / 1024:.1f} KB")

    logging.info(f"compile_mesh_sklearn complete: "
                 f"model={config.model_name}, rmse={final_rmse:.6f}, "
                 f"size={model_bytes/1024:.1f} KB")

    x0, x1, x2 = sympy.symbols('x0 x1 x2')
    eq = Equation(
        sympy_expr    = sympy.Integer(0),  # placeholder
        rmse          = final_rmse,
        degree        = 0,
        alphas        = None,
        feature_names = None,
    )
    eq.model  = model
    eq.center = center
    eq.scale  = scale
    return eq