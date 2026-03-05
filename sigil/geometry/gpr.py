# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s'
)

# ---------------------------------------------------------------------------
# Kernel construction
# ---------------------------------------------------------------------------

def make_kernel(length_scale_init):
    """
    Build the GPR kernel: ConstantKernel * RBF + WhiteKernel.

    ConstantKernel: overall amplitude -- lets GPR scale the output range
    RBF:            smoothness -- length scale controls how quickly
                    similarity drops off with distance
    WhiteKernel:    noise term -- absorbs the fact that face-interior
                    points are approximations, not exact surface points

    length_scale_init: float -- initial guess, should be ~half the patch
                       bounding box diagonal. GPR will tune it from here
                       via maximum likelihood during fitting.

    returns: sklearn kernel object
    """
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e6))
        * RBF(length_scale=length_scale_init,
              length_scale_bounds=(length_scale_init * 0.01,
                                   length_scale_init * 100))
        + WhiteKernel(noise_level=1e-4,
                      noise_level_bounds=(1e-8, 1e-1))
    )
    return kernel


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_gpr(X, y, backend='sklearn'):
    """
    Fit a GPR model to scalar field training data.

    X:       (N, 3) training points
    y:       (N,)  scalar labels (0, +epsilon, -epsilon)
    backend: 'sklearn' or 'gpytorch'

    returns: fitted model object
             for sklearn: GaussianProcessRegressor
             for gpytorch: NotImplementedError for now

    The length scale is initialized from the spatial extent of X,
    not hardcoded -- this makes it scale correctly across patches
    of different sizes.
    """
    if backend == 'gpytorch':
        raise NotImplementedError(
            "GPyTorch backend not yet implemented. "
            "Requires worker queue design for CUDA context management -- "
            "see pipeline.py. Use backend='sklearn' for now."
        )

    if backend != 'sklearn':
        raise ValueError(f"Unknown backend '{backend}'. Use 'sklearn' or 'gpytorch'.")

    return _fit_gpr_sklearn(X, y)


def _fit_gpr_sklearn(X, y):
    """
    sklearn GaussianProcessRegressor backend.

    Max ~500 training points before O(N³) matrix inversion becomes slow.
    normalize_y=True: centers y around its mean before fitting.
    This helps when the three label categories have very different scales,
    and improves numerical stability of the kernel hyperparameter search.

    n_restarts_optimizer=3: refit kernel hyperparameters from 3 random
    starting points, keep the best. Reduces chance of landing in a bad
    local optimum of the log-marginal-likelihood.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor

    # Length scale init: half the diagonal of X's bounding box
    bbox_min = X.min(axis=0)    # (3,)
    bbox_max = X.max(axis=0)    # (3,)
    diagonal = np.linalg.norm(bbox_max - bbox_min)  # scalar
    length_scale_init = diagonal * 0.5

    kernel = make_kernel(length_scale_init)

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=3,
        normalize_y=True,
        alpha=1e-6          # numerical stability -- added to diagonal of
                            # kernel matrix before inversion
    )

    gpr.fit(X, y)

    logging.debug(f"GPR fitted: {len(X)} points, "
                  f"kernel={gpr.kernel_}, "
                  f"log_marginal_likelihood={gpr.log_marginal_likelihood_value_:.4f}")

    return gpr


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(model, X_query, backend='sklearn'):
    """
    Evaluate fitted GPR at query points.

    model:    fitted model from fit_gpr()
    X_query:  (M, 3) query points
    backend:  must match the backend used in fit_gpr()

    returns:  (M,) predicted scalar field values

    We only return the mean prediction, not the variance.
    Variance would be useful for active learning (sample more points
    where uncertainty is high) -- possible future extension.
    """
    if backend == 'gpytorch':
        raise NotImplementedError("GPyTorch backend not yet implemented.")

    y_pred = model.predict(X_query, return_std=False)  # (M,)
    return y_pred


# ---------------------------------------------------------------------------
# Query point generation
# ---------------------------------------------------------------------------

def generate_query_points(patch_vertices, mesh, resolution=20):
    """
    Generate a 3D grid of query points over the patch bounding box.
    These are the points where we evaluate the fitted GPR to get a
    dense scalar field for the SR step.

    patch_vertices: (N_patch,) int array
    mesh:           trimesh.Trimesh
    resolution:     int -- number of points per axis
                    resolution=20 gives 20³ = 8000 query points
                    resolution=30 gives 30³ = 27000 query points

    returns: (resolution³, 3) array of 3D grid points

    The grid is axis-aligned and covers the patch bounding box with
    a small margin so SR sees field values just outside the surface too.
    """
    positions = mesh.vertices[patch_vertices]   # (N_patch, 3)
    bbox_min = positions.min(axis=0)            # (3,)
    bbox_max = positions.max(axis=0)            # (3,)

    # Add 10% margin on each side so the grid extends slightly beyond
    # the surface -- SR needs to see the field shape outside the patch
    margin = (bbox_max - bbox_min) * 0.1        # (3,)
    bbox_min = bbox_min - margin
    bbox_max = bbox_max + margin

    # Build 1D grids along each axis then combine into 3D grid
    xs = np.linspace(bbox_min[0], bbox_max[0], resolution)  # (resolution,)
    ys = np.linspace(bbox_min[1], bbox_max[1], resolution)  # (resolution,)
    zs = np.linspace(bbox_min[2], bbox_max[2], resolution)  # (resolution,)

    # np.meshgrid with indexing='ij' gives arrays of shape
    # (resolution, resolution, resolution)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='ij')

    # Flatten and stack into (resolution³, 3)
    query_points = np.column_stack([
        XX.ravel(),   # (resolution³,)
        YY.ravel(),   # (resolution³,)
        ZZ.ravel(),   # (resolution³,)
    ])

    logging.debug(f"Query grid: {resolution}³ = {len(query_points)} points, "
                  f"bbox [{bbox_min} -> {bbox_max}]")

    return query_points