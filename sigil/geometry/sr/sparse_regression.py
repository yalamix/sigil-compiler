# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from sigil.geometry.sr.base import _denormalize_expr
from sigil.geometry.sr.base import *
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Lasso fit (Phase 1)
# ---------------------------------------------------------------------------

def _normalize_X(X):
    """
    Center and scale X to [-1, 1]^3 before building the feature matrix.
    
    Without normalization, polynomial features have wildly different scales:
    for coordinates near 0.1, x^4 ~ 1e-4 while the bias term is 1.0 --
    a 10000x difference. Lasso and gradient descent both struggle with this.

    Returns X_normalized, center, scale so the caller can invert if needed.
    center: (3,) -- subtract before scaling
    scale:  scalar -- divide after centering
    """
    center = X.mean(axis=0)          # (3,) centroid of query points
    X_centered = X - center          # (N, 3)
    scale = np.abs(X_centered).max() # scalar -- max absolute value
    if scale < 1e-10:
        scale = 1.0
    X_normalized = X_centered / scale
    return X_normalized, center, scale


def _lasso_fit(Phi, y, initial_alphas=None):
    from sklearn.linear_model import LassoCV, Lasso

    if initial_alphas is None:
        # Cold start: use cross-validation to find best lambda
        lasso = LassoCV(
            cv=5,
            fit_intercept=False,
            max_iter=10000,
            n_jobs=-1
        )
        lasso.fit(Phi, y)
        best_lambda = lasso.alpha_
        alphas = lasso.coef_

    else:
        # Warm start: we already have a good solution from a nearby patch.
        # Run LassoCV first to find the best lambda, then warm-start
        # Lasso from initial_alphas at that lambda.
        lasso_cv = LassoCV(
            cv=5,
            fit_intercept=False,
            max_iter=10000,
            n_jobs=-1
        )
        lasso_cv.fit(Phi, y)
        best_lambda = lasso_cv.alpha_

        # Now fine-tune from the warm start at the discovered lambda
        lasso = Lasso(
            alpha=best_lambda,
            fit_intercept=False,
            max_iter=10000,
            warm_start=True
        )
        lasso.coef_ = initial_alphas.astype(np.float64)
        lasso.fit(Phi, y)
        alphas = lasso.coef_

    logging.info(f"Lasso: lambda={best_lambda:.6f}, "
                  f"nonzero={np.sum(alphas != 0)}/{len(alphas)}")

    return alphas


# ---------------------------------------------------------------------------
# Torch refinement (Phase 2)
# ---------------------------------------------------------------------------

def _refine_torch(Phi, y, alphas_sparse, n_steps=500, lr=1e-3):
    """
    Phase 2: fine-tune nonzero coefficients via gradient descent.
    Removes L1 shrinkage bias from Lasso.

    Phi:            (N, n_features) feature matrix
    y:              (N,) target values
    alphas_sparse:  (n_features,) sparse alphas from Phase 1
    n_steps:        gradient descent iterations
    lr:             learning rate

    returns: (n_features,) refined alphas
             zero entries from Phase 1 remain exactly zero --
             we only optimize the nonzero terms

    Why only optimize nonzero terms:
        Optimizing all terms would re-introduce the dense solution.
        We respect the sparsity pattern Lasso discovered.
    """
    import torch

    nonzero_mask = alphas_sparse != 0          # (n_features,) bool
    nonzero_idx  = np.where(nonzero_mask)[0]   # indices of nonzero terms

    if len(nonzero_idx) == 0:
        logging.warning("All alphas are zero after Lasso -- "
                        "returning zeros without refinement")
        return alphas_sparse

    # Extract the columns of Phi corresponding to nonzero terms
    Phi_sparse = Phi[:, nonzero_idx]           # (N, n_nonzero)

    # Convert to torch tensors
    Phi_t = torch.tensor(Phi_sparse, dtype=torch.float32)  # (N, n_nonzero)
    y_t   = torch.tensor(y,          dtype=torch.float32)  # (N,)

    # Initialize from Lasso solution -- not random, not cold
    init = alphas_sparse[nonzero_idx].astype(np.float32)
    params = torch.tensor(init, requires_grad=True)        # (n_nonzero,)

    optimizer = torch.optim.Adam([params], lr=lr)

    pbar = tqdm(range(n_steps), desc=f"Refining {len(params)} params", 
                    unit="step", leave=False)        

    for step in pbar:
        optimizer.zero_grad()

        y_pred = Phi_t @ params                # (N,) -- matrix-vector product
        loss   = torch.mean((y_pred - y_t)**2) # scalar MSE

        loss.backward()
        optimizer.step()

    pbar.close()

    # Write refined values back into full alphas array
    alphas_refined = alphas_sparse.copy()
    alphas_refined[nonzero_idx] = params.detach().numpy()

    logging.debug(f"Refinement: MSE {loss.item():.6f} after {n_steps} steps")

    return alphas_refined                      # (n_features,)


# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------

def _compute_rmse(Phi, y, alphas):
    """
    Evaluate fit quality: root mean squared error on training data.

    Phi:    (N, n_features)
    y:      (N,)
    alphas: (n_features,)

    returns: float
    """
    y_pred = Phi @ alphas           # (N,) -- predicted values
    residuals = y_pred - y          # (N,)
    return float(np.sqrt(np.mean(residuals**2)))


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------

class SparseRegressionBackend(SRBackend):
    """
    Sparse regression SR backend.

    Conceptually inspired by SINDy (Brunton et al. 2016) -- sparse
    identification via L1-penalized regression over a polynomial feature
    library. No dynamical systems machinery, just L1 least squares.

    Two-phase fit:
        1. LassoCV: find sparsity pattern (which terms survive)
        2. PyTorch gradient descent: refine nonzero coefficients
           without L1 bias

    Warm starting via initial_alphas: if provided, Lasso starts from
    the previous patch's solution rather than zeros. For smooth surfaces,
    adjacent patches have similar equations, so convergence is faster
    and the sparsity pattern is more stable.
    """

    def __init__(self, degree=4, n_refine_steps=500, lr=1e-3):
        """
        degree:          max polynomial degree (default 4)
        n_refine_steps:  PyTorch refinement iterations (default 500)
        lr:              refinement learning rate (default 1e-3)
        """
        self.degree         = degree
        self.n_refine_steps = n_refine_steps
        self.lr             = lr

    def get_feature_names(self):
        """
        Return feature names for this backend's degree.
        Defined here (not on ABC) because feature names are a
        property of the polynomial feature library, which only
        sparse regression has.
        """
        # Build a dummy single-point feature matrix to get names
        dummy = np.zeros((1, 3))
        _, feature_names = build_feature_matrix(dummy, self.degree)
        return feature_names

    def fit(self, X, y, initial_alphas=None):
        """
        X:              (N, 3) query points
        y:              (N,) scalar field values
        initial_alphas: (n_features,) optional -- from previous patch

        returns: Equation
        """
        X_norm, center, scale = _normalize_X(X)
        Phi, feature_names = build_feature_matrix(X_norm, self.degree)
        # Phi: (N, n_features)

        # Validate initial_alphas shape if provided
        if initial_alphas is not None:
            if len(initial_alphas) != Phi.shape[1]:
                logging.warning(
                    f"initial_alphas length {len(initial_alphas)} != "
                    f"n_features {Phi.shape[1]} -- ignoring warm start"
                )
                initial_alphas = None

        # Phase 1: sparse structure via Lasso
        alphas_sparse = _lasso_fit(Phi, y, initial_alphas)

        # Phase 2: refine nonzero coefficients via gradient descent
        alphas_refined = _refine_torch(
            Phi, y, alphas_sparse,
            n_steps=self.n_refine_steps,
            lr=self.lr
        )

        rmse = _compute_rmse(Phi, y, alphas_refined)

        # Convert to sympy expression
        sympy_expr = alphas_to_sympy(alphas_refined, feature_names)

        logging.debug(f"SparseRegression: degree={self.degree}, "
                      f"n_nonzero={np.sum(alphas_refined != 0)}, "
                      f"rmse={rmse:.6f}, "
                      f"expr={sympy_expr}")

        # After building sympy_expr from normalized alphas:
        sympy_expr_world = _denormalize_expr(sympy_expr, center, scale)
        sympy_expr_world = sympy.expand(sympy_expr_world)

        return Equation(
            sympy_expr    = sympy_expr_world,   # world coordinates
            rmse          = rmse,
            degree        = self.degree,
            alphas        = alphas_refined,
            feature_names = feature_names,
            # no _center, no _scale
        )