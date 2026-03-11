# sigil/geometry/sr/pysr_backend.py

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import logging
import tempfile
import shutil

import numpy as np
import sympy

from sigil.geometry.sr.base import (
    Equation,
    SRBackend,
)
from sigil.geometry.sr.sparse_regression import (
    _normalize_X,
    _compute_rmse,
)


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------

class PySRBackend(SRBackend):
    """
    PySR symbolic regression backend.

    Uses genetic programming to search the space of mathematical expression
    trees. Can discover non-polynomial expressions (sin, exp, etc.) that
    sparse regression cannot.

    Julia runs in a subprocess -- Python GIL is not held during search.
    Startup cost ~5-10s first call (Julia JIT compilation).

    Each instance writes to a unique temp directory to prevent collisions
    when multiple instances run in parallel via ProcessPoolExecutor.

    Warm starting and residual correction are handled at the pipeline/merge
    level, not here. This backend only knows how to fit (X, y) -> Equation.
    """

    def __init__(self,
                 niterations=25,
                 binary_operators=None,
                 unary_operators=None,
                 populations=15,
                 random_state=None,
                 batching=True,
                 batch_size=1000,
                 loss_function=None):
        self.niterations      = niterations
        self.binary_operators = binary_operators or ["+", "-", "*", "/"]
        self.unary_operators  = unary_operators  or ["sin", "cos",
                                                      "exp", "abs"]
        self.populations      = populations
        self.random_state     = random_state
        self.batching         = batching
        self.batch_size       = batch_size
        self.loss_function    = loss_function

    def fit(self, X, y, initial_alphas=None):
        """
        Fit a symbolic equation to scalar field data.

        X:              (N, 3) query points -- world coordinates
        y:              (N,) scalar field values
        initial_alphas: ignored -- PySR does not use alpha initialization

        returns: Equation in world coordinates
        """
        try:
            from pysr import PySRRegressor
        except ImportError:
            raise ImportError(
                "PySR not installed. Run: pip install pysr "
                "and ensure Julia is available."
            )

        from sigil.geometry.sr.base import _denormalize_expr

        X_norm, center, scale = _normalize_X(X)

        tempdir = tempfile.mkdtemp(prefix='sigil_pysr_')

        extra_kwargs = {}
        if self.random_state is not None:
            extra_kwargs['deterministic'] = True
            extra_kwargs['parallelism']   = 'serial'

        try:
            model = PySRRegressor(
                niterations      = self.niterations,
                binary_operators = self.binary_operators,
                unary_operators  = self.unary_operators,
                populations      = self.populations,
                random_state     = self.random_state,
                tempdir          = tempdir,
                batching         = self.batching,
                batch_size       = self.batch_size,
                loss_function    = self.loss_function,
                maxsize          = 60,
                delete_tempfiles = True,
                verbosity        = 1,
                **extra_kwargs
            )

            model.fit(X_norm, y, variable_names=['x0', 'x1', 'x2'])

            expr_norm  = model.sympy()
            expr_world = _denormalize_expr(expr_norm, center, scale)
            expr_world = sympy.simplify(expr_world)

            logging.debug(f"PySR normalized: {expr_norm}")
            logging.debug(f"PySR world:      {expr_world}")

            eq = Equation(
                sympy_expr    = expr_world,
                rmse          = 0.0,
                degree        = 0,
                alphas        = None,
                feature_names = None,
            )

            y_pred  = eq(X)
            eq.rmse = float(np.sqrt(np.mean((y_pred - y) ** 2)))

            logging.info(f"PySR fit: expr={expr_world}, rmse={eq.rmse:.6f}")
            return eq

        finally:
            shutil.rmtree(tempdir, ignore_errors=True)