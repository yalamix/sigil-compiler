# sigil/geometry/sr/base.py

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import sympy
from sklearn.preprocessing import PolynomialFeatures
import re
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s'
)

# ---------------------------------------------------------------------------
# Equation
# ---------------------------------------------------------------------------

@dataclass
class Equation:
    """
    A symbolic equation f(x0, x1, x2) representing a scalar field patch.
    The implicit surface is the zero level set: f(x0, x1, x2) = 0.

    sympy_expr:    source of truth -- always present, always correct.
                   Variables are always named x0, x1, x2.

    rmse:          fit error on the query points used during SR.

    degree:        max polynomial degree. 0 if the expression is not
                   polynomial (e.g. contains sin, exp).

    alphas:        coefficient array for linear backends (sparse regression).
                   None for tree-based backends (PySR, DARTS).
                   Stored for warm-starting future SR fits on nearby patches.

    feature_names: list of feature name strings corresponding to alphas.
                   None when alphas is None.
                   Ordering matches sklearn PolynomialFeatures output:
                   graded lexicographic -- ['1', 'x0', 'x1', 'x2',
                   'x0^2', 'x0 x1', ...].

    _compiled_fn:  cached lambdified numpy function, populated on first
                   __call__. Not stored in repr or equality checks.
    """
    sympy_expr:    sympy.Expr
    rmse:          float
    degree:        int
    alphas:        Optional[np.ndarray]  = field(default=None)
    feature_names: Optional[list]        = field(default=None)
    additive_terms: Optional[list]       = field(default=None)
    _compiled_fn:  object                = field(default=None,
                                                  repr=False,
                                                  compare=False)

    def __post_init__(self):
        terms = sympy.Add.make_args(self.sympy_expr)
        self.additive_terms = [t.as_coeff_Mul() for t in terms]

    def __call__(self, points):
        """
        Evaluate f(x0, x1, x2) at query points.

        points: (N, 3) float array -- columns are x0, x1, x2
        returns: (N,) float array of scalar field values

        Compiles sympy_expr to a numpy function on first call via
        lambdify, then caches it. ~100-1000x faster than subs() in a loop.
        The compiled function handles numpy broadcasting natively, so
        sin, exp, etc. map directly to np.sin, np.exp.
        """
        if self._compiled_fn is None:
            x0, x1, x2 = sympy.symbols('x0 x1 x2')
            self._compiled_fn = sympy.lambdify(
                [x0, x1, x2],
                self.sympy_expr,
                modules='numpy'
            )

        # Apply normalization if this equation was fitted on normalized coords
        center = getattr(self, '_center', None)
        scale  = getattr(self, '_scale',  1.0)
        if center is not None:
            points = (points - center) / scale

        return self._compiled_fn(
            points[:, 0],   # (N,) -- x0 column
            points[:, 1],   # (N,) -- x1 column
            points[:, 2]    # (N,) -- x2 column
        )

    def gradient(self, points):
        """
        Evaluate the gradient [df/dx0, df/dx1, df/dx2] at query points.
        This is the surface normal direction: n = grad(f) / ||grad(f)||.

        Computed analytically from sympy_expr via symbolic differentiation,
        then lambdified. Exact normals -- no finite differences needed.

        points: (N, 3)
        returns: (N, 3) -- unnormalized gradient vectors
        """
        if not hasattr(self, '_grad_fns') or self._grad_fns is None:
            x0, x1, x2 = sympy.symbols('x0 x1 x2')
            df_dx0 = sympy.diff(self.sympy_expr, x0)
            df_dx1 = sympy.diff(self.sympy_expr, x1)
            df_dx2 = sympy.diff(self.sympy_expr, x2)
            self._grad_fns = [
                sympy.lambdify([x0, x1, x2], df_dx0, modules='numpy'),
                sympy.lambdify([x0, x1, x2], df_dx1, modules='numpy'),
                sympy.lambdify([x0, x1, x2], df_dx2, modules='numpy'),
            ]

        # Apply normalization if this equation was fitted on normalized coords
        center = getattr(self, '_center', None)
        scale  = getattr(self, '_scale',  1.0)
        if center is not None:
            points = (points - center) / scale

        g0 = self._grad_fns[0](points[:, 0], points[:, 1], points[:, 2])
        g1 = self._grad_fns[1](points[:, 0], points[:, 1], points[:, 2])
        g2 = self._grad_fns[2](points[:, 0], points[:, 1], points[:, 2])

        return np.column_stack([g0, g1, g2])   # (N, 3)

    def normal(self, points):
        """
        Unit surface normals at query points.
        points: (N, 3)
        returns: (N, 3) unit vectors
        """
        grad = self.gradient(points)                                    # (N, 3)
        norms = np.linalg.norm(grad, axis=1, keepdims=True)            # (N, 1)
        return grad / (norms + 1e-8)                                    # (N, 3)

    def __repr__(self):
        return (f"Equation(expr={self.sympy_expr}, "
                f"degree={self.degree}, rmse={self.rmse:.6f})")


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class SRBackend(ABC):
    """
    Abstract base class for all symbolic regression backends.

    All backends receive GPR-evaluated scalar field data (X, y) and
    return an Equation representing f(x0, x1, x2) ≈ 0 on the surface.

    Backends are allowed to ignore parameters they don't support.
    The pipeline always calls fit(X, y, initial_alphas=...) regardless
    of backend -- backends that don't use initial_alphas simply ignore it.
    """

    @abstractmethod
    def fit(self, X, y, initial_alphas=None) -> Equation:
        """
        Fit a symbolic equation to scalar field data.

        X:              (N, 3) query points (from generate_query_points)
        y:              (N,) scalar field values (from predict)
        initial_alphas: (n_features,) optional warm start coefficients.
                        Meaningful for SparseRegressionBackend.
                        Ignored by PySRBackend and DARTSBackend.

        returns: Equation
        """
        ...


# ---------------------------------------------------------------------------
# Shared feature matrix utility
# ---------------------------------------------------------------------------

def _feature_name_to_sympy(fname, var_map):
    """
    Convert a PolynomialFeatures feature name string to a sympy expression.
    
    sklearn uses:  '1', 'x0', 'x0^2', 'x0 x1', 'x0^2 x1'
    sympy needs:   '1', 'x0', 'x0**2', 'x0*x1', 'x0**2*x1'
    """
    s = fname.replace('^', '**')  # exponents
    s = re.sub(r'(\w)\s+(\w)', r'\1*\2', s)  # implicit multiplication
    return sympy.sympify(s, locals=var_map)


def build_feature_matrix(X, degree):
    """
    Build polynomial feature matrix from 3D query points.

    Uses sklearn PolynomialFeatures which produces features in graded
    lexicographic order -- the canonical ordering all backends agree on.

    X:      (N, 3) -- columns are x0, x1, x2
    degree: int

    returns:
        Phi:           (N, n_features) float array
        feature_names: list of n_features strings
                       e.g. degree=2: ['1', 'x0', 'x1', 'x2',
                                       'x0^2', 'x0 x1', 'x0 x2',
                                       'x1^2', 'x1 x2', 'x2^2']

    Example:
        Phi[i, j] = value of feature j at point X[i]
        For feature 'x0^2 x1': Phi[i, j] = X[i, 0]**2 * X[i, 1]
    """
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    Phi = poly.fit_transform(X)                     # (N, n_features)

    # sklearn uses spaces and ^ notation internally
    # get_feature_names_out returns e.g. ['1', 'x0', 'x1', 'x2', 'x0^2'...]
    feature_names = poly.get_feature_names_out(['x0', 'x1', 'x2']).tolist()

    return Phi, feature_names                       # (N, n_features), list


def alphas_to_sympy(alphas, feature_names):
    """
    Convert coefficient array to a sympy expression.

    alphas:        (n_features,) float array
    feature_names: list of strings in PolynomialFeatures ordering

    returns: sympy.Expr

    Example:
        alphas = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
        feature_names = ['1','x0','x1','x2','x0^2','x0 x1',
                         'x0 x2','x1^2','x1 x2','x2^2']
        result: x0**2 + x1**2 + x2**2  (a sphere)
    """
    x0, x1, x2 = sympy.symbols('x0 x1 x2')
    var_map = {'x0': x0, 'x1': x1, 'x2': x2}

    expr = sympy.Integer(0)
    for alpha, fname in zip(alphas, feature_names):
        if abs(alpha) < 1e-10:          # skip near-zero terms
            continue
        term = _feature_name_to_sympy(fname, var_map)
        expr = expr + alpha * term

    return sympy.simplify(expr)


def sympy_to_alphas(sympy_expr, feature_names):
    """
    Extract coefficient array from a sympy polynomial expression.
    Used for warm-starting: convert a known equation back to alphas
    so the next patch can use it as initial_alphas.

    sympy_expr:    sympy.Expr -- must be polynomial in x0, x1, x2
    feature_names: list of strings in PolynomialFeatures ordering

    returns: (n_features,) float array
             zero for features not present in sympy_expr

    Returns None if sympy_expr is not polynomial (e.g. contains sin).
    Caller should check for None before using as warm start.
    """
    x0, x1, x2 = sympy.symbols('x0 x1 x2')

    try:
        poly = sympy.Poly(sympy.expand(sympy_expr), x0, x1, x2)
    except sympy.PolynomialError:
        # Expression is not polynomial -- can't extract alphas
        return None

    alphas = np.zeros(len(feature_names))
    var_map = {'x0': x0, 'x1': x1, 'x2': x2}

    for i, fname in enumerate(feature_names):
        term = _feature_name_to_sympy(fname, var_map)
        try:
            # Get the exponent tuple for this monomial
            term_poly = sympy.Poly(term, x0, x1, x2)
            monomial = term_poly.monoms()[0]    # e.g. (2, 1, 0) for x0^2*x1
            alphas[i] = float(poly.nth(*monomial))
        except Exception:
            alphas[i] = 0.0

    return alphas