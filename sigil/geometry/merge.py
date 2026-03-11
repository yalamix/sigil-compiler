# sigil/geometry/merge.py

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import logging
import re
from typing import Optional
import warnings
import numpy as np
import sympy
import torch
from tqdm import tqdm
from sigil.geometry.sr.base import Equation


# ---------------------------------------------------------------------------
# Blend functions
# ---------------------------------------------------------------------------

def blend_smin(eq_a, eq_b, k=0.1):
    """
    Smooth minimum blend of two implicit surface equations.

    The zero level set of smin(f_a, f_b, k) smoothly interpolates
    between the zero level sets of f_a and f_b.

    smin(a, b, k) = -k * log(exp(-a/k) + exp(-b/k))

    k=0 is exact minimum -- non-differentiable crease at boundary.
    k>0 is smooth -- blend radius proportional to k.
    Larger k = wider, smoother transition. Smaller k = sharper join.

    Result always contains log/exp -- never polynomial.
    Renderer requires Newton's method for ray marching.

    eq_a: Equation
    eq_b: Equation
    k:    float -- smoothness parameter (default 0.1)

    returns: Equation
    """
    if k <= 0:
        warnings.warn(
            f"blend_smin called with k={k}. "
            f"k=0 produces a non-differentiable crease. "
            f"Use k>0 for smooth blending.",
            UserWarning
        )

    fa = eq_a.sympy_expr
    fb = eq_b.sympy_expr
    k_sym = sympy.Float(k)

    expr = -k_sym * sympy.log(
        sympy.exp(-fa / k_sym) + sympy.exp(-fb / k_sym)
    )

    # Combined RMSE is not meaningful here -- no ground truth to compare
    # against. Set to max of the two child RMSEs as a conservative estimate.
    rmse = max(eq_a.rmse, eq_b.rmse)

    return Equation(
        sympy_expr    = expr,
        rmse          = rmse,
        degree        = 0,      # non-polynomial due to log/exp
        alphas        = None,
        feature_names = None,
    )


def blend_polynomial(eq_a, eq_b, X_overlap, y_overlap, degree=2):
    """
    Fit weight w(x,y,z) as a polynomial over the overlap region such that
    f_blend = (1 - w) * f_a + w * f_b matches y_overlap.

    Solving for w:
        y = (1 - w) * f_a + w * f_b
        y = f_a + w * (f_b - f_a)
        w = (y - f_a) / (f_b - f_a)

    Then fit a polynomial to those w values, clamped to [0, 1].
    """
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge
    import sympy

    x0, x1, x2 = sympy.symbols('x0 x1 x2')

    # Evaluate both equations over overlap
    f_a_vals = np.asarray(eq_a(X_overlap), dtype=float)
    f_b_vals = np.asarray(eq_b(X_overlap), dtype=float)

    if np.all(f_a_vals == 0):
        # eq_a is zero everywhere -- blend is just eq_b weighted by w=0
        # return eq_b directly, the correction step will handle the residual
        return eq_b

    if np.all(f_b_vals == 0):
        return eq_a

    # Broadcast scalars to full array if needed
    if f_a_vals.ndim == 0:
        f_a_vals = np.full(len(X_overlap), float(f_a_vals))
    if f_b_vals.ndim == 0:
        f_b_vals = np.full(len(X_overlap), float(f_b_vals))

    denom = f_b_vals - f_a_vals
    safe  = np.abs(denom) > 1e-10

    if safe.sum() < 10:
        # Equations are identical over overlap -- return either
        return eq_a

    # Compute target w values where safe
    # w_target         = np.zeros(len(X_overlap))
    # w_target[safe]   = (y_overlap[safe] - f_a_vals[safe]) / denom[safe]
    # w_target         = np.clip(w_target, 0.0, 1.0)
    f_a_scale = np.sqrt(np.mean(f_a_vals**2)) + 1e-10
    f_b_scale = np.sqrt(np.mean(f_b_vals**2)) + 1e-10

    f_a_norm = f_a_vals / f_a_scale
    f_b_norm = f_b_vals / f_b_scale
    y_norm   = y_overlap / ((f_a_scale + f_b_scale) / 2)

    denom    = f_b_norm - f_a_norm
    w_target = (y_norm - f_a_norm) / denom    

    # Fit polynomial to w
    poly    = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly  = poly.fit_transform(X_overlap[safe])
    ridge   = Ridge(alpha=1e-3)
    ridge.fit(X_poly, w_target[safe])

    # Build sympy weight expression
    feature_names = poly.get_feature_names_out(['x0', 'x1', 'x2'])
    w_expr = sympy.Float(ridge.intercept_)
    for coef, name in zip(ridge.coef_, feature_names):
        if abs(coef) > 1e-10 and name != '1':
            term    = sympy.Float(coef)
            factors = name.split(' ')
            for factor in factors:
                if '^' in factor:
                    var, exp = factor.split('^')
                    term *= sympy.Symbol(var) ** int(exp)
                else:
                    term *= sympy.Symbol(factor)
            w_expr += term

    # Clamp w_expr to [0,1] symbolically -- use sigmoid approximation
    # w_clamped = 1 / (1 + exp(-k*(w - 0.5))) with large k
    # Actually just trust Ridge to stay near [0,1] -- it's fit to clamped targets
    
    blend_expr = (1 - w_expr) * eq_a.sympy_expr + w_expr * eq_b.sympy_expr
    blend_expr = sympy.expand(blend_expr)

    # Evaluate blend RMSE
    compiled = sympy.lambdify([x0, x1, x2], blend_expr, modules='numpy')
    y_blend  = compiled(X_overlap[:, 0], X_overlap[:, 1], X_overlap[:, 2])
    rmse     = float(np.sqrt(np.mean((y_blend - y_overlap) ** 2)))

    return Equation(
        sympy_expr    = blend_expr,
        rmse          = rmse,
        degree        = max(eq_a.degree or 0, eq_b.degree or 0) + degree,
        alphas        = None,
        feature_names = None,
    )


def blend_partition_of_unity(eq_a, eq_b, seed_a, seed_b, sigma):
    """
    seed_a, seed_b: (3,) world coordinate arrays -- seed vertex positions
    sigma:          float -- bandwidth, typically patch radius
    """
    x0, x1, x2 = sympy.symbols('x0 x1 x2')
    
    sa = [float(seed_a[0]), float(seed_a[1]), float(seed_a[2])]
    sb = [float(seed_b[0]), float(seed_b[1]), float(seed_b[2])]
    
    phi_a = sympy.exp(
        -((x0 - sa[0])**2 + (x1 - sa[1])**2 + (x2 - sa[2])**2)
        / sympy.Float(sigma**2)
    )
    phi_b = sympy.exp(
        -((x0 - sb[0])**2 + (x1 - sb[1])**2 + (x2 - sb[2])**2)
        / sympy.Float(sigma**2)
    )
    
    denom = phi_a + phi_b
    w     = phi_a / denom
    
    expr  = sympy.expand(w * eq_a.sympy_expr + (1 - w) * eq_b.sympy_expr)
    
    return Equation(
        sympy_expr    = expr,
        rmse          = max(eq_a.rmse, eq_b.rmse),
        degree        = 0,
        alphas        = None,
        feature_names = None,
    )


# ---------------------------------------------------------------------------
# Coefficient refinement
# ---------------------------------------------------------------------------

def _extract_float_constants(expr):
    """
    Find all sympy.Float atoms in expr that are worth optimizing.

    We skip:
        - sympy.Integer atoms (structural: 0, 1, -1, 2 etc.)
        - Very small floats near zero (< 1e-10) -- likely numerical noise
        - The float 1.0 and -1.0 -- effectively structural

    returns: list of sympy.Float atoms, deduplicated by value
             (same float appearing twice is one parameter -- they move together)
    """
    all_floats = expr.atoms(sympy.Float)

    def is_worth_optimizing(f):
        val = abs(float(f))
        if val < 1e-10:
            return False      # numerical noise, not a meaningful constant
        if abs(val - 1.0) < 1e-10:
            return False      # effectively structural
        return True

    return [f for f in all_floats if is_worth_optimizing(f)]


def refine_coefficients(eq, X, y, steps=1000, lr=1e-3, n_surface=None):
    """
    Freeze expression structure, optimize all numerical constants
    via gradient descent (Adam).

    Works on any Equation regardless of which SR backend produced it.
    Handles arbitrary sympy expressions including sin, cos, exp, Abs, etc.

    Optimizes:
        - All sympy.Float constants (coefficients, offsets, frequencies)
        - Float exponents (e.g. x**1.5 -- the 1.5 is a Float)
    Does NOT optimize:
        - Integer exponents (x**2 stays x**2 -- structural)
        - Constants near 0 or ±1 (treated as structural)

    eq:    Equation -- structure is frozen, only constants change
    X:     (N, 3) world coordinate query points
    y:     (N,) scalar field values (GPR ground truth)
    steps: int -- Adam iterations (default 1000)
    lr:    float -- learning rate (default 1e-3)

    returns: Equation with refined constants, same structure
    """
    float_constants = _extract_float_constants(eq.sympy_expr)

    if not float_constants:
        logging.warning(
            "refine_coefficients: no float constants found in expression. "
            "Returning equation unchanged."
        )
        return eq

    # Before running torch optimization
    if any(abs(float(c)) > 20.0 for c in float_constants):
        logging.warning("Large constants detected -- skipping refinement to avoid overflow")
        return eq

    # Replace each unique float constant with a named symbol _c0, _c1, ...
    # If the same float appears multiple times, all instances move together.
    replacements = {}      # sympy.Float -> sympy.Symbol
    init_values  = []      # initial float values for torch parameters

    for i, f in enumerate(float_constants):
        sym = sympy.Symbol(f'_c{i}')
        replacements[f] = sym
        init_values.append(float(f))

    expr_parametric = eq.sympy_expr.subs(replacements)

    # Build lambdified torch function
    # modules='torch' maps sin->torch.sin, exp->torch.exp, etc.
    # autograd tracks all operations automatically
    x0, x1, x2   = sympy.symbols('x0 x1 x2')
    param_symbols = list(replacements.values())   # [_c0, _c1, ...]

    try:
        f_torch = sympy.lambdify(
            [x0, x1, x2] + param_symbols,
            expr_parametric,
            modules=[{'Abs': torch.abs,
                      'acos': torch.acos,
                      'asin': torch.asin,
                      'atan': torch.atan}, 'torch']
        )
    except Exception as e:
        logging.warning(
            f"refine_coefficients: failed to compile torch function: {e}. "
            f"Returning equation unchanged."
        )
        return eq

    # Convert data to torch tensors
    X_t = torch.tensor(X, dtype=torch.float32)      # (N, 3)
    y_t = torch.tensor(y, dtype=torch.float32)      # (N,)

    # Initialize parameters from current constant values
    params = [
        torch.tensor(v, dtype=torch.float32, requires_grad=True)
        for v in init_values
    ]

    optimizer = torch.optim.Adam(params, lr=lr)

    rmse_before = float(np.sqrt(np.mean((eq(X) - y) ** 2)))

    pbar = tqdm(range(steps), desc=f"Refining {len(params)} params", 
                    unit="step", leave=False)    

    for step in pbar:
        optimizer.zero_grad()
        y_pred = f_torch(X_t[:, 0], X_t[:, 1], X_t[:, 2], *params)
        if y_pred.ndim == 0:
            y_pred = y_pred.expand(len(y_t))

        if n_surface is not None:
            surf = y_pred[:n_surface]
            rest = y_pred[n_surface:]
            sign_y = torch.sign(y_t[n_surface:])
            loss = surf.pow(2).mean() + torch.relu(-rest * sign_y).pow(2).mean()
        else:
            loss = torch.mean((y_pred - y_t) ** 2)

        loss.backward()
        optimizer.step()

    pbar.close()

    # Substitute optimized values back into parametric expression
    back_substitution = {
        sym: sympy.Float(params[i].item())
        for i, sym in enumerate(param_symbols)
    }
    expr_refined = expr_parametric.subs(back_substitution)
    expr_refined = sympy.simplify(expr_refined)

    rmse_after = float(np.sqrt(np.mean(
        (Equation(
            sympy_expr    = expr_refined,
            rmse          = 0.0,
            degree        = eq.degree,
            alphas        = None,
            feature_names = None,
        )(X) - y) ** 2
    )))

    logging.info(
        f"refine_coefficients: "
        f"RMSE {rmse_before:.6f} -> {rmse_after:.6f}, "
        f"{len(params)} parameters, {steps} steps"
    )

    return Equation(
        sympy_expr    = expr_refined,
        rmse          = rmse_after,
        degree        = eq.degree,
        alphas        = None,   # alphas no longer valid after refinement
        feature_names = eq.feature_names,
    )


# ---------------------------------------------------------------------------
# Residual correction merge
# ---------------------------------------------------------------------------

def _build_extra_mappings(prev_equation):
    """
    Extract top-level additive terms from prev_equation.
    Used to bias PySR's search toward already-discovered structure.

    prev_equation.additive_terms is [(coeff, func), ...] from __post_init__.

    returns: (dict of name->sympy_expr, list of names)
             empty if prev_equation is None or has no non-constant terms
    """
    if prev_equation is None or prev_equation.additive_terms is None:
        return {}, []

    mappings = {}
    names    = []
    one      = sympy.Integer(1)

    for i, (coeff, func) in enumerate(prev_equation.additive_terms):
        if sympy.simplify(func - one) == 0:
            continue
        name = f'_term{i}'
        mappings[name] = func
        names.append(name)
        logging.debug(f"Warm start term: {name} = {func}")

    return mappings, names


def residual_correction_merge(eq_blend, X, y, backend,
                               is_root=False,
                               refine_steps=1000,
                               refine_lr=1e-3):
    """
    Fit the residual (y - eq_blend(X)) using backend, return
    eq_blend + correction as a single refined Equation.

    eq_blend:     Equation -- result of blend_smin or blend_polynomial
    X:            (N, 3) world coordinate query points over merged region
    y:            (N,) GPR scalar field values (ground truth)
    backend:      SRBackend instance
    is_root:      bool -- if True, tells PySRBackend to use 4x iterations
    refine_steps: int -- Adam steps for post-merge coefficient refinement
    refine_lr:    float -- learning rate for refinement

    returns: Equation -- blend + correction, coefficients refined

    Pipeline usage:
        1. blend eq_a and eq_b -> eq_blend
        2. fetch overlap GPR data -> X, y
        3. eq_final = residual_correction_merge(eq_blend, X, y, backend)
    """
    y_blend    = eq_blend(X)                         # (N,)
    y_residual = y - y_blend                         # (N,)

    residual_rmse = float(np.sqrt(np.mean(y_residual ** 2)))
    logging.info(
        f"residual_correction_merge: "
        f"blend RMSE={residual_rmse:.6f}, is_root={is_root}"
    )

    if residual_rmse < 1e-4:
        logging.info("Blend already excellent -- skipping SR and refinement")
        return eq_blend

    # Skip correction if residual is already negligible
    if residual_rmse < 1e-3:
        logging.info(
            "Residual below threshold -- skipping SR correction. "
            "Running coefficient refinement only."
        )
        return refine_coefficients(
            eq_blend, X, y,
            steps=refine_steps,
            lr=refine_lr
        )

    # Inject additive terms as extra mappings for PySR warm start
    try:
        from sigil.geometry.sr.pysr_backend import PySRBackend
        if isinstance(backend, PySRBackend):
            mappings, names = _build_extra_mappings(eq_blend)
            if mappings:
                original_unary = list(backend.unary_operators)
                backend.unary_operators = original_unary + names
                backend._extra_sympy_mappings = mappings
                logging.debug(
                    f"Injected {len(mappings)} warm start terms into PySR"
                )
    except ImportError:
        pass

    # Tell PySR to use 4x iterations at root
    original_niterations = None
    try:
        from sigil.geometry.sr.pysr_backend import PySRBackend
        if isinstance(backend, PySRBackend) and is_root:
            original_niterations = backend.niterations
            backend.niterations  = backend.niterations * 4
            logging.info(
                f"Root merge: PySR iterations "
                f"{original_niterations} -> {backend.niterations}"
            )
    except ImportError:
        pass

    # Fit correction
    eq_correction = backend.fit(X, y_residual)

    # Restore backend state
    try:
        from sigil.geometry.sr.pysr_backend import PySRBackend
        if isinstance(backend, PySRBackend):
            backend.unary_operators = original_unary
            if hasattr(backend, '_extra_sympy_mappings'):
                del backend._extra_sympy_mappings
            if original_niterations is not None:
                backend.niterations = original_niterations
    except ImportError:
        pass

    # Combine and simplify
    combined_expr = eq_blend.sympy_expr + eq_correction.sympy_expr
    # simplified    = sympy.expand(combined_expr)

    eq_merged = Equation(
        sympy_expr    = combined_expr,
        rmse          = 0.0,
        degree        = 0,
        alphas        = None,
        feature_names = None,
    )

    # Mandatory coefficient refinement -- re-optimizes all constants
    # jointly now that the full structure is known
    eq_refined = refine_coefficients(
        eq_merged, X, y,
        steps=refine_steps,
        lr=refine_lr
    )

    logging.info(
        f"Merge complete: "
        f"final RMSE={eq_refined.rmse:.6f}, "
        f"expr={eq_refined.sympy_expr}"
    )

    return eq_refined