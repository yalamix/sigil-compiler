# sigil/pipeline/balloon_hybrid.py

import logging
import numpy as np
import sympy
import torch
from scipy.special import sph_harm_y
from dataclasses import dataclass, field
from sigil.geometry.scalar_field import sample_mesh_sdf
from sigil.geometry.sr.base import build_feature_matrix, Equation
from sigil.geometry.sr.sparse_regression import _normalize_X, _denormalize_expr, alphas_to_sympy
from sigil.pipeline.balloon_pipeline import _sphere_alphas


# ---------------------------------------------------------------------------
# Spherical harmonic feature matrix
# ---------------------------------------------------------------------------

def cartesian_to_spherical(X):
    """X: (N, 3) -> r, theta, phi"""
    r     = np.linalg.norm(X, axis=1)
    r     = np.where(r < 1e-10, 1e-10, r)           # avoid divide by zero
    theta = np.arccos(np.clip(X[:, 2] / r, -1, 1))  # polar [0, pi]
    phi   = np.arctan2(X[:, 1], X[:, 0])             # azimuthal [-pi, pi]
    return r, theta, phi


def build_sh_feature_matrix(X, l_max, r_max_degree=4):
    """
    Build feature matrix with columns r^k * Y_l^m(theta, phi).
    Real spherical harmonics: .real for m>=0, .imag for m<0.

    Total features: (r_max_degree+1) * (l_max+1)^2
    """
    r, theta, phi = cartesian_to_spherical(X)

    cols  = []
    names = []
    for k in range(r_max_degree + 1):
        r_k = r ** k
        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                Y_complex = sph_harm_y(l, m, theta, phi)
                Y = Y_complex.real if m >= 0 else Y_complex.imag
                cols.append(r_k * Y)
                names.append(f'r^{k}*Y_{l}^{m}')

    return np.column_stack(cols), names


def build_hybrid_feature_matrix(X, poly_degree=6, l_max=8, r_max_degree=3):
    """
    Concatenate polynomial and spherical harmonic features.
    Lasso/OMP selects from both bases automatically.
    """
    Phi_poly, names_poly = build_feature_matrix(X, poly_degree)
    Phi_sh,   names_sh   = build_sh_feature_matrix(X, l_max, r_max_degree)

    # Remove the constant column from SH (r^0 * Y_0^0 is a constant,
    # already present in polynomial basis)
    const_mask = np.array([not n.startswith('r^0*Y_0') for n in names_sh])
    Phi_sh     = Phi_sh[:, const_mask]
    names_sh   = [n for n, keep in zip(names_sh, const_mask) if keep]

    Phi  = np.hstack([Phi_poly, Phi_sh])
    names = names_poly + names_sh
    return Phi, names


# ---------------------------------------------------------------------------
# OMP support discovery
# ---------------------------------------------------------------------------

def _omp_fit(Phi, y, n_nonzero=50):
    """
    Orthogonal Matching Pursuit: find sparse support then least squares.
    Much faster than Lasso for large feature matrices.
    """
    from sklearn.linear_model import OrthogonalMatchingPursuit
    from sklearn.preprocessing import normalize
    
    # Normalize columns manually since normalize param was removed in sklearn 1.2
    Phi_norm, norms = normalize(Phi, axis=0, return_norm=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero)
    omp.fit(Phi_norm, y)
    
    # Rescale coefficients back
    return omp.coef_ / norms


# ---------------------------------------------------------------------------
# Hybrid refinement (sign loss, no monomial assumption)
# ---------------------------------------------------------------------------

def refine_hybrid(
    Phi,              # (N, n_features) prebuilt feature matrix
    y,                # (N,) SDF values
    alphas,           # (n_features,) initial coefficients
    n_surface,        # number of surface points
    n_steps  = 2000,
    lr       = 1e-3,
    lam_reg  = 1e-5,
    lambda_sign = 1.0,
):
    """
    Gradient descent with sign loss on a prebuilt feature matrix.
    No monomial assumption — works for any basis.
    f(x) = Phi @ alpha, optimize alpha only.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"refine_hybrid: {Phi.shape[1]} features, device={device}")

    Phi_t = torch.tensor(Phi, dtype=torch.float32, device=device)
    y_t   = torch.tensor(y,   dtype=torch.float32, device=device)

    alpha_t = torch.nn.Parameter(
        torch.tensor(alphas, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam([alpha_t], lr=lr)

    for step in range(n_steps):
        optimizer.zero_grad()

        f_vals = Phi_t @ alpha_t                          # (N,)

        L_surf = torch.mean(f_vals[:n_surface] ** 2)

        f_off  = f_vals[n_surface:]
        y_off  = y_t[n_surface:]
        L_sign = torch.mean(torch.relu(-f_off * torch.sign(y_off)) ** 2)

        L_reg  = torch.sum(alpha_t ** 2)
        loss   = L_surf + lambda_sign * L_sign + lam_reg * L_reg

        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            logging.info(
                f"  step {step:4d}: surf={L_surf.item():.6f}  "
                f"sign={L_sign.item():.6f}  "
                f"reg={L_reg.item():.6f}  "
                f"total={loss.item():.6f}"
            )

    return alpha_t.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Sympy expression from hybrid alphas
# ---------------------------------------------------------------------------

def hybrid_alphas_to_sympy(alphas, feature_names, threshold=1e-4):
    """
    Build sympy expression from hybrid (poly + SH) coefficients.
    Polynomial terms use existing alphas_to_sympy convention.
    SH terms are expressed as r^k * Re/Im(Y_l^m) in Cartesian form
    via sympy's Znm (real spherical harmonics).

    For simplicity: SH terms are left as symbolic r^k*Y_l^m strings
    and lambdified via numpy/scipy at eval time rather than expanded
    into full Cartesian polynomials (which are enormous).
    """
    x0, x1, x2 = sympy.symbols('x0 x1 x2')
    r_sym       = sympy.sqrt(x0**2 + x1**2 + x2**2)
    theta_sym   = sympy.acos(x2 / (r_sym + sympy.Float(1e-10)))
    phi_sym     = sympy.atan2(x1, x0)

    expr = sympy.Integer(0)
    for alpha, name in zip(alphas, feature_names):
        if abs(alpha) < threshold:
            continue

        if name.startswith('r^'):
            # SH term: r^k*Y_l^m
            # parse name
            parts = name.split('*')           # ['r^k', 'Y_l^m']
            k     = int(parts[0][2:])
            lm    = parts[1][2:].split('^')   # ['l', 'm']
            l, m  = int(lm[0]), int(lm[1])

            # Use sympy Znm (real spherical harmonics) if available,
            # otherwise fall back to numerical lambda
            r_k   = r_sym ** k
            Y_sym = sympy.Znm(l, m, theta_sym, phi_sym)  # real SH in sympy
            expr  = expr + sympy.Float(alpha) * r_k * Y_sym
        else:
            # Polynomial term: use existing convention
            expr = expr + sympy.Float(alpha) * _name_to_poly_sympy(name, x0, x1, x2)

    return sympy.expand(expr)


def _name_to_poly_sympy(name, x0, x1, x2):
    """Convert sklearn feature name to sympy monomial."""
    if name == '1':
        return sympy.Integer(1)
    result = sympy.Integer(1)
    for factor in name.split(' '):
        if '^' in factor:
            var, exp = factor.split('^')
            sym = [x0, x1, x2][int(var[1])]
            result *= sym ** int(exp)
        else:
            result *= [x0, x1, x2][int(factor[1])]
    return result


# ---------------------------------------------------------------------------
# Hybrid evaluator (for RMSE computation)
# ---------------------------------------------------------------------------

def make_hybrid_evaluator(alphas, feature_names, threshold=1e-4):
    """
    Returns a fast numpy callable f(X) for the hybrid basis.
    Avoids sympy lambdify overhead for large SH expressions.
    """
    nonzero_idx   = [i for i, a in enumerate(alphas) if abs(a) >= threshold]
    nonzero_alpha = alphas[nonzero_idx]
    nonzero_names = [feature_names[i] for i in nonzero_idx]

    def evaluator(X):
        result = np.zeros(len(X))
        r, theta, phi = cartesian_to_spherical(X)
        for alpha, name in zip(nonzero_alpha, nonzero_names):
            if name.startswith('r^'):
                parts = name.split('*')
                k     = int(parts[0][2:])
                lm    = parts[1][2:].split('^')
                l, m  = int(lm[0]), int(lm[1])
                Y_c   = sph_harm_y(l, m, theta, phi)
                Y     = Y_c.real if m >= 0 else Y_c.imag
                result += alpha * (r ** k) * Y
            else:
                # polynomial term
                result += alpha * _eval_poly_name_numpy(name, X)
        return result

    return evaluator


def _eval_poly_name_numpy(name, X):
    if name == '1':
        return np.ones(len(X))
    result = np.ones(len(X))
    for factor in name.split(' '):
        if '^' in factor:
            var, exp = factor.split('^')
            result *= X[:, int(var[1])] ** int(exp)
        else:
            result *= X[:, int(factor[1])]
    return result


# ---------------------------------------------------------------------------
# Main hybrid pipeline
# ---------------------------------------------------------------------------

@dataclass
class HybridConfig:
    n_surface:         int   = 20000
    epsilon:           float = 0.05
    poly_degree:       int   = 6

    # SH complexity schedule
    l_max_start:       int   = 4
    l_max_max:         int   = 12
    l_max_step:        int   = 2
    r_max_degree_start: int  = 2
    r_max_degree_max:  int   = 5

    # Sparsity schedule
    n_nonzero_start:   int   = 40
    n_nonzero_max:     int   = 150
    n_nonzero_step:    int   = 20

    # Convergence
    rmse_threshold:    float = 0.01
    plateau_patience:  int   = 3

    # Gradient descent
    gd_steps:          int   = 2000
    gd_lr:             float = 1e-3
    lam_reg:           float = 1e-5
    lambda_sign:       float = 1.0

    visualize_progress: bool = True


def compile_mesh_hybrid(mesh, config=None):
    if config is None:
        config = HybridConfig()

    logging.info("compile_mesh_hybrid: starting")
    logging.info(f"Config: {config}")

    # Stage 1: sample
    logging.info(f"Stage 1: sampling ({config.n_surface} surface pts, eps={config.epsilon})")
    X, y = sample_mesh_sdf(mesh, n_surface=config.n_surface, epsilon=config.epsilon)
    logging.info(f"Dataset: {len(X)} points")
    X_norm, center, scale = _normalize_X(X)

    l_max        = config.l_max_start
    r_max_degree = config.r_max_degree_start
    best_rmse    = np.inf
    plateau_count = 0
    alphas       = None
    feature_names = None

    while True:
        # Build feature matrix at current complexity
        logging.info(
            f"Building features: poly_deg={config.poly_degree}, "
            f"l_max={l_max}, r_max={r_max_degree}"
        )
        Phi_new, names_new = build_hybrid_feature_matrix(
            X_norm,
            poly_degree  = config.poly_degree,
            l_max        = l_max,
            r_max_degree = r_max_degree,
        )
        logging.info(f"Feature matrix: {Phi_new.shape}")

        # Expand alphas when complexity increases, preserving existing coefficients
        if alphas is None:
            _, poly_names = build_feature_matrix(np.zeros((1,3)), config.poly_degree)
            r_world = float(np.max(np.linalg.norm(X, axis=1))) * 1.1
            r_norm  = r_world / scale
            sphere_a = _sphere_alphas(r_norm, len(poly_names))
            
            alphas = np.zeros(Phi_new.shape[1])
            alphas[:len(poly_names)] = sphere_a  # poly terms get sphere init
            # SH terms get small noise
            alphas[len(poly_names):] = np.random.randn(
                Phi_new.shape[1] - len(poly_names)
            ) * 1e-3
        else:
            n_new = Phi_new.shape[1]
            n_old = len(alphas)

            if n_new > n_old:
                new_alphas = np.zeros(n_new)
                new_alphas[:n_old] = alphas
                noise_scale = float(np.std(alphas)) * 0.01 if np.std(alphas) > 0 else 1e-4
                new_alphas[n_old:] = np.random.randn(n_new - n_old) * noise_scale
                alphas = new_alphas
                logging.info(f"Expanded alphas {n_old} -> {n_new}, noise_scale={noise_scale:.2e}")
            elif n_new < n_old:
                # Should never happen since complexity only increases
                raise ValueError(f"Feature matrix shrank: {n_old} -> {n_new}")                 

        feature_names = names_new   

        # Gradient descent on full feature matrix — all terms compete freely
        alphas = refine_hybrid(
            Phi_new, y,
            alphas      = alphas,
            n_surface   = config.n_surface,
            n_steps     = config.gd_steps,
            lr          = config.gd_lr,
            lam_reg     = config.lam_reg,
            lambda_sign = config.lambda_sign,
        )

        # RMSE
        y_pred = Phi_new @ alphas
        rmse   = float(np.sqrt(np.mean((y_pred - y) ** 2)))
        logging.info(f"RMSE: {rmse:.6f} (best: {best_rmse:.6f})")

        if config.visualize_progress:
            _visualize_hybrid(
                alphas, feature_names, mesh, rmse, X_norm, l_max, r_max_degree, 0
            )

        # Convergence
        if rmse < config.rmse_threshold:
            logging.info(f"Converged: rmse={rmse:.6f}")
            best_rmse = rmse
            break

        # Plateau detection
        if rmse < best_rmse * 0.99:
            best_rmse     = rmse
            plateau_count = 0
        else:
            plateau_count += 1
            logging.info(f"Plateau ({plateau_count}/{config.plateau_patience})")

        # Ceiling check
        if (l_max >= config.l_max_max and
            r_max_degree >= config.r_max_degree_max):
            logging.info("Reached maximum complexity, stopping")
            break

        # Increase complexity on plateau
        if plateau_count >= config.plateau_patience:
            plateau_count = 0
            if l_max < config.l_max_max:
                l_max += config.l_max_step
                logging.info(f"Increasing l_max -> {l_max}")
            elif r_max_degree < config.r_max_degree_max:
                r_max_degree += 1
                logging.info(f"Increasing r_max_degree -> {r_max_degree}")

    # OMP at the end for compression only
    logging.info(f"Final OMP compression (n_nonzero={config.n_nonzero_max})")
    alphas_sparse = _omp_fit(Phi_new, y, n_nonzero=config.n_nonzero_max)
    n_nonzero     = int(np.sum(alphas_sparse != 0))
    logging.info(f"OMP kept {n_nonzero} terms")

    # Short refinement after pruning
    support      = np.where(alphas_sparse != 0)[0]
    Phi_sparse   = Phi_new[:, support]
    alphas_init  = alphas_sparse[support]
    alphas_refined = refine_hybrid(
        Phi_sparse, y,
        alphas      = alphas_init,
        n_surface   = config.n_surface,
        n_steps     = 500,
        lr          = config.gd_lr,
        lam_reg     = config.lam_reg,
        lambda_sign = config.lambda_sign,
    )

    alphas_full = np.zeros(len(feature_names))
    alphas_full[support] = alphas_refined
    feature_names_sparse = [feature_names[i] for i in support]

    # Final RMSE
    evaluator  = make_hybrid_evaluator(alphas_full, feature_names)
    y_pred     = evaluator(X_norm)
    final_rmse = float(np.sqrt(np.mean((y_pred - y) ** 2)))
    logging.info(f"Final RMSE after compression: {final_rmse:.6f}")

    sympy_expr = hybrid_alphas_to_sympy(alphas_full, feature_names)
    logging.info(
        f"compile_mesh_hybrid complete: rmse={final_rmse:.6f}, "
        f"n_terms={n_nonzero}"
    )

    eq = Equation(
        sympy_expr    = sympy_expr,
        rmse          = final_rmse,
        degree        = config.poly_degree,
        alphas        = alphas_full,
        feature_names = feature_names,
    )
    eq.center = center
    eq.scale  = scale
    return eq


def _visualize_hybrid(alphas, feature_names, mesh, rmse, X_norm, l_max, r_max, n_nonzero):
    """Marching cubes visualization using the hybrid evaluator."""
    try:
        import skimage.measure
        import trimesh
        logging.info("Saving hybrid visualization...")
        res = 64
        lin = np.linspace(-1.5, 1.5, res)
        xx, yy, zz = np.meshgrid(lin, lin, lin, indexing='ij')
        X_grid  = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        evaluator = make_hybrid_evaluator(alphas, feature_names)
        f_grid    = evaluator(X_grid).reshape(res, res, res)

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

        out_path = (
            f'C:\\Users\\yalam\\Documents\\sigil-compiler\\'
            f'outputs\\balloon_progress\\'
            f'hybrid_l{l_max}_r{r_max}_n{n_nonzero}_rmse_{rmse:.4f}.png'
        )
        with open(out_path, 'wb') as f:
            f.write(png)
        logging.info(f"Saved: {out_path}")
    except Exception as e:
        logging.info(f"Visualization failed: {e}")