# sigil/pipeline/balloon_eikonal.py
#
# Eikonal + curvature loss for the balloon pipeline.
#
# Add to BalloonConfig:
#   lambda_eikonal:  float = 0.1
#   lambda_curv:     float = 0.01
#
# In compile_mesh_balloon, before the optimization loop, compute:
#   curvature_target = compute_mesh_curvature(mesh, X[:n_surface])
#
# Replace _refine_torch calls with:
#   alphas = refine_eikonal(
#       X_norm, y, alphas, feature_names, n_surface,
#       curvature_target, n_steps=config.gd_steps, lr=config.gd_lr,
#       lambda_eikonal=config.lambda_eikonal, lambda_curv=config.lambda_curv,
#   )

import logging
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Parse monomial powers from sklearn feature names
# ---------------------------------------------------------------------------

def _parse_feature_powers(feature_names):
    """
    Parse sklearn PolynomialFeatures feature names into exponent arrays.

    feature_names: list like ['1', 'x0', 'x1', 'x2', 'x0^2', 'x0 x1', ...]

    Returns: (n_features, 3) int32 array -- exponents [p0, p1, p2] per feature.
             e.g. 'x0^2 x1' -> [2, 1, 0]
    """
    powers = np.zeros((len(feature_names), 3), dtype=np.int32)
    for i, name in enumerate(feature_names):
        if name == '1':
            continue
        for factor in name.split(' '):
            if '^' in factor:
                var, exp = factor.split('^')
                powers[i, int(var[1])] = int(exp)
            else:
                powers[i, int(factor[1])] = 1
    return powers


# ---------------------------------------------------------------------------
# Polynomial evaluation differentiable w.r.t. X
# ---------------------------------------------------------------------------

def _eval_poly_from_X(X_t, alpha_t, powers_t):
    """
    f(X) = sum_j alpha_j * x0^p0_j * x1^p1_j * x2^p2_j

    X_t:      (N, 3)        requires_grad=True
    alpha_t:  (n_features,)
    powers_t: (n_features, 3) int32

    Returns: (N,) differentiable w.r.t. both X_t and alpha_t.

    Note: x^0 = 1 always, including x=0. torch.pow(0, 0) = 1. Correct.
    Note: for integer powers of possibly-negative x, torch.pow is fine.
          We cast powers to float for the call but values are integers.
    """
    x0 = X_t[:, 0:1]                          # (N, 1)
    x1 = X_t[:, 1:2]
    x2 = X_t[:, 2:3]

    p0 = powers_t[:, 0].float()               # (n_features,)
    p1 = powers_t[:, 1].float()
    p2 = powers_t[:, 2].float()

    # (N, 1) ** (n_features,) -> broadcasts to (N, n_features)
    Phi = (x0 ** p0) * (x1 ** p1) * (x2 ** p2)

    return Phi @ alpha_t                       # (N,)


# ---------------------------------------------------------------------------
# Main refinement function
# ---------------------------------------------------------------------------

def refine_eikonal(
    X_norm,
    y,
    alphas,
    feature_names,
    n_surface,
    curvature_target,
    current_degree,
    n_steps        = 2000,
    lr             = 1e-3,
    lambda_eikonal = 0.1,
    lambda_curv    = 0.01,
    lam_reg        = 1e-4
):
    """
    Minimize:
      L = L_surface + lambda_eikonal * L_eikonal + lambda_curv * L_curv

    L_surface:  mean((f(x) - y)^2)
    L_eikonal:  mean((|grad f(x)| - 1)^2)
    L_curv:     mean((H_f(x) - H_mesh(x))^2)  -- surface pts only

    Parameters
    ----------
    X_norm:           (N, 3) normalized coordinates
    y:                (N,) SDF values -- 0 on surface, ±eps off surface
    alphas:           (n_features,) initial polynomial coefficients
    feature_names:    list of sklearn feature name strings
    n_surface:        number of surface points (first n_surface rows of X_norm)
    curvature_target: (n_surface,) normalized mesh mean curvature, or None
    n_steps:          gradient descent steps
    lr:               Adam learning rate
    lambda_eikonal:   weight for eikonal term
    lambda_curv:      weight for curvature term

    Returns
    -------
    (n_features,) numpy array of refined coefficients
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"refine_eikonal: {len(alphas)} params, device={device}, "
                 f"lam_eik={lambda_eikonal}, lam_curv={lambda_curv}, lam_reg={lam_reg}")

    powers_np = _parse_feature_powers(feature_names)
    powers_t  = torch.tensor(powers_np, dtype=torch.int32, device=device)

    # X needs requires_grad for eikonal/curvature terms
    X_t = torch.tensor(X_norm, dtype=torch.float32,
                       device=device, requires_grad=True)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)

    alpha_t = torch.nn.Parameter(
        torch.tensor(alphas, dtype=torch.float32, device=device)
    )

    use_curv = (curvature_target is not None) and (lambda_curv > 0.0)
    if use_curv:
        H_target_t = torch.tensor(curvature_target,
                                  dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam([alpha_t], lr=lr)

    for step in range(n_steps):
        optimizer.zero_grad()

        # f(X) -- differentiable w.r.t. X_t and alpha_t
        f_vals = _eval_poly_from_X(X_t, alpha_t, powers_t)   # (N,)

        # L_surface
        L_surf = torch.mean((f_vals - y_t) ** 2)

        # Gradient of f w.r.t. X
        # create_graph=True needed only if we compute curvature (2nd derivs)
        grad_f = torch.autograd.grad(
            f_vals.sum(), X_t,
            create_graph = use_curv,
            retain_graph = True,
        )[0]                                              # (N, 3)

        grad_mag  = torch.sqrt((grad_f ** 2).sum(dim=1) + 1e-8)
        grad_mag_surf = torch.sqrt((grad_f[:n_surface] ** 2).sum(dim=1) + 1e-8)
        L_nondegen = torch.mean(1.0 / (grad_mag_surf + 1e-2))

        # Curvature loss (surface points only)
        if use_curv:
            gx = grad_f[:n_surface, 0]
            gy = grad_f[:n_surface, 1]
            gz = grad_f[:n_surface, 2]
            gm = grad_mag[:n_surface]

            # Diagonal Hessian entries (for Laplacian)
            gxx = torch.autograd.grad(gx.sum(), X_t, retain_graph=True)[0][:n_surface, 0]
            gyy = torch.autograd.grad(gy.sum(), X_t, retain_graph=True)[0][:n_surface, 1]
            gzz = torch.autograd.grad(gz.sum(), X_t, retain_graph=True)[0][:n_surface, 2]

            # Off-diagonal Hessian entries
            gxy = torch.autograd.grad(gx.sum(), X_t, retain_graph=True)[0][:n_surface, 1]
            gxz = torch.autograd.grad(gx.sum(), X_t, retain_graph=True)[0][:n_surface, 2]
            gyz = torch.autograd.grad(gy.sum(), X_t, retain_graph=True)[0][:n_surface, 2]

            laplacian = gxx + gyy + gzz

            gHg = (gx**2 * gxx + gy**2 * gyy + gz**2 * gzz
                   + 2*gx*gy*gxy + 2*gx*gz*gxz + 2*gy*gz*gyz)

            H_implicit = (gm**2 * laplacian - gHg) / (2.0 * (gm**3 + 1e-8))
            L_curv     = torch.mean((H_implicit - H_target_t) ** 2)
        else:
            L_curv = torch.tensor(0.0, device=device)

        L_reg = torch.sum(alpha_t ** 2)
        lam_reg_effective = lam_reg / (current_degree ** 2)        
        if current_degree > 4:
            loss = L_surf + lambda_eikonal * L_nondegen + lambda_curv * L_curv + lam_reg_effective * L_reg
        else:
            loss = L_surf + lambda_eikonal * L_nondegen + lambda_curv * L_curv + lam_reg * L_reg
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            logging.info(
                f"  step {step:4d}: surf={L_surf.item():.6f}  "
                f"nondegen={L_nondegen.item():.6f}  "
                f"curv={L_curv.item():.6f}  "
                f"reg={L_reg.item():.6f}  "
                f"total={loss.item():.6f}"
            )

    return alpha_t.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Mesh curvature helper
# ---------------------------------------------------------------------------

def compute_mesh_curvature(mesh, X_surface):
    """
    Interpolate mean curvature from mesh vertices to sampled surface points.
    Returns normalized (zero mean, unit std) curvature values.

    mesh:      trimesh.Trimesh
    X_surface: (n_surface, 3) points on mesh surface in same coordinate frame
    """
    import trimesh

    try:
        H_verts = trimesh.curvature.discrete_mean_curvature_measure(
            mesh, mesh.vertices, radius=0.05
        )
        logging.info("Using discrete mean curvature")
    except Exception as e:
        logging.warning(f"Mean curvature failed ({e}), falling back to Gaussian")
        H_verts = trimesh.curvature.discrete_gaussian_curvature_measure(
            mesh, mesh.vertices, radius=0.05
        )

    _, v_idx  = trimesh.proximity.ProximityQuery(mesh).vertex(X_surface)
    H_surface = H_verts[v_idx].astype(np.float32)

    # Normalize so curvature loss is scale-invariant
    H_std     = float(np.std(H_surface)) + 1e-8
    H_surface = H_surface / H_std

    logging.info(f"Curvature target: min={H_surface.min():.3f}, "
                 f"max={H_surface.max():.3f}, std=1.0 (normalized)")

    return H_surface
