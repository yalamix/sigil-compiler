# sigil/pipeline/balloon_nn.py

import logging
import numpy as np
import torch
import torch.nn as nn
import trimesh
import skimage.measure
from dataclasses import dataclass
from pathlib import Path

from sigil.geometry.scalar_field import sample_mesh_sdf
from sigil.geometry.sr.base import Equation
from sigil.geometry.sr.sparse_regression import _normalize_X
import sympy

OUTPUT_DIR = Path(r'C:\Users\yalam\Documents\sigil-compiler\outputs\balloon_progress')

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class NNConfig:
    # Sampling
    n_surface:      int   = 30000
    epsilon:        float = 0.05

    # Architecture
    hidden_dim:     int   = 128
    n_layers:       int   = 5        # total layers including input/output
    activation:     str   = 'relu'   # 'relu', 'sine', 'tanh'

    # Training
    n_epochs:       int   = 110000
    lr:             float = 1e-4
    lam_reg:        float = 1e-5
    lambda_sign:    float = 1.0
    batch_size:     int   = 0        # 0 = full batch

    # Convergence
    rmse_threshold: float = 0.01

    # Output
    visualize_every:   int  = 10000    # save visualization every N epochs
    visualize_progress: bool = True


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

def _init_as_sphere(model, r_norm):
    """
    Initialize the network to approximate a sphere of radius r_norm.
    Sets the last layer weights/bias so f(x) ≈ x0²+x1²+x2²-r²
    at initialization, giving a valid closed surface from step 0.
    """
    # Freeze all layers except last, do a quick least squares fit
    # of the last layer to match sphere values on random points
    import torch
    device = next(model.parameters()).device
    
    rng = np.random.default_rng(42)
    X_probe = rng.standard_normal((2000, 3)).astype(np.float32)
    y_sphere = (X_probe ** 2).sum(axis=1) - r_norm ** 2
    
    X_t = torch.tensor(X_probe, device=device)
    y_t = torch.tensor(y_sphere, device=device)
    
    # Get penultimate layer activations
    with torch.no_grad():
        h = X_t
        for layer in model.layers[:-1]:
            h = layer(h)
            if model.activation_name == 'sine':
                h = torch.sin(30.0 * h)
            else:
                h = model.act(h)
        H = h.cpu().numpy()  # (2000, hidden_dim)
    
    # Least squares: H @ w + b ≈ y_sphere
    H_aug = np.hstack([H, np.ones((len(H), 1))])
    wb, _, _, _ = np.linalg.lstsq(H_aug, y_sphere, rcond=None)
    
    with torch.no_grad():
        model.layers[-1].weight.copy_(
            torch.tensor(wb[:-1], dtype=torch.float32, device=device).unsqueeze(0)
        )
        model.layers[-1].bias.copy_(
            torch.tensor([wb[-1]], dtype=torch.float32, device=device)
        )


class ImplicitMLP(nn.Module):
    def __init__(self, hidden_dim=64, n_layers=4, activation='relu'):
        super().__init__()

        act_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sine': None,   # handled separately
        }
        self.activation_name = activation

        layers = [nn.Linear(3, hidden_dim)]
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, 1))
        self.layers = nn.ModuleList(layers)

        if activation != 'sine':
            self.act = act_map[activation]
        else:
            self.act = None

        self._init_weights()

    def _init_weights(self):
        for i, layer in enumerate(self.layers[:-1]):
            if self.activation_name == 'sine':
                # SIREN initialization
                w_std = (1.0 / layer.in_features if i == 0
                         else np.sqrt(6.0 / layer.in_features))
                nn.init.uniform_(layer.weight, -w_std, w_std)
            else:
                nn.init.kaiming_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            if self.activation_name == 'sine':
                x = torch.sin(30.0 * x)
            else:
                x = self.act(x)
        return self.layers[-1](x).squeeze(-1)


# ---------------------------------------------------------------------------
# Save/load weights in simple binary format (no PyTorch dependency to load)
# ---------------------------------------------------------------------------

def save_nn_sof(model, config, center, scale, path):
    """
    Save MLP weights in a simple binary format readable without PyTorch.
    Format:
        header: magic(4) + version(1) + n_layers(1) + hidden_dim(2) +
                activation(1) + center(3*f32) + scale(f32)
        weights: for each layer: rows(2) + cols(2) + W(rows*cols*f32) + b(cols*f32)
    """
    import struct

    ACTIVATIONS = {'relu': 0, 'tanh': 1, 'sine': 2}

    with open(path, 'wb') as f:
        # Magic + header
        f.write(b'SIGIL')
        f.write(struct.pack('B', 1))                          # version
        f.write(struct.pack('B', config.n_layers))
        f.write(struct.pack('H', config.hidden_dim))
        f.write(struct.pack('B', ACTIVATIONS[config.activation]))
        f.write(struct.pack('3f', *center.astype(np.float32)))
        f.write(struct.pack('f', float(scale)))

        # Weights
        for layer in model.layers:
            W = layer.weight.detach().cpu().numpy().astype(np.float32)
            b = layer.bias.detach().cpu().numpy().astype(np.float32)
            rows, cols = W.shape
            f.write(struct.pack('HH', rows, cols))
            f.write(W.tobytes())
            f.write(b.tobytes())

    size_kb = path.stat().st_size / 1024
    logging.info(f"Saved NN weights: {path} ({size_kb:.1f} KB)")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _visualize_nn(model, center, scale, mesh, rmse, epoch, device):
    try:
        logging.info("Saving NN visualization...")
        res = 64
        lin = np.linspace(-1.5, 1.5, res)
        xx, yy, zz = np.meshgrid(lin, lin, lin, indexing='ij')
        X_grid = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        with torch.no_grad():
            X_t   = torch.tensor(X_grid, dtype=torch.float32, device=device)
            f_grid = model(X_t).cpu().numpy().reshape(res, res, res)

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

        out_path = OUTPUT_DIR / f'nn_epoch_{epoch:05d}_rmse_{rmse:.4f}.png'
        with open(out_path, 'wb') as f:
            f.write(png)
        logging.info(f"Saved: {out_path}")
    except Exception as e:
        logging.info(f"Visualization failed: {e}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def compile_mesh_nn(mesh, config=None):
    if config is None:
        config = NNConfig()

    logging.info("compile_mesh_nn: starting")
    logging.info(f"Config: {config}")

    # Stage 1: sample
    logging.info(f"Stage 1: sampling ({config.n_surface} surface pts)")
    X, y = sample_mesh_sdf(mesh, n_surface=config.n_surface, epsilon=config.epsilon, n_volume=config.n_surface//2)
    logging.info(f"Dataset: {len(X)} points")

    X_norm, center, scale = _normalize_X(X)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")

    X_t = torch.tensor(X_norm, dtype=torch.float32, device=device)
    y_t = torch.tensor(y,      dtype=torch.float32, device=device)

    n_surface = config.n_surface

    # Stage 2: build model
    model = ImplicitMLP(
        hidden_dim = config.hidden_dim,
        n_layers   = config.n_layers,
        activation = config.activation,
    ).to(device)
    r_world = float(np.max(np.linalg.norm(X, axis=1))) * 1.1
    r_norm  = r_world / scale
    _init_as_sphere(model, r_norm)    

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=500, min_lr=1e-6
    )

    # Stage 3: training loop
    logging.info("Stage 3: training")
    best_rmse = np.inf

    for epoch in range(config.n_epochs):
        model.train()
        optimizer.zero_grad()

        if config.batch_size > 0:
            idx   = torch.randperm(len(X_t))[:config.batch_size]
            X_b   = X_t[idx]
            y_b   = y_t[idx]
            # ensure surface points are always included
            surf_idx = torch.arange(n_surface, device=device)
            X_b  = torch.cat([X_t[:n_surface], X_b])
            y_b  = torch.cat([y_t[:n_surface], y_b])
        else:
            X_b, y_b = X_t, y_t

        f_vals = model(X_b)

        L_surf = torch.mean(f_vals[:n_surface] ** 2)

        f_off  = f_vals[n_surface:]
        y_off  = y_b[n_surface:]
        L_sign = torch.mean(torch.relu(-f_off * torch.sign(y_off)) ** 2)

        L_reg  = sum(p.pow(2).sum() for p in model.parameters())
        # loss   = L_surf + config.lambda_sign * L_sign + config.lam_reg * L_reg
        loss   = L_surf + config.lambda_sign * L_sign 

        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

        if epoch % 200 == 0:
            model.eval()
            with torch.no_grad():
                f_all  = model(X_t)
                rmse   = float(torch.sqrt(torch.mean((f_all - y_t) ** 2)))
            logging.info(
                f"  epoch {epoch:5d}: surf={L_surf.item():.6f}  "
                f"sign={L_sign.item():.6f}  "
                f"reg={L_reg.item():.6f}  "
                f"rmse={rmse:.6f}"
            )
            if rmse < best_rmse:
                best_rmse = rmse

            if rmse < config.rmse_threshold:
                logging.info(f"Converged at epoch {epoch}")
                break

        if config.visualize_progress and epoch % config.visualize_every == 0:
            model.eval()
            with torch.no_grad():
                f_all = model(X_t)
                rmse  = float(torch.sqrt(torch.mean((f_all - y_t) ** 2)))
            _visualize_nn(model, center, scale, mesh, rmse, epoch, device)

    # Stage 4: save weights
    weights_path = OUTPUT_DIR / 'nn_weights.sigil'
    save_nn_sof(model, config, center, scale, weights_path)

    # Final RMSE
    model.eval()
    with torch.no_grad():
        f_final   = model(X_t)
        final_rmse = float(torch.sqrt(torch.mean((f_final - y_t) ** 2)))
    logging.info(f"Final RMSE: {final_rmse:.6f}")

    # Wrap in Equation with a lambda as sympy_expr placeholder
    # Real evaluation goes through model, not sympy
    x0, x1, x2 = sympy.symbols('x0 x1 x2')
    sympy_expr  = sympy.Integer(0)  # placeholder — use model for actual eval

    eq = Equation(
        sympy_expr    = sympy_expr,
        rmse          = final_rmse,
        degree        = 0,
        alphas        = None,
        feature_names = None,
    )
    eq.model  = model
    eq.center = center
    eq.scale  = scale
    return eq