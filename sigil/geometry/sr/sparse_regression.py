# Sparse polynomial regression backend (default).
#
# Conceptually inspired by SINDy (feature library + L1 sparsity) but
# implemented directly with no dynamical systems machinery and no time
# series -- just L1-penalised least squares over a polynomial feature library.
#
# Warm start: accepts initial_alphas to seed gradient descent from a
# known-good equation (e.g. extracted via sympy_to_alphas from an adjacent patch).
