# [FUTURE] Differentiable tree SR backend.
#
# Relaxes discrete operator choices at each tree node to continuous weights,
# runs gradient descent over the relaxed space, then discretizes (DARTS-style).
# Same interface as all other backends: fit(X, y, initial_alphas) -> Equation.
#
# References:
#   DGP (2023) Differentiable Genetic Programming, arXiv:2304.08915
#   DDSR-NN (2024) Deep Differentiable SR Neural Network
