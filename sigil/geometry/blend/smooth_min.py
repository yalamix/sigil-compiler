# Smooth minimum blend (Inigo Quilez formulation).
# smin(a, b, k) = a - (max(k - |a-b|, 0))^2 / (4k)
# Not polynomial -> Newton's method required for ray intersection.
# Produces the most natural-looking joins for organic surfaces.
