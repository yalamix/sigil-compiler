# Track sinusoidal partials over time.
# Each partial: amplitude envelope A(t), frequency envelope f(t), phase offset.
# SR finds compact symbolic expressions for A(t) and f(t).
#
# Ray-traced audio note:
#   Occlusion: evaluate sign of f(listener) and f(source) from .sof geometry.
#   Sign change = occluded. O(1), no BVH traversal, no precomputed data.
#   Reflections use grad(f) as surface normal -- also free.
