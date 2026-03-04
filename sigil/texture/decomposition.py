# Decompose texture into three layers:
#   Layer 1: Analytic   -- SR polynomial over surface coordinates
#   Layer 2: Procedural -- noise parameterisation (type + params, not samples)
#   Layer 3: Residual   -- conventional block compression (BC7/ASTC)
#
# Note: normal maps are redundant for SIGIL assets.
# The surface normal is grad(f(x,y,z)), computed analytically for free.
