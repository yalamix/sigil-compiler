# Learnable polynomial blend w(x,y,z).
# Jointly optimises [alpha_A, alpha_B, w_coefficients] via gradient descent.
# Preserves polynomial-in-t property -> analytical ray intersection.
#
# Loss terms:
#   MSE(f_AB, gpr_values)
#   + lambda_sparse * (||alpha_A|| + ||alpha_B||)
#   + lambda_crisp  * mean(w * (1 - w))   <- push toward 0 or 1
