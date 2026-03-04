# Abstract base class for all SR backends.
#
# class SRBackend(ABC):
#     def fit(self, X, y, initial_alphas=None) -> Equation: ...
#
# class Equation:
#     alphas: np.ndarray
#     feature_names: list[str]
#     sympy_expr: sympy.Expr
#     degree: int
#     rmse: float
