import sys
import torch
import kan.spline
import kan.MultKAN
import kan.KANLayer
import kan.Symbolic_KANLayer
from kan.spline import B_batch

_TARGETS = ("kan.spline", "kan.MultKAN", "kan.KANLayer")


def curve2coef(x_eval, y_eval, grid, k, lamb=1e-8):
    mat = B_batch(x_eval, grid, k)
    n_coef = grid.shape[1] - k - 1
    X = mat.permute(1, 0, 2)
    Y = y_eval.permute(1, 0, 2)
    XtX = X.transpose(1, 2) @ X
    Xty = X.transpose(1, 2) @ Y
    eye = torch.eye(n_coef, device=mat.device, dtype=mat.dtype).expand_as(XtX)
    coef = torch.linalg.pinv(XtX + lamb * eye) @ Xty
    return coef.permute(0, 2, 1).contiguous()


_SymbolicLayer = sys.modules["kan.Symbolic_KANLayer"].Symbolic_KANLayer
_symbolic_forward = _SymbolicLayer.forward


# pykan loops over every edge in python even where the symbolic mask is all zero; with symbolic_enabled on for the
# fixed stoich layer that is ~1000 dead edge calls per forward, so layers with no symbolic edge return zeros.
def symbolic_forward(self, x, singularity_avoiding=False, y_th=10.):
    if not bool(self.mask.any()):
        return (x.new_zeros(x.shape[0], self.out_dim),
                x.new_zeros(x.shape[0], self.out_dim, self.in_dim))
    return _symbolic_forward(self, x, singularity_avoiding, y_th)


def install():
    for name in _TARGETS:
        sys.modules[name].curve2coef = curve2coef
    _SymbolicLayer.forward = symbolic_forward
    print("patched pykan curve2coef (rank-safe ridge solve; cuda gels workaround), "
          "symbolic layers with no fixed edge skipped")
