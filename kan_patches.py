import sys
import torch
import kan.spline
import kan.MultKAN
import kan.KANLayer
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


def install():
    for name in _TARGETS:
        sys.modules[name].curve2coef = curve2coef
    print("patched pykan curve2coef (rank-safe ridge solve; cuda gels workaround)")
