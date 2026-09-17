import sys
import torch
import kan.spline
import kan.MultKAN
import kan.KANLayer
from kan.spline import B_batch

# pykan 0.2.8 refits spline coefficients with a bare torch.linalg.lstsq. after
# prune() most of those systems are rank-deficient (measured here: 152 of 219,
# worst had rank 1 of 13 needed). lstsq's default driver is gelsy on cpu, which
# copes, but cuda only offers gels, which assumes full rank and silently returns
# inf/nan -- so on gpu the whole model turns to nan at the first refine().
# pykan 0.2.7 used a ridge-regularised pseudo-inverse here and 0.2.8 replaced it;
# this restores that. pinv takes the same path on cpu and cuda, and unlike solve()
# it stays defined when the system is rank-deficient.
# upstream reports: KindXiaoming/pykan#498, #416
_TARGETS = ("kan.spline", "kan.MultKAN", "kan.KANLayer")


def curve2coef(x_eval, y_eval, grid, k, lamb=1e-8):
    mat = B_batch(x_eval, grid, k)
    n_coef = grid.shape[1] - k - 1
    X = mat.permute(1, 0, 2)
    Y = y_eval.permute(1, 0, 2)
    XtX = X.transpose(1, 2) @ X
    Xty = X.transpose(1, 2) @ Y
    eye = torch.eye(n_coef, device=mat.device, dtype=mat.dtype).expand_as(XtX)
    # pinverse rather than solve: after prune some of these systems are rank 1 of
    # 13, and a 1e-8 ridge sits below float32 resolution, so solve() still reports
    # a singular matrix. the pseudo-inverse is defined at any rank, and matches
    # what 0.2.7 did here
    coef = torch.linalg.pinv(XtX + lamb * eye) @ Xty
    return coef.permute(0, 2, 1).contiguous()


def install():
    # each module resolves curve2coef from its own globals, and kan/__init__.py
    # rebinds the name kan.MultKAN to the class, so go through sys.modules
    for name in _TARGETS:
        sys.modules[name].curve2coef = curve2coef
    print("patched pykan curve2coef (rank-safe ridge solve; cuda gels workaround)")
