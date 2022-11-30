import torch
import tensorly as tl
from tensorly.base import unfold
from tensorly.decomposition import partial_tucker
from tensorly.tenalg import multi_mode_dot, partial_svd


def tucker_to_tensor(core, factors, modes=None):
    return multi_mode_dot(core, factors, modes=modes, transpose=False)


def tucker(tensor: torch.Tensor, rank, modes=None, method='HOOI', svd='truncated_svd', tol=10e-5):
    if tl.backend.BackendManager.get_backend() != 'pytorch':
        tl.backend.BackendManager.set_backend('pytorch')
    if method == 'HOSVD':
        return tucker_HOSVD(tensor, rank=rank, modes=modes, svd=svd)
    elif method == 'HOOI':
        return tucker_HOOI(tensor, rank=rank, modes=modes, svd=svd, tol=tol)


def tucker_HOSVD(tensor, rank, modes=None, svd='truncated_svd'):
    if modes is None:
        modes = list(range(tl.ndim(tensor)))
    factors = []
    svd_fun = partial_svd
    for index, mode in enumerate(modes):
        eigenvecs, _, _ = svd_fun(unfold(tensor, mode), n_eigenvecs=rank[index])
        factors.append(eigenvecs)
    core = multi_mode_dot(tensor, factors, modes=modes, transpose=True)
    return core, factors


def tucker_HOOI(tensor, rank, modes=None, svd='truncated_svd', tol=10e-5):
    return partial_tucker(tensor, rank=rank, modes=modes, svd=svd, tol=tol)
