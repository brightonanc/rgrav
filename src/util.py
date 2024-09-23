import torch
from scipy.stats import ortho_group
import warnings


def get_standard_basis(N, K, dtype=None, dev=None):
    """
    Gets the standard orthobasis for [N, K] space

    Parameters
    ----------
    N : int
    K : int
    dtype : torch.dtype
    dev : torch.device

    Returns
    -------
    U : tensor[N, K]
    """
    U = torch.zeros(N, K, dtype=dtype, device=dev)
    U[range(K), range(K)] = 1.
    return U

def get_standard_basis_like(X):
    """
    Convenience function to wrap get_standard_basis and expand
    """
    rv = get_standard_basis(*X.shape[-2:], dtype=X.dtype, dev=X.device)
    rv = rv.expand(X.shape)
    return rv

def get_orthobasis(X, mode='qr-stable', others_X=None, return_S=False):
    """
    Returns an orthobasis corresponding to the column space of a given matrix

    Parameters
    ----------
    X : tensor[..., N, K]
        The matrix whose columns are used to compute the orthobasis
    mode : str
        The mode by which the orthobasis is computed
    others : iterable[tensor[..., N_o, K]]
        A collection of tensors which should be modified by the same right-side
        linear transformations as X is. N_o may be different for each tensor in
        others
    return_S : bool
        If True, this function will also return the matrix S as the last value.
        S is such that U = X @ S.

    Returns
    -------
    U : tensor[..., N, K]
        A representative orthobasis
    others_U : iterable[tensor[..., N_o, K]]
        A collection of tensors modified by the same right-side linear
        transformations as X was
    """
    assert X.shape[-2] >= X.shape[-1]
    K = X.shape[-1]
    match mode:
        case 'qr':
            U, R = torch.linalg.qr(X)
            d_R = R[..., range(K), range(K)]
            if d_R.abs().min() < torch.finfo(d_R.dtype).resolution:
                warnings.warn(
                    f'util.get_orthobasis: {mode=} applied on a degenerate'
                    ' matrix'
                )
            if others_X:
                others_U = [
                    torch.linalg.solve_triangular(
                        R,
                        other_X,
                        upper=True,
                        left=False,
                    ) for other_X in others_X
                ]
            if return_S:
                I = torch.eye(K)
                S = torch.linalg.solve_triangular(R, I, upper=True, left=False)
        case 'qr-stable':
            Q, R = torch.linalg.qr(X)
            sd_R = R[..., range(K), range(K)]
            if sd_R.abs().min() < torch.finfo(sd_R.dtype).resolution:
                warnings.warn(
                    f'util.get_orthobasis: {mode=} applied on a degenerate'
                    ' matrix'
                )
            sd_R = sd_R.sign()
            U = Q * sd_R[..., None, :]
            if others_X or return_S:
                R *= sd_R[..., :, None]
                if others_X:
                    others_U = [
                        torch.linalg.solve_triangular(
                            R,
                            other_X,
                            upper=True,
                            left=False,
                        ) for other_X in others_X
                    ]
                if return_S:
                    I = torch.eye(K)
                    S = torch.linalg.solve_triangular(
                        R,
                        I,
                        upper=True,
                        left=False
                    )
        case 'svd':
            U_, s_, Vh_ = torch.linalg.svd(X, full_matrices=False)
            if s_.min() < torch.finfo(s_.dtype).resolution:
                warnings.warn(
                    f'util.get_orthobasis: {mode=} applied on a degenerate'
                    ' matrix'
                )
            U = U_ @ Vh_
            if others_X or return_S:
                S = Vh_.mT @ ((1. / s_)[..., None] * Vh_)
                if others_X:
                    others_U = [other_X @ S for other_X in others_X]
        case _:
            raise ValueError(f'{mode=} not recognized')
    rv_arr = [U]
    if others_X is not None:
        rv_arr.append(others_U)
    if return_S:
        rv_arr.append(S)
    if 1 == len(rv_arr):
        return rv_arr[0]
    else:
        return tuple(rv_arr)

def grassmannian_dist(U1, U2, assume_ortho=False):
    """
    Computes the Grassmannian geodesic distance between two orthobases

    Parameters
    ----------
    U1 : tensor[..., N, K]
    U2 : tensor[..., N, K]
    assume_ortho : bool
        If True, U1 and U2 are used to already be orthobases and
        orthonormalization is skipped

    Returns
    -------
    dist : tensor[...]
    """
    if not assume_ortho:
        U1 = get_orthobasis(U1)
        U2 = get_orthobasis(U2)
    c = torch.linalg.svdvals(U2.mT @ U1)
    c[1. < c] = 1.
    theta = torch.acos(c)
    return torch.linalg.norm(theta, dim=-1)

def grassmannian_log(U1, U2):
    """
    Computes the Grassmannian logarithm between two orthobases

    Parameters
    ----------
    U1 : tensor[..., N, K]
    U2 : tensor[..., N, K]

    Returns
    -------
    tang : tensor[..., N, K]
        A tangent vector at U1 such that U1.mT @ tang is all zeros
    """
    tmp = (U2 @ torch.linalg.inv(U1.mT @ U2)) - U1
    U, s, Vh = torch.linalg.svd(tmp, full_matrices=False)
    return U @ (torch.atan(s)[..., None] * Vh)

def grassmannian_exp(U, tang):
    """
    Computes the Grassmannian exponential map from an orthobasis to an
    orthobasis

    Parameters
    ----------
    U : tensor[..., N, K]
    tang : tensor[..., N, K]

    Returns
    -------
    U_ : tensor[..., N, K]
    """
    U_, s_, Vh = torch.linalg.svd(tang, full_matrices=False)
    c = torch.cos(s_)
    s = torch.sin(s_)
    return (U @ (Vh.mT * c[..., None, :])) + (U_ * s[..., None, :])

def grassmannian_linfty_dist(U1, U2):
    """
    Computes the l-infinity Grassmannian distance between two orthobases

    Parameters
    ----------
    U1 : tensor[..., N, K]
    U2 : tensor[..., N, K]

    Returns
    -------
    dist : tensor[...]
    """
    c = torch.linalg.svdvals(U2.mT @ U1)
    c_min = c[..., -1]
    c_min[1. < c_min] = 1.
    theta_max = torch.acos(c_min)
    return theta_max

def get_random_clustered_grassmannian_points(N, K, M, radius, Q_center=None):
    """
    Computes random Grassmannian points as orthobases within a particular
    Grassmannian distance radius

    Parameters
    ----------
    N : int
    K : int
    M : int
        Number of random sample points
    radius:
        The radius around which random samples may be generated
    Q_center : tensor[N, K]
        The center point of the iteratively-generated distributed

    Returns
    -------
    U_arr : tensor[M, N, K]
        A collection of M orthobases on the Grassmannian
    """
    if (0.25*torch.pi) <= radius:
        print(f'WARNING: {radius=} may incur points without unique Frechet'
               ' mean')
    if Q_center is None:
        Q_center = torch.from_numpy(ortho_group.rvs(N))
    sigma = radius * torch.rand(M, K)
    c = torch.cos(sigma)
    s = torch.sin(sigma)
    if 1 < K:
        V = torch.from_numpy(ortho_group.rvs(K, size=M))
    else:
        V = 1 - (2 * torch.randint(2, (M, 1, 1)).type(torch.float64))
    if 1 < (N-K):
        U = torch.from_numpy(ortho_group.rvs(N-K, size=M)[:, :, :K])
    else:
        U = 1 - (2 * torch.randint(2, (M, 1, 1)).type(torch.float64))
    U_arr = (Q_center[:, :K] @ (V * c[:, None, :])) \
            + (Q_center[:, K:] @ (U * s[:, None, :]))
    return U_arr

