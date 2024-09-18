import torch
from scipy.stats import ortho_group


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

def get_orthobasis(X, mode='qr-stable'):
    """
    Returns an orthobasis corresponding to the column space of a given matrix

    Parameters
    ----------
    X : tensor[..., N, K]
        The matrix whose columns are used to compute the orthobasis
    mode : str
        The mode by which the orthobasis is computed

    Returns
    -------
    U : tensor[..., N, K]
        A representative orthobasis
    """
    assert X.shape[-2] >= X.shape[-1]
    match mode:
        case 'qr':
            return torch.linalg.qr(X).Q
        case 'svd':
            U_, _, Vh_ = torch.linalg.svd(X, full_matrices=False)
            return U_ @ Vh_
        case 'qr-stable':
            Q, R = torch.linalg.qr(X)
            K = R.shape[-1]
            return Q * R[..., None, range(K), range(K)].sign()
        case _:
            raise ValueError(f'{mode=} not recognized')

def grassmannian_dist(U1, U2):
    """
    Computes the Grassmannian geodesic distance between two orthobases

    Parameters
    ----------
    U1 : tensor[..., N, K]
    U2 : tensor[..., N, K]

    Returns
    -------
    dist : tensor[...]
    """
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

