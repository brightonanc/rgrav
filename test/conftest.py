import pytest
import torch
import numpy as np

from src import util

@pytest.fixture(
    scope='session',
    params=[
        (2, 1, 2, 0.5 * (0.25*torch.pi)),
        (2, 1, 4, 0.5 * (0.5*torch.pi)),
        (2, 1, 128, 0.5 * (0.5*torch.pi)),
        (13, 1, 2, 0.5 * (0.5*torch.pi)),
        (13, 1, 4, 0.5 * (0.5*torch.pi)),
        (13, 1, 128, 0.5 * (0.5*torch.pi)),
        (4, 2, 2, 0.5 * (0.5*torch.pi)),
        (4, 2, 4, 0.5 * (0.5*torch.pi)),
        (4, 2, 128, 0.5 * (0.5*torch.pi)),
        (19, 2, 2, 0.5 * (0.5*torch.pi)),
        (19, 2, 4, 0.5 * (0.5*torch.pi)),
        (19, 2, 128, 0.5 * (0.5*torch.pi)),
        (32, 10, 2, 0.5 * (0.5*torch.pi)),
        (32, 10, 4, 0.5 * (0.5*torch.pi)),
        (32, 10, 128, 0.5 * (0.5*torch.pi)),
    ],
)
def U_arr(request):
    N, K, M, radius = request.param
    np.random.seed(0)
    torch.manual_seed(0)
    return util.get_random_clustered_grassmannian_points(
        N=N,
        K=K,
        M=M,
        radius=radius,
    )
