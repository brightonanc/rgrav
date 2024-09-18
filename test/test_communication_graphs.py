import pytest
import torch

from src.communication_graphs import *


@pytest.mark.parametrize('edges, expected', [
    (
        [(0,1)],
        torch.tensor([
            [0., 1],
            [1, 0]
        ]),
    ), (
        [(0,1), (1, 0)],
        torch.tensor([
            [0., 1],
            [1, 0]
        ]),
    ), (
        [(0,1), (1,2), (3,2), (3,0)],
        torch.tensor([
            [0., 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ]),
    )
])
def test_adjacency_matrix_from_edges(edges, expected):
    A = adjacency_matrix_from_edges(edges)
    print(f'{A=}')
    assert torch.all(A == expected)

@pytest.mark.parametrize('num_vertices, ord, expected', [
    (2, '2', (1/2) * torch.tensor([
        [1., 1],
        [1, 1],
    ])),
    (2, 2, (1/2) * torch.tensor([
        [1., 1],
        [1, 1],
    ])),
    (2, 'inf', (1/2) * torch.tensor([
        [1., 1],
        [1, 1],
    ])),
    (3, '2', (1/5) * torch.tensor([
        [1., 2, 2],
        [2, 3, 0],
        [2, 0, 3],
    ])),
    (3, 2, (1/5) * torch.tensor([
        [1., 2, 2],
        [2, 3, 0],
        [2, 0, 3],
    ])),
    (3, 'inf', (1/2) * torch.tensor([
        [0., 1, 1],
        [1, 1, 0],
        [1, 0, 1],
    ])),
    (8, '2', (1/5) * torch.tensor([
        [-2., 1, 1, 1, 1, 1, 1, 1],
        [1, 4, 0, 0, 0, 0, 0, 0],
        [1, 0, 4, 0, 0, 0, 0, 0],
        [1, 0, 0, 4, 0, 0, 0, 0],
        [1, 0, 0, 0, 4, 0, 0, 0],
        [1, 0, 0, 0, 0, 4, 0, 0],
        [1, 0, 0, 0, 0, 0, 4, 0],
        [1, 0, 0, 0, 0, 0, 0, 4],
    ])),
    (8, 'inf', (1/9) * torch.tensor([
        [-5., 2, 2, 2, 2, 2, 2, 2],
        [2, 7, 0, 0, 0, 0, 0, 0],
        [2, 0, 7, 0, 0, 0, 0, 0],
        [2, 0, 0, 7, 0, 0, 0, 0],
        [2, 0, 0, 0, 7, 0, 0, 0],
        [2, 0, 0, 0, 0, 7, 0, 0],
        [2, 0, 0, 0, 0, 0, 7, 0],
        [2, 0, 0, 0, 0, 0, 0, 7],
    ])),
])
def test_StarGraph_get_optimal_lapl_based_comm_W(num_vertices, ord, expected):
    res = StarGraph.get_optimal_lapl_based_comm_W(num_vertices, ord=ord)
    assert 1e-6 > (res - expected).abs().max()

@pytest.mark.parametrize('hc_dim, expected', [
    (1, (1/2) * torch.tensor([
        [1., 1],
        [1, 1],
    ])),
    (2, (1/3) * torch.tensor([
        [1., 1, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [0, 1, 1, 1],
    ])),
    (3, (1/4) * torch.tensor([
        [1., 1, 1, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 1, 0, 0],
        [1, 0, 1, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 1, 1, 0],
        [0, 1, 0, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 1, 1, 1],
    ])),
])
def test_HypercubeGraph_get_optimal_lapl_based_comm_W(hc_dim, expected):
    res = HypercubeGraph.get_optimal_lapl_based_comm_W(hc_dim)
    assert 1e-6 > (res - expected).abs().max()
