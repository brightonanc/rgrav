import pytest
import torch

from src.consensus import SimpleConsensus, FastMixDeEPCA, \
        ChebyshevConsensus, BalancedChebyshevConsensus
from src import HypercubeGraph

@pytest.fixture(
    scope='module',
    params=[2, 3, 4, 10],
)
def comm_W(request):
    hc_dim = request.param
    return HypercubeGraph.get_optimal_lapl_based_comm_W(hc_dim)

@pytest.fixture(
    scope='module',
    params=[2, 3, 4, 10],
)
def comm_W_psd(request):
    hc_dim = request.param
    return HypercubeGraph.get_positive_optimal_lapl_based_comm_W(hc_dim)

def test_SimpleConsensus(comm_W):
    res = SimpleConsensus(comm_W, 64)(torch.eye(comm_W.shape[-1]))
    res = res - res.mean(-2, keepdim=True)
    assert torch.finfo(res.dtype).resolution > res.abs().max()

def test_FastMixDeEPCA(comm_W_psd):
    res = FastMixDeEPCA(comm_W_psd, 32)(torch.eye(comm_W_psd.shape[-1]))
    res = res - res.mean(-2, keepdim=True)
    assert torch.finfo(res.dtype).resolution > res.abs().max()

def test_ChebyshevConsensus(comm_W_psd):
    res = ChebyshevConsensus(comm_W_psd, 32)(torch.eye(comm_W_psd.shape[-1]))
    res = res - res.mean(-2, keepdim=True)
    assert torch.finfo(res.dtype).resolution > res.abs().max()

def test_BalancedChebyshevConsensus(comm_W):
    res = BalancedChebyshevConsensus(comm_W, 32)(torch.eye(comm_W.shape[-1]))
    res = res - res.mean(-2, keepdim=True)
    assert torch.finfo(res.dtype).resolution > res.abs().max()
