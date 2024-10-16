import pytest
import torch

from src.algorithm_base import get_constant_period_scheduler, \
        get_geometric_period_scheduler

@pytest.mark.parametrize('period, offset, expected', (
    (1, 0, range(100)),
    (2, 0, range(0, 100, 2)),
    (2, 1, range(1, 100, 2)),
    (7, 3, range(3, 100, 7)),
))
def test_constant_period_scheduler(period, offset, expected):
    scheduler = get_constant_period_scheduler(period, offset)
    res = [x for x in range(100) if scheduler(x)]
    assert tuple(res) == tuple(expected)

@pytest.mark.parametrize('factor, period, offset, expected', (
    (1, 1, 0, range(100)),
    (2, 1, 0, [(2**x)-1 for x in range(7)]),
    (2, 1, 1, [2**x for x in range(7)]),
    (2, 2, 0, [0, 2, 6, 14, 30, 62]),
    (2, 2, 2, [2*(2**x) for x in range(6)]),
    (2, 3, 0, [0, 3, 9, 21, 45, 93]),
    (2, 3, 3, [3*(2**x) for x in range(6)]),
    (3, 1, 0, [0, 1, 4, 13, 40]),
    (1.7, 1, 0, [0, 1, 2, 5, 10, 18, 33, 57, 98]),
))
def test_geometric_period_scheduler(factor, period, offset, expected):
    scheduler = get_geometric_period_scheduler(factor, period, offset)
    res = [x for x in range(100) if scheduler(x)]
    assert tuple(res) == tuple(expected)
