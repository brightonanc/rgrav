import torch
from types import SimpleNamespace
import random

from .algorithm_base import DecentralizedConsensusAlgorithm
from . import util


class GrassmannianGossip(DecentralizedConsensusAlgorithm):
    """ See https://arxiv.org/abs/1705.00467 """

    def __init__(self, edges, a, b, rounds):
        super().__init__(None)
        self.edges = tuple({frozenset(edge) for edge in edges})
        self.a = a
        self.b = b
        self.rounds = rounds

    def algo_iters(self, U_arr):
        it = 0
        ctr = 0
        iter_frame = SimpleNamespace()
        rng = random.Random(0)
        U = U_arr.clone()
        iter_frame.U = U
        yield iter_frame
        while True:
            it += 1
            iter_frame = SimpleNamespace()
            for _ in range(self.rounds):
                edges_shuffled = list(self.edges).copy()
                rng.shuffle(edges_shuffled)
                vert_set = set(range(U_arr.shape[0]))
                while vert_set and edges_shuffled:
                    m1, m2 = edges_shuffled.pop(0)
                    if (m1 not in vert_set) or (m2 not in vert_set):
                        # Agent is "busy", so skip
                        continue
                    vert_set.remove(m1)
                    vert_set.remove(m2)
                    gamma = self.a / (1 + (self.b * ctr))
                    grad1 = -util.grassmannian_log(U[m1], U[m2])
                    grad2 = -util.grassmannian_log(U[m2], U[m1])
                    U[m1] = util.grassmannian_exp(U[m1], -gamma * grad1)
                    U[m2] = util.grassmannian_exp(U[m2], -gamma * grad2)
                    ctr += 1
            iter_frame.U = U
            yield iter_frame

