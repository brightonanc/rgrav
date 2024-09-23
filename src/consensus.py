from abc import ABC, abstractmethod
import torch

from .chebyshev import ChebyshevMagicNumbers
from . import util


class ConsensusMethod(ABC):
    """
    The base class for a decentralized average consensus method
    """
    @abstractmethod
    def __call__(self, X):
        """
        Computes average consensus

        Parameters
        ----------
        X : tensor[M, ...]
            Tensor to be averaged over the first dim

        Returns
        -------
        X_cons : tensor[M, ...]
            The consensus averaged tensors
        """
        pass

class SimpleConsensus(ConsensusMethod):
    """
    Implements simple memory-less linear average consensus
    """
    def __init__(self, comm_W, cons_rounds=8):
        """
        Parameters
        ----------
        comm_W : tensor[M, M]
            Symmetric consensus communication matrix such that the ones vector
            is an eigenvector (with eigenvalue 1) and the spectral radius of
            (comm_W - (1/M) 1 1^T) is strictly less than 1
        cons_rounds : int
            The number of rounds for consensus
        """
        self.comm_W = comm_W
        self.cons_rounds = cons_rounds
    def __call__(self, X):
        shape_X = X.shape
        X_ = X.reshape(shape_X[0], -1)
        for _ in range(self.cons_rounds):
            X_ = self.comm_W @ X_
        X_cons = X_.view(shape_X)
        return X_cons

class FastMixDeEPCA(ConsensusMethod):
    """
    Implements the FastMix algorithm as described in the DeEPCA paper
    """
    def __init__(self, comm_W, cons_rounds=8, lambda2=None):
        """
        Parameters
        ----------
        comm_W : tensor[M, M]
            Positive semidefinite symmetric consensus communication matrix such
            that the ones vector is an eigenvector (with eigenvalue 1) and the
            spectral radius of (comm_W - (1/M) 1 1^T) is strictly less than 1
        cons_rounds : int
            The number of rounds for consensus
        lambda2 : float
            The estimate of the spectral radius of (comm_W - (1/M) 1 1^T). If
            not supplied, it will be computed exactly from comm_W
        """
        self.comm_W = comm_W
        self.cons_rounds = cons_rounds
        if lambda2 is None:
            lambda2 = torch.linalg.eigvalsh(self.comm_W)[-2].abs()
        tmp = (1 - (lambda2**2))**0.5
        eta = (1 - tmp) / (1 + tmp)
        self._eta = eta
    def __call__(self, X):
        shape_X = X.shape
        X_ = X.reshape(shape_X[0], -1)
        prev_X_ = X_
        for _ in range(self.cons_rounds):
            next_X_ = ((1 + self._eta) * self.comm_W @ X_) \
                    - (self._eta * prev_X_)
            prev_X_ = X_
            X_ = next_X_
        X = X_.view(shape_X)
        return X

class ChebyshevConsensus(ConsensusMethod):
    """
    Implements an accelerated consensus procedure using Chebyshev polynomial
    approximations
    """
    def __init__(self, comm_W, cons_rounds=8, lambda2=None):
        """
        Parameters
        ----------
        comm_W : tensor[M, M]
            Positive semidefinite symmetric consensus communication matrix such
            that the ones vector is an eigenvector (with eigenvalue 1) and the
            spectral radius of (comm_W - (1/M) 1 1^T) is strictly less than 1
        cons_rounds : int
            The number of rounds for consensus
        lambda2 : float
            The estimate of the spectral radius of (comm_W - (1/M) 1 1^T). If
            not supplied, it will be computed exactly from comm_W
        """
        self.comm_W = comm_W
        self.cons_rounds = cons_rounds
        if lambda2 is None:
            lambda2 = torch.linalg.eigvalsh(self.comm_W)[-2].item()
        self.cmn = ChebyshevMagicNumbers(lambda2)
    def __call__(self, X):
        shape_X = X.shape
        X_ = X.reshape(shape_X[0], -1)
        for it in range(1, self.cons_rounds+1):
            if 1 == it:
                next_X_ = self.comm_W @ X_
            else:
                a = self.cmn.a(it)
                b = self.cmn.b(it)
                c = self.cmn.c(it)
                next_X_ = a * ((self.comm_W @ X_) + (b * X_) + (c * prev_X_))
            prev_X_ = X_
            X_ = next_X_
        X = X_.view(shape_X)
        return X

