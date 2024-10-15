from abc import ABC, abstractmethod

from . import util


def get_constant_period_scheduler(period, offset=0):
    """
    Returns a constant period scheduler for operations that do not execute
    every iteration

    Parameters
    ----------
    period : int
        The period with which iterations become active
    offset : int
        The offset for the first active iteration
    """
    def fun(it):
        return 0 == ((it - offset) % period)
    return fun

class GeometricPeriodScheduler:
    def __init__(self, factor, period_init=1., offset=0):
        self.factor = factor
        self.prev_it_frac = 0.
        self.period_frac = float(period_init)
        self.offset = offset
        self._cache = [0]
    def __call__(self, it):
        if self.offset > it:
            return False
        while (it - self.offset) > self._cache[-1]:
            next_it_frac = self.prev_it_frac + self.period_frac
            next_it = int(next_it_frac)
            if next_it > self._cache[-1]:
                self._cache.append(next_it)
            self.prev_it_frac = next_it_frac
            self.period_frac *= self.factor
        return (it - self.offset) in self._cache
def get_geometric_period_scheduler(factor, period_init=1., offset=0):
    """
    Returns a geometric period scheduler for operations that do not execute
    every iteration

    Parameters
    ----------
    factor : float
        The factor by which the subsequent periods grow
    period_init : float
        The initial period between subsequent active iterations
    offset : int
        The offset for the first active iteration
    """
    return GeometricPeriodScheduler(factor, period_init, offset)


class GrassmannianAveragingAlgorithm(ABC):
    """
    The base class for a Grassmannian averaging algorithm
    """

    def __init__(self):
        self.U0 = None

    def set_U0(self, U0):
        self.U0 = U0
        return self

    def get_U0(self, U_arr):
        if self.U0 is None:
            self.U0 = util.get_standard_basis_like(U_arr[0])
        return self.U0

    def average(self, U_arr, max_iter=64, tol=None):
        """
        Averages the subspaces in U_arr

        Parameters
        ----------
        U_arr : tensor[M, N, K]
            A set of M subspace orthobases of size [N, K] to be averaged
        max_iter : int
            Maximum number of iterations to run
        tol : float
            The tolerance for algorithm stopping

        Returns
        -------
        U : tensor[N, K] or tensor[M, N, K]
            The average subspace orthobasis or set of M averaged subspaces for
            decentralized algorithms
        """
        gen = self.algo_iters(U_arr)
        for i in range(max_iter+1):
            iter_frame = next(gen)
            if tol is not None:
                if not hasattr(iter_frame, 'err_criterion'):
                    raise ValueError("tol can't be used with an algorithm not"
                                     " specifying err_criterion")
                if iter_frame.err_criterion < tol:
                    return iter_frame.U
        return iter_frame.U

    @abstractmethod
    def algo_iters(self, U_arr):
        """
        Returns a generator for running the algorithm ad infinitum.

        Parameters
        ----------
        U_arr : tensor[M, N, K]
            A set of M subspace orthobases of size [N, K] to be averaged.

        Yields
        -------
        iter_frame : SimpleNamespace
            A namespace of objects in the iteration's frame. Two names are
            prescribed at this level:
            U : tensor[N, K] or tensor[M, N, K]
                The subspace estimate at this iteration.
            err_criterion : float
                In the centralized case, this represents the error criterion
                for which the averaging algorithm may be stopped.
        """
        pass


class DecentralizedConsensusAlgorithm(GrassmannianAveragingAlgorithm):
    """
    The base class for a decentralized Grassmannian averaging algorithm
    """

    def __init__(self, consensus):
        """
        Parameters
        ----------
        consensus : ConsensusMethod
            The method by which consensus will be performed
        """
        super().__init__()
        self.consensus = consensus

    def get_U0(self, U_arr):
        if self.U0 is None:
            self.U0 = util.get_standard_basis_like(U_arr)
        return self.U0

