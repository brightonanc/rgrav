import torch

def adjacency_matrix_from_edges(edges, num_vertices=None, dtype=None, dev=None):
    """
    Returns the adjacency matrix for a set of edges

    Parameters
    ----------
    edges : iterable[iterable[len=2][int]]
        The edges of the graph represented as an iterable of pairs of vertices
    num_vertices : int
        The number of vertices in this graph. By default, this is assumed from
        the edges
    dtype : torch.dtype
    dev : torch.device

    Returns
    -------
    A : tensor[num_vertices, num_vertices]
        The adjacency matrix corresponding to the edges, where self-loops are
        not included by default (but may be specified with edges)
    """
    edge_set = {frozenset(edge) for edge in edges}
    if num_vertices is None:
        num_vertices = max(max(edge) for edge in edge_set)+1
    A = torch.zeros((num_vertices, num_vertices), dtype=dtype, device=dev)
    for edge in edge_set:
        i, j = sorted(edge)
        A[i, j] = 1
        A[j, i] = 1
    return A

class StarGraph:
    @staticmethod
    def _get_edges(num_vertices):
        """
        Returns the edges of a star graph

        Parameters
        ----------
        num_vertices : int

        Returns
        -------
        edges : tuple[tuple[int, int]]
        """
        return tuple((0, x) for x in range(1, num_vertices))
    @classmethod
    def get_optimal_lapl_based_comm_W(cls, num_vertices, ord='inf', dtype=None, dev=None):
        """
        Returns the optimal laplacian-based communication matrix W

        Parameters
        ----------
        num_vertices : int
        ord : str
            (default='inf') The ord used for optimality.
                * 'inf': Minimizes the operator norm of W - (1 1^T / M)
                * '2': Minimizes the Frobenius norm of W - (1 1^T / M)

        Returns
        -------
        W : tensor[num_vertices, num_vertices]
            The optimal laplacian-based communication matrix W. W is such that
            W 1 = 1 and the ord-based quantity is minimized all choice of
            laplacian-based communication matrices.
        """
        A = adjacency_matrix_from_edges(
            cls._get_edges(num_vertices),
            dtype=dtype,
            dev=dev,
        )
        match ord:
            case 'inf':
                if 2 < num_vertices:
                    r = (num_vertices+1)/(2*(num_vertices-1))
                else:
                    r = 2.
            case '2' | 2:
                r = (num_vertices+2)/(2*(num_vertices-1))
            case _:
                raise ValueError(f'Unrecognized {ord=}')
        d = A.sum(-1)
        d_max = d.max()
        D = d.diag_embed()
        L = D - A
        I = torch.eye(A.shape[-1], dtype=dtype, device=dev)
        W = I - ((1/(r*d_max)) * L)
        return W

class HypercubeGraph:
    @staticmethod
    def _get_edges(hc_dim):
        """
        Returns the edges of a hypercube graph

        Parameters
        ----------
        hc_dim : int
            The dimension in which this hypercube exists. The number of
            vertices will be 2**hc_dim.

        Returns
        -------
        edges : tuple[tuple[int, int]]
        """
        edges = []
        for i in range(2**hc_dim):
            for j in range(hc_dim):
                bitmask = (1 << j)
                if 0 == (i & bitmask):
                    edges.append((i, i | bitmask))
        return tuple(edges)
    @classmethod
    def get_optimal_lapl_based_comm_W(cls, hc_dim, dtype=None, dev=None):
        """
        Returns the optimal laplacian-based communication matrix W

        Parameters
        ----------
        hc_dim : int
            The dimension in which this hypercube exists. The number of
            vertices will be 2**hc_dim.

        Returns
        -------
        W : tensor[2**hc_dim, 2**hc_dim]
            The optimal laplacian-based communication matrix W. W is such that
            W 1 = 1 and both of the following two quantities are minimized over
            all choice of laplacian-based communication matrices:
                * The operator norm of W - (1 1^T / M)
                * The Frobenius norm of W - (1 1^T / M)
        """
        A = adjacency_matrix_from_edges(
            cls._get_edges(hc_dim),
            dtype=dtype,
            dev=dev,
        )
        r = (hc_dim+1) / hc_dim
        d = A.sum(-1)
        d_max = d.max()
        D = d.diag_embed()
        L = D - A
        I = torch.eye(A.shape[-1], dtype=dtype, device=dev)
        W = I - ((1/(r*d_max)) * L)
        return W

