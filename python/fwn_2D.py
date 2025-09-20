from . import lib
import ctypes 
import numpy as np 
import os 
from time import perf_counter
import pathlib

print_perf = os.getenv("SHOW_TIME") is not None

lib.WNQuadTree_create.restype = ctypes.c_void_p
lib.WNQuadTree_create.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags=("C", "ALIGNED")),
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=2, flags=("C", "ALIGNED")),
    ctypes.c_uint32,
    ctypes.c_uint32
]

lib.WNQuadTree_query.restype = None
lib.WNQuadTree_query.argtypes = [
    ctypes.c_void_p,
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags=("C", "ALIGNED")),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags=("C", "ALIGNED", "WRITEABLE")),
    ctypes.c_uint32
]

lib.WNQuadTree_destroy.restype = None
lib.WNQuadTree_destroy.argtypes = [ctypes.c_void_p]


class WNQuadTree:
    """
    A quadtree to perform 2D Winding Number computations.

    Attributes
    ----------
    tree : libquadtree.WNQuadTree
        A pointer to the quadtree constructed by the C++ library.

    Methods
    -------
    query(q) :
        Perform a batch winding number queries for the points q
    """

    def __init__(self, nodes, edges):
        """
        Initialize the boundary of the domain.

        In 2D the boundary of the domain is a set of M edges. These edges 
        are segments over a set of N nodes. 

        The boundary does not need to form a closed curve. However if the 
        curve is open the queries will be approximate.

        Parameters
        ----------
        nodes : numpy.ndarray of shape (N, 2)
            The nodes contained in the boundary.
        edges : numpy.ndarray of shape (M, 2)
            pairs of indices of the nodes forming the edges in the boundary.
            The curve(s) must be oriented in couter-clockwise direction.
        """

        nodes = nodes.reshape((-1, 2)).astype(np.float64)
        edges = edges.reshape((-1, 2)).astype(np.uint32)

        t0 = perf_counter()
        self.tree = lib.WNQuadTree_create(
            nodes,
            edges,
            nodes.shape[0],
            edges.shape[0]
        )
        if print_perf:
            print(f"Time taken to initialize and construct tree (#nodes={nodes.shape[0]}, #edges={edges.shape[0]}) : {(perf_counter()-t0)*1000:.5f} ms")

        return

    def __del__(self):
        """Release associated C memory"""
        lib.WNQuadTree_destroy(self.tree)
        return

    def query(self, q):
        """Query points @q for winding number

        Parameters
        ----------
        q : numpy.ndarray of shape (Q, 2)
            The query points for the winding number

        Returns
        -------
        wn : numpy.ndarray of shape (Q,)
            The winding number result for each query point
        """

        q = q.reshape((-1, 2)).astype(np.float64)
        wn = np.zeros(q.shape[0], dtype=np.float64)
        nqueries = wn.shape[0]

        t0 = perf_counter()
        lib.WNQuadTree_query(
            self.tree,
            q,
            wn,
            nqueries
        )
        if print_perf:
            print(f"Time taken to make {nqueries} queries : {(perf_counter()-t0)*1000:.5f} ms")

        return wn
