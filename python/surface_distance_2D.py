from . import lib
import ctypes
import numpy as np
import os
from time import perf_counter

print_perf = os.getenv("SHOW_TIME") is not None

lib.SurfaceTree2D_create.restype = ctypes.c_void_p
lib.SurfaceTree2D_create.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags=("C", "ALIGNED")),
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=2, flags=("C", "ALIGNED")),
    ctypes.c_uint32,
    ctypes.c_uint32
]

lib.SurfaceTree2D_query.restype = None
lib.SurfaceTree2D_query.argtypes = [
    ctypes.c_void_p,
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags=("C", "ALIGNED")),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags=("C", "ALIGNED", "WRITEABLE")),
    ctypes.c_uint32
]

lib.SurfaceTree2D_destroy.restype = None
lib.SurfaceTree2D_destroy.argtypes = [ctypes.c_void_p]


class SurfaceTree2D:
    """
    A quadtree to perform 2D point-edge nearest neighbor computations.
    (a.k.a. distance to curves)

    Attributes
    ----------
    tree : libquadtree.WNQuadTree
        A pointer to the quadtree constructed by the C++ library.

    Methods
    -------
    query(q) :
        Perform a batch of distance queries for the points q
    """

    def __init__(self, nodes, edges):
        """
        Initialize the boundary of the domain.

        In 2D the boundary of the domain is a set of M edges. These edges 
        are segments over a set of N nodes. 

        The boundary does not need to form a closed curve. 

        Parameters
        ----------
        nodes : numpy.ndarray of shape (N, 2)
            The nodes contained in the boundary.
        edges : numpy.ndarray of shape (M, 2)
            pairs of indices of the nodes forming the edges in the boundary.
        """

        nodes = nodes.reshape((-1, 2)).astype(np.float64)
        edges = edges.reshape((-1, 2)).astype(np.uint32)

        t0 = perf_counter()
        self.tree = lib.SurfaceTree2D_create(
            nodes,
            edges,
            nodes.shape[0],
            edges.shape[0]
        )
        if print_perf:
            print(f"Time taken to initialize and construct tree (#nodes={nodes.shape[0]}, #edges={edges.shape[0]}) : {(perf_counter()-t0)*1000:.5f} ms")

        return

    def __del__(self):
        """ Release C associated memory """
        lib.SurfaceTree2D_destroy(self.tree)
        return

    def query(self, q):
        """Query points @q for distance to surface

        Parameters
        ----------
        q : numpy.ndarray of shape (Q, 2)
            The query points for the distance

        Returns
        -------
        wn : numpy.ndarray of shape (Q,)
            The resulting distance for each query point
        """

        q = q.reshape((-1, 2)).astype(np.float64)
        d = np.zeros(q.shape[0], dtype=np.float64)
        nqueries = d.shape[0]

        t0 = perf_counter()
        lib.SurfaceTree2D_query(
            self.tree,
            q,
            d,
            nqueries
        )
        if print_perf:
            print(f"Time taken to make {nqueries} queries : {(perf_counter()-t0)*1000:.5f} ms")

        return d
