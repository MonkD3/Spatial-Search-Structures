from . import lib
import ctypes
import numpy as np
import os
from time import perf_counter


print_perf = os.getenv("SHOW_TIME") is not None

lib.WNOcTree_create.restype = ctypes.c_void_p
lib.WNOcTree_create.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags=("C", "ALIGNED")),
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=2, flags=("C", "ALIGNED")),
    ctypes.c_uint32,
    ctypes.c_uint32
]

lib.WNOcTree_query.restype = None
lib.WNOcTree_query.argtypes = [
    ctypes.c_void_p,
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags=("C", "ALIGNED")),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags=("C", "ALIGNED", "WRITEABLE")),
    ctypes.c_uint32
]

lib.WNOcTree_destroy.restype = None
lib.WNOcTree_destroy.argtypes = [ctypes.c_void_p]


class WNOcTree:
    """
    An Octree to perform 3D Winding Number computations.

    Attributes
    ----------
    tree : libquadtree.WNOcTree
        A pointer to the Octree constructed by the C++ library.

    Methods
    -------
    query(q) :
        Perform a batch winding number queries for the points q
    """

    def __init__(self, nodes, tri):
        """
        Initialize the surface of the domain.

        @param nodes: the nodes of the boundary
        @param tri: triplet of indices of the nodes forming the triangles of the boundary
        Initialize the boundary of the domain.

        In 3D the boundary of the domain is a set of M trianges. These triangles 
        are triplets over a set of N nodes. 

        The boundary does not need to form a closed surface. However if the 
        surface is open the queries will be approximate.

        Parameters
        ----------
        nodes : numpy.ndarray of shape (N, 3)
            The nodes contained in the boundary.
        tri : numpy.ndarray of shape (M, 3)
            triplets of indices of the nodes forming the triangles in the boundary.
            The triangles normal must be outward to the interior.
        """

        nodes = nodes.reshape((-1, 3)).astype(np.float64)
        tri = tri.reshape((-1, 3)).astype(np.uint32)

        t0 = perf_counter()
        self.tree = lib.WNOcTree_create(
            nodes,
            tri,
            nodes.shape[0],
            tri.shape[0]
        )
        if print_perf:
            print(f"Time taken to initialize and construct tree (#nodes={nodes.shape[0]}, #tri={tri.shape[0]}) : {(perf_counter()-t0)*1000:.5f} ms")

        return

    def __del__(self):
        """ Release C associated memory """
        lib.WNOcTree_destroy(self.tree)
        return

    def query(self, q):
        """Query points @q for winding number

        Parameters
        ----------
        q : numpy.ndarray of shape (Q, 3)
            The query points for the winding number

        Returns
        -------
        wn : numpy.ndarray of shape (Q,)
            The winding number result for each query point
        """

        q = q.reshape((-1, 3)).astype(np.float64)
        wn = np.zeros(q.shape[0], dtype=np.float64)
        nqueries = wn.shape[0]

        t0 = perf_counter()
        lib.WNOcTree_query(
            self.tree,
            q,
            wn,
            nqueries
        )
        if print_perf:
            print(f"Time taken to make {nqueries} queries : {(perf_counter()-t0)*1000:.5f} ms")

        return wn
