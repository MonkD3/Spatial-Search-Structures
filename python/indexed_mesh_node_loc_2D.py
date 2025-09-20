from . import lib
import ctypes
import numpy as np
import os
from time import perf_counter

print_perf = os.getenv("SHOW_TIME") is not None

lib.IndexedMeshTree2D_create.restype = ctypes.c_void_p
lib.IndexedMeshTree2D_create.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags=("C", "ALIGNED")),
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=2, flags=("C", "ALIGNED")),
    ctypes.c_uint32,
    ctypes.c_uint32
]

lib.IndexedMeshTree2D_query.restype = None
lib.IndexedMeshTree2D_query.argtypes = [
    ctypes.c_void_p,
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags=("C", "ALIGNED")),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags=("C", "ALIGNED", "WRITEABLE")),
    ctypes.c_uint32
]

lib.IndexedMeshTree2D_destroy.restype = None
lib.IndexedMeshTree2D_destroy.argtypes = [ctypes.c_void_p]


class IndexedMeshTree2D:
    """
    A quadtree to perform 2D point-triangle collisions computations.

    Attributes
    ----------
    tree : libquadtree.IndexedMeshTree2D
        A pointer to the quadtree constructed by the C++ library.

    Methods
    -------
    query(q) :
        Perform a batch queries for the points q
    """

    def __init__(self, nodes, tri):
        """
        Initialize the quadtree for a triangle mesh

        Parameters
        ----------
        nodes : numpy.ndarray of shape (N, 2)
            The nodes contained in the boundary.
        tri : numpy.ndarray of shape (M, 3)
            triplets of indices of the nodes forming the triangles in the boundary.
        """

        nodes = nodes.reshape((-1, 2)).astype(np.float64)
        tri = tri.reshape((-1, 3)).astype(np.uint32)

        t0 = perf_counter()
        self.tree = lib.IndexedMeshTree2D_create(
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
        lib.IndexedMeshTree2D_destroy(self.tree)
        return

    def query(self, q):
        """Query points @q for triangle collisions

        Parameters
        ----------
        q : numpy.ndarray of shape (Q, 2)
            The query points 

        Returns
        -------
        tri_id : numpy.ndarray of shape (Q,)
            The resulting index of the triangle for each point, -1 if no collision
        """

        q = q.reshape((-1, 2)).astype(np.float64)
        tri_id = np.zeros(q.shape[0], dtype=np.int32)
        nqueries = tri_id.shape[0]

        t0 = perf_counter()
        lib.IndexedMeshTree2D_query(
            self.tree,
            q,
            tri_id,
            nqueries
        )
        if print_perf:
            print(f"Time taken to make {nqueries} queries : {(perf_counter()-t0)*1000:.5f} ms")

        return tri_id
