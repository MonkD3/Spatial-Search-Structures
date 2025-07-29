import ctypes 
import numpy as np 
import os 
from time import perf_counter
import pathlib

print_perf = os.getenv("SHOW_TIME") is not None

lib_path = pathlib.Path().absolute() / "build/src/libquadtree.so"
lib = ctypes.CDLL(lib_path)


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
    class WNQuadTree
    """

    def __init__(self, nodes, edges):
        """
        Initialize the surface of the domain.

        @param nodes: the nodes of the boundary
        @param edges: pairs of indices of the nodes forming the edges of the boundary
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
            print(f"Time taken to initialize and construct tree : {(perf_counter()-t0)*1000:.5f} ms")

        return

    def __del__(self):
        """ Release C associated memory """
        lib.WNQuadTree_destroy(self.tree)
        return

    def query(self, q):
        """Query points @q for winding number

        @param q : the coordinates of the query points
        @return wn: the winding number for each corresponding point.
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


# ========================== Winding number 3D ===============================

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
    class WNOcTree
    """

    def __init__(self, nodes, tri):
        """
        Initialize the surface of the domain.

        @param nodes: the nodes of the boundary
        @param tri: triplet of indices of the nodes forming the triangles of the boundary
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
            print(f"Time taken to initialize and construct tree : {(perf_counter()-t0)*1000:.5f} ms")

        return

    def __del__(self):
        """ Release C associated memory """
        lib.WNOcTree_destroy(self.tree)
        return

    def query(self, q):
        """Query points @q for winding number

        @param q : the coordinates of the query points
        @return wn: the winding number for each corresponding point.
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
