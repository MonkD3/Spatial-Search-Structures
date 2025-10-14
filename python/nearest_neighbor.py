from . import lib
import ctypes
import numpy as np
import os
from time import perf_counter

print_perf = os.getenv("SHOW_TIME") is not None

lib.NNTree2D_create.restype = ctypes.c_void_p
lib.NNTree2D_create.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags=("C", "ALIGNED")),
    ctypes.c_uint32
]

lib.NNTree2D_query.restype = None
lib.NNTree2D_query.argtypes = [
    ctypes.c_void_p,
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags=("C", "ALIGNED")),
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags=("C", "ALIGNED", "WRITEABLE"))
]

lib.NNTree2D_destroy.restype = None
lib.NNTree2D_destroy.argtypes = [ctypes.c_void_p]


class NNTree2D:
    """
    A quadtree to perform point-point nearest neighbor computations.

    Attributes
    ----------
    tree : libquadtree.NNTree2D
        A pointer to the quadtree constructed by the C++ library.

    Methods
    -------
    query(q) :
        Perform the nearest neighbor queries for every nodes
    """

    def __init__(self, nodes):
        """
        Initialize the spatial search structure

        Parameters
        ----------
        nodes : numpy.ndarray of shape (N, 2)
            The nodes contained in the boundary.
        """

        self.nodes = nodes.reshape((-1, 2)).astype(np.float64)

        t0 = perf_counter()
        self.tree = lib.NNTree2D_create(
            nodes,
            nodes.shape[0]
        )
        if print_perf:
            print(f"Time taken to initialize and construct tree (#nodes={nodes.shape[0]}) : {(perf_counter()-t0)*1000:.5f} ms")

        return

    def __del__(self):
        """ Release C associated memory """
        lib.NNTree2D_destroy(self.tree)
        return

    def query(self, s):
        """Maximum search distance @s for nearest neighbor

        Parameters
        ----------
        s : numpy.ndarray of shape (N,)
            The maximum search distance

        Returns
        -------
        idx : numpy.ndarray of shape (Q,)
            The index of the closest node
        """

        s = s.astype(np.float64)
        idx = np.arange(0, len(self.nodes), 1, dtype=np.uint32)

        t0 = perf_counter()
        lib.NNTree2D_query(
            self.tree,
            s,
            idx
        )
        if print_perf:
            print(f"Time taken to make {len(self.nodes)} queries : {(perf_counter()-t0)*1000:.5f} ms")

        return idx
