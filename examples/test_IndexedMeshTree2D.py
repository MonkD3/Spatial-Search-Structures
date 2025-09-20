from search_structures.indexed_mesh_node_loc_2D import IndexedMeshTree2D
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

plot = False
n = 500
ntest = 1000000

x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
X, Y = np.meshgrid(x, y)

coords = np.zeros((n*n, 2))
coords[:, 0] = X.flatten()
coords[:, 1] = Y.flatten()

triangulation = Delaunay(coords)

tree = IndexedMeshTree2D(coords, triangulation.simplices)

query = np.random.uniform(-0.5, 1.5, size=(ntest, 2))

tri_id = tree.query(query)

if plot:

    tri_col = np.zeros(triangulation.simplices.shape[0])
    for i, val in enumerate(tri_id):
        if val >= 0:
            tri_col[val] = 1

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.tripcolor(coords[:, 0], coords[:, 1], tri_col, triangles=triangulation.simplices)
    ax.triplot(coords[:, 0], coords[:, 1], triangulation.simplices)
    ax.scatter(query[:, 0], query[:, 1], marker="x", color="red")

    plt.show()
