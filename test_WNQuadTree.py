from spatial_structures import WNQuadTree
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc

plot = False
n = 1000000
ntest = 2000

t = np.linspace(0, 2*np.pi, n, endpoint=False)
coords = np.zeros((n, 2))
coords[:, 0] = np.cos(t)
coords[:, 1] = np.sin(t)

edges = np.zeros((n, 2), dtype=np.uint32)
for i in range(n):
    edges[i, 0] = i
    edges[i, 1] = (i+1) % n


tree = WNQuadTree(coords, edges)

query = np.zeros((ntest*ntest, 2))
for i in range(ntest):
    for j in range(ntest):
        query[i*ntest + j, 0] = -1.5 + i*3.0/(ntest-1)
        query[i*ntest + j, 1] = -1.5 + j*3.0/(ntest-1)

wn = tree.query(query)

if plot:
    lines = np.zeros((n, 4))
    lines[:, 0] = coords[edges[:, 0], 0]
    lines[:, 1] = coords[edges[:, 0], 1]
    lines[:, 2] = coords[edges[:, 1], 0]
    lines[:, 3] = coords[edges[:, 1], 1]
    surface = lines.reshape((-1, 2, 2))

    lc = mc.LineCollection(surface, linewidth=5, color="white")

    fig, ax = plt.subplots()

    ax.add_collection(lc)
    ax.set_aspect("equal")
    scatter = ax.scatter(query[:, 0], query[:, 1], c=wn>=0.5, vmin=-1, vmax=1)

    fig.colorbar(scatter, ax=ax, orientation='vertical')
    plt.show()
