from search_structures.surface_distance_2D import SurfaceTree2D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc

plot = True
n = 100
ntest = 100

t = np.linspace(0, 2*np.pi, n, endpoint=False)
coords = np.zeros((n, 2))
coords[:, 0] = np.cos(t)
coords[:, 1] = np.sin(t)

edges = np.zeros((n, 2), dtype=np.uint32)
for i in range(n):
    edges[i, 0] = i
    edges[i, 1] = (i+1) % n


tree = SurfaceTree2D(coords, edges)

query = np.zeros((ntest*ntest, 2))
for i in range(ntest):
    for j in range(ntest):
        query[i*ntest + j, 0] = -1.5 + i*3.0/(ntest-1)
        query[i*ntest + j, 1] = -1.5 + j*3.0/(ntest-1)

dist = tree.query(query)

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
    scatter = ax.scatter(query[:, 0], query[:, 1], c=dist, vmin=-0, vmax=1)
    cs = ax.contour(
        query[:, 0].reshape(ntest, ntest),
        query[:, 1].reshape(ntest, ntest),
        dist.reshape(ntest, ntest),
        levels=15,
        colors="k",
    )
    ax.clabel(cs, fontsize=10)

    fig.colorbar(scatter, ax=ax, orientation='vertical')
    plt.show()
