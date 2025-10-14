from search_structures.nearest_neighbor import NNTree2D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc

plot = True
n = 1000

coords = np.random.uniform(0, 1, size=(n, 2))

tree = NNTree2D(coords)

max_dist = np.ones(len(coords)) * np.inf
nn_idx = tree.query(max_dist)

if plot:
    lines = np.zeros((n, 4))
    lines[:, 0] = coords[:, 0]
    lines[:, 1] = coords[:, 1]
    lines[:, 2] = coords[nn_idx[:], 0]
    lines[:, 3] = coords[nn_idx[:], 1]
    surface = lines.reshape((-1, 2, 2))

    lc = mc.LineCollection(surface, linewidth=1, color="black")

    fig, ax = plt.subplots()

    ax.add_collection(lc)
    ax.set_aspect("equal")
    scatter = ax.scatter(coords[:, 0], coords[:, 1])
    plt.show()
