import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import sys

surface = np.loadtxt(sys.argv[1], delimiter=",")
query = np.loadtxt(sys.argv[2], delimiter=",")
wn = np.loadtxt(sys.argv[3], delimiter=",")

surface = surface.reshape((-1, 2, 2))
lc = mc.LineCollection(surface)

fig, ax = plt.subplots()

ax.add_collection(lc)
scatter = ax.scatter(query[:, 0], query[:, 1], c=wn, vmin=-1, vmax=2)

fig.colorbar(scatter, ax=ax, orientation='vertical')
plt.show()
