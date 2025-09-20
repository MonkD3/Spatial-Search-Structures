from search_structures.fwn_3D import WNOcTree
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import igl
import gmsh

plot = False

gmsh.initialize()
gmsh.option.setNumber("Mesh.MeshSizeFactor", 0.2)

n = 20
s = 0.2    # shift parameter

# Generate two circles 
left_sphere = gmsh.model.occ.addSphere(-0.5+s, 0.0, 0.0, 0.5)
right_sphere = gmsh.model.occ.addSphere(0.5-s, 0.0, 0.0, 0.5)
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)

nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes(dim=2, includeBoundary=True)
_, elemTags, elemNodes = gmsh.model.mesh.getElements(dim=2)
gmsh.finalize()
g2i = dict(zip(nodeTags,np.arange(0, nodeTags.shape[0])))

nodeCoords = nodeCoords.reshape((-1, 3))
elemNodesIGL = elemNodes[0]
elemNodesIGL = np.array([g2i[i] for i in elemNodesIGL]).reshape((-1, 3))

query = np.zeros((n*n*n, 3))
for i in range(n):
    for j in range(n):
        for k in range(n):
            query[n*(n*i+j) + k, 0] = -1 + 2.0/n * i
            query[n*(n*i+j) + k, 1] = -1 + 2.0/n * j
            query[n*(n*i+j) + k, 2] = -1 + 2.0/n * k

# query = np.zeros((n*n, 3))
# for i in range(n):
#     for j in range(n):
#         query[(n*i+j), 0] = -1 + 2.0/n * i
#         query[(n*i+j), 1] = -1 + 2.0/n * j

# IGL winding number
t0 = perf_counter()
wn_IGL = igl.fast_winding_number(nodeCoords, elemNodesIGL, query) 
print(f"Time for IGL : {perf_counter() - t0}s")

# My winding number
t0 = perf_counter()
tree = WNOcTree(nodeCoords, elemNodesIGL)
wn_mine = tree.query(query) 
print(f"Time for mine : {perf_counter() - t0}s")

mean_error = np.mean((wn_IGL-wn_mine)**2)
max_error = np.max(abs(wn_IGL - wn_mine))
print(f"Mean squared error between IGL and me : {mean_error}\nMax error : {max_error}")

if plot:
    verts = np.array([
        [nodeCoords[i] for i in elem] 
        for elem in elemNodesIGL
    ])

    # Cannot reuse collection in another axis, copies the collection for each axis
    pc = [
        Poly3DCollection(verts, shade=False, facecolors = None),
        Poly3DCollection(verts, shade=False, facecolors = None)
    ]

    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, subplot_kw={"projection": "3d"})

    s1 = ax[0].scatter(query[:, 0], query[:, 1], query[:, 2], c=wn_IGL)
    ax[0].add_collection(pc[0])
    ax[0].set_aspect("equal")

    s2 = ax[1].scatter(query[:, 0], query[:, 1], query[:, 2], c=wn_mine, vmin=0, vmax=2)
    ax[1].add_collection(pc[1])
    ax[1].set_aspect("equal")

    plt.colorbar(s1)
    plt.colorbar(s2)

    plt.show()
