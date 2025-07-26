import gmsh 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.collections as mc


gmsh.initialize()
gmsh.open("./square_with_hole.geo")
gmsh.model.mesh.generate(1)

nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes(dim=1, includeBoundary=True)
nodeCoords = nodeCoords.reshape((-1, 3))[:, :2]
tag2idx = {
    nodeTags[i]: i
    for i in range(len(nodeTags))
}

_, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim=1)

edgesTags = elemNodeTags[0].reshape((-1, 2))
edges = [
    [nodeCoords[tag2idx[s]], nodeCoords[tag2idx[t]]]
    for (s, t) in edgesTags
]
lines = mc.LineCollection(edges)

fig, ax = plt.subplots()
ax.add_collection(lines)
plt.show()

gmsh.finalize()
