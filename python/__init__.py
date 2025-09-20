import numpy as np
from time import perf_counter
import pathlib

search_paths = [
    "@CMAKE_LIBRARY_OUTPUT_DIRECTORY@",
    "@LIB_INSTALL_DIR@",
    pathlib.Path(__file__).parent.parent / "lib"
]

libdir_path = ""
for path in search_paths:
    if pathlib.Path(path).exists():
        libdir_path = path 
        break

lib = np.ctypeslib.load_library("libquadtree", libdir_path)
