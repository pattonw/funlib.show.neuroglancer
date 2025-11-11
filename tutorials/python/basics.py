# %% [markdown]
# # Visualizing volumes from a python script
#
# This example shows how to visualize volumes from a python script using the `funlib.show.neuroglancer` package.

# %% [markdown]
# ## Setup
# `funlib.show.neuroglancer` interfaces best with `funlib.persistence` since it has robust support for
# metadata defining axis names, units, types, and more. We will be using standard dataset examples
# from scipy and numpy but you can use any dataset.

# %%
from funlib.show.neuroglancer import visualize
from funlib.persistence import Array
import numpy as np
from skimage import data
from skimage.measure import label
import networkx as nx

# %%
# First a simple 2D gray scale array
array = Array(
    data=data.camera(),
    axis_names=["y", "x"],
    voxel_size=(1, 1),
    offset=(0, 0),
)
graph = nx.Graph()
graph.add_nodes_from(
    [
        (0, {"position": (10, 20), "skeleton_id": 1}),
        (1, {"position": (50, 80), "skeleton_id": 1}),
        (2, {"position": (80, 120), "skeleton_id": 2}),
        (3, {"position": (120, 150), "skeleton_id": 2}),
    ]
)
graph.add_edges_from([(0, 1), (2, 3)])
graph.graph["axis_names"] = ["y", "x"]
graph.graph["voxel_size"] = [1, 1]
graph.graph["offset"] = [0, 0]

visualize({"camera": array}, graphs={"test": graph}, bind_address="localhost")
# problems:
# 1. neuroglancer opens in 3D viewer but the array is 2D
# 2. the array is unexpectedly transposed

# %%
# A 2D color array
array = Array(
    data=data.astronaut(),
    axis_names=["y", "x", "c"],
    types=["space", "space", "color"],
    voxel_size=(1, 1),
    offset=(0, 0),
)

visualize({"face": array}, bind_address="localhost")

# %%
# 3D 2 color array
array = Array(
    data=data.cells3d(),
    axis_names=["z", "c", "y", "x"],
    types=["space", "color", "space", "space"],
    voxel_size=(1, 1, 1),
    offset=(0, 0, 0),
)

visualize({"cells": array}, bind_address="localhost")

# %%
# binary blobs 3D
array = Array(
    data=data.binary_blobs(length=128, blob_size_fraction=0.1, n_dim=3),
    axis_names=["z", "y", "x"],
    types=["space", "space", "space"],
    voxel_size=(1, 1, 1),
    offset=(0, 0, 0),
)

visualize({"blobs": array}, bind_address="localhost")

# %%
# binary blobs 4D plus color
array = Array(
    data=np.array(
        [
            data.binary_blobs(length=64, blob_size_fraction=0.1, n_dim=4)
            for _ in range(3)
        ]
    ),
    axis_names=["c", "t", "z", "y", "x"],
    types=["color", "time", "space", "space", "space"],
    voxel_size=(1, 1, 1, 1),
    offset=(0, 0, 0, 0),
)

visualize({"blobs": array}, bind_address="localhost")

# %%
# segmentation layers uint32, uint64, int32, int64
arrays = {
    f"{dtype}": Array(
        data=label(
            data.binary_blobs(length=64, blob_size_fraction=0.1, n_dim=2)
        ).astype(dtype=dtype),
        axis_names=["y", "x"],
        types=["space", "space"],
        voxel_size=(1, 1),
        offset=(0, 0),
    )
    for dtype in [
        np.uint32,
        np.uint64,
        np.int32,
        np.int64,
    ]
}
visualize(arrays, bind_address="localhost")

# %%
# fractional voxel offset
data = data.camera()
data_downsampled = data[::4, ::4]

offset = (2, 2)

camera_array = Array(
    data=data,
    axis_names=["y", "x"],
    voxel_size=(1, 1),
    offset=offset,
)
downsampled_camera_array = Array(
    data=data_downsampled,
    axis_names=["y", "x"],
    voxel_size=(4, 4),
    offset=offset,
)
visualize(
    arrays={"camera": camera_array, "downsampled_camera": downsampled_camera_array},
    bind_address="localhost",
)
