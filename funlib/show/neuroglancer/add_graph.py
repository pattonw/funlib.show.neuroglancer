import itertools
import neuroglancer
import random
import networkx as nx

import numpy as np

from pathlib import Path


class SkeletonSource(neuroglancer.skeleton.SkeletonSource):
    def __init__(
        self,
        dimensions: neuroglancer.CoordinateSpace,
        graph: nx.Graph,
        skeleton_attribute: str = "skeleton_id",
    ):
        super().__init__(dimensions)
        self.graph = graph
        self.vertex_attributes["affinity"] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32,
            num_components=1,
        )
        skeleton_ids = {data[skeleton_attribute] for _, data in graph.nodes(data=True)}
        self.skeletons = {
            sid: self.graph.subgraph(
                [n for n, d in graph.nodes(data=True) if d[skeleton_attribute] == sid]
            )
            for sid in skeleton_ids
        }

    def get_skeleton(self, i):
        skeleton = self.skeletons[i]
        vertex_positions = [pos for pos in skeleton.nodes(data="position")]
        edges = [[u, v] for u, v in skeleton.edges()]
        return neuroglancer.skeleton.Skeleton(
            vertex_positions=vertex_positions,
            edges=edges,
        )


def skeleton_id_to_color(skeleton_id):
    random.seed(hash(skeleton_id))

    r = random.randint(128, 255)
    g = random.randint(128, 255)
    b = random.randint(128, 255)

    hex_color = f"#{r:02x}{g:02x}{b:02x}"

    return hex_color


def add_graph(
    context, nx_graph: nx.Graph, name: str
):
    ndims = len(nx_graph.nodes[next(iter(nx_graph.nodes))]["position"])

    axis_names = nx_graph.graph["axis_names"]
    scales = nx_graph.graph.get("voxel_size")
    units = nx_graph.graph.get("units")
    if axis_names is None:
        axis_names = [f"d{i}" for i in range(ndims)]
    if scales is None:
        scales = [1] * ndims
    if units is None:
        units = [""] * ndims

    dimensions = neuroglancer.CoordinateSpace(
        names=axis_names, units=units, scales=scales
    )
    skeleton_id = 0
    edges = []
    ngid = itertools.count()

    for u, v in nx_graph.edges():
        pos_u = nx_graph.nodes[u]["position"]
        pos_v = nx_graph.nodes[v]["position"]

        edges.append(
            neuroglancer.LineAnnotation(point_a=pos_u, point_b=pos_v, id=next(ngid))
        )

    context.layers[name] = neuroglancer.LocalAnnotationLayer(
        dimensions=dimensions,
        annotation_color=skeleton_id_to_color(skeleton_id),
        annotations=edges,
    )
