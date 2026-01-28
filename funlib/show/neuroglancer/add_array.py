from .scale_pyramid import ScalePyramid
import neuroglancer
from funlib.persistence import Array

import numpy as np
from dataclasses import dataclass
from collections.abc import Sequence


@dataclass
class ShaderMetadata:
    dtype: np.dtype
    channel_dims: Sequence[int]


color = """
void main() {
    emitRGB(
        vec3(
            toNormalized(getDataValue(0)),
            toNormalized(getDataValue(1)),
            toNormalized(getDataValue(2)))
        );
}"""

color_shader_code = """
void main() {
    emitRGBA(
        vec4(
        %f, %f, %f,
        toNormalized(getDataValue()))
        );
}"""

binary_shader_code = """
void main() {
  emitGrayscale(255.0*toNormalized(getDataValue()));
}"""

heatmap_shader_code = """
void main() {
    float v = toNormalized(getDataValue(0));
    vec4 rgba = vec4(0,0,0,0);
    if (v != 0.0) {
        rgba = vec4(colormapJet(v), 1.0);
    }
    emitRGBA(rgba);
}"""


def create_coordinate_space(
    array: Array,
) -> tuple[neuroglancer.CoordinateSpace, list[int]]:
    assert array.spatial_dims > 0

    def interleave(list, fill_value, axis_types):
        """
        Fill in values for non spatial axes.
        """
        return_list = [fill_value] * len(axis_types)
        for i, axis_type in enumerate(axis_types):
            if axis_type in ["time", "space"]:
                return_list[i] = list.pop(0)
        return return_list

    units = interleave(list(array.units), "", array.types)
    scales = interleave(list(array.voxel_size), 1, array.types)
    offset = interleave(
        [o / v for o, v in zip(array.offset, array.voxel_size)], 0, array.types
    )
    axis_names = [
        axis_name
        if axis_type in ["time", "space"] or axis_name.endswith("^")
        else f"{axis_name}^"
        for axis_name, axis_type in zip(array.axis_names, array.types)
    ]

    affine_transform = np.eye(len(scales) + 1)[:-1, :].tolist()
    for i, off in enumerate(offset):
        affine_transform[i][-1] = off

    return (
        neuroglancer.CoordinateSpace(names=axis_names, units=units, scales=scales),
        affine_transform,
    )


def guess_shader_code(array: Array):
    """
    TODO: This function is not used yet.
    It should make some reasonable guesses for basic visualization parameters.
    Guess volume type (or read from optional metadata?):
        - bool/uint32/uint64/int32/int64 -> Segmentation
        - floats/int8/uint8 -> Image
    Guess shader for Image volumes:
        - 1 channel dimension:
            - 1 channel -> grayscale (add shader options for color and threshold)
            - 2 channels -> projected RGB (set B to 0 or 1 or R+G?)
            - 3 channels -> RGB
            - 4 channels -> projected RGB (PCA? Random linear combinations? Randomizable with "l" key?)
        - multiple channel dimensions?:
    """
    raise NotImplementedError()
    channel_dim_shapes = [
        array.shape[i]
        for i in range(len(array.axis_names))
        if "^" in array.axis_names[i]
    ]
    if len(channel_dim_shapes) == 0:
        return None  # default shader

    if len(channel_dim_shapes) == 1:
        num_channels = channel_dim_shapes[0]
        if num_channels == 1:
            return None  # default shader
        if num_channels == 2:
            return projected_rgb_shader_code % num_channels
        if num_channels == 3:
            return rgb_shader_code % (0, 1, 2)
        if num_channels > 3:
            return projected_rgb_shader_code % num_channels


def create_shader_code(
    shader, channel_dims, rgb_channels=None, color=None, scale_factor=1.0
):
    if shader is None:
        if channel_dims > 1:
            shader = "rgb"
        else:
            return None

    if rgb_channels is None:
        rgb_channels = [0, 1, 2]

    if shader == "rgb":
        return rgb_shader_code % (
            scale_factor,
            rgb_channels[0],
            rgb_channels[1],
            rgb_channels[2],
        )

    if shader == "color":
        assert color is not None, (
            "You have to pass argument 'color' to use the color shader"
        )
        return color_shader_code % (
            color[0],
            color[1],
            color[2],
        )

    if shader == "binary":
        return binary_shader_code

    if shader == "heatmap":
        return heatmap_shader_code


def make_neuroglancer_compatible(array: Array):
    """
    Make the `funlib.persistence.Array` compatible with neuroglancer.
    """

    if array.dtype == bool:
        # neuroglancer does not support bool arrays, convert to float32
        # this is not efficient, but visualizes as black/white as we would expect a mask to do
        array.lazy_op(lambda data: data.astype(np.float32))
    if array.dtype == np.int64:
        # neuroglancer does not support int64 arrays, convert to uint64
        array.lazy_op(lambda data: data.astype(np.uint64))


def add_layer(
    context,
    array: Array | list[Array],
    name: str,
    shader: str | None = None,
    channel_dim: int | None = None,
):
    """Add a layer to a neuroglancer context.

    Args:

        context:

            The neuroglancer context to add a layer to, as obtained by
            ``viewer.txn()``.

        array:

            A ``funlib.persistence.Array``

        name:

            The name of the layer.

        color:

            A list of floats representing the RGB values for the constant color
            shader.

        visible:

            A bool which defines the initial layer visibility.

        value_scale_factor:

            A float to scale array values with for visualization.
    """
    # shader metadata:
    dtype = array.dtype
    if channel_dim is None:
        assert array.channel_dims <= 1, (
            "Cannot handle multiple channel dimensions. "
            "Please specify a channel dimension"
        )
        if array.channel_dims == 1:
            channel_dim = [
                i for i, t in enumerate(array.types) if t not in ["time", "space"]
            ][0]
    num_channels = array.shape[channel_dim] if channel_dim is not None else None

    # guess shader based on dtype/num_channels
    if dtype in [bool, np.bool] and (num_channels is None or num_channels == 1):
        # mask array. Will be converted to float
        volume_type = "image"
        shader = None  # Gray is default
    elif dtype in [bool, np.bool] and num_channels > 1:
        # color mask array. Will be converted to float
        volume_type = "image"
        shader = color
    elif dtype in [np.uint8, np.uint16, np.float16, np.float32, np.float64, float] and (
        num_channels is None or num_channels == 1
    ):
        # grey scale integer embedding
        volume_type = "image"
        shader = None  # Gray is default
    elif dtype in [np.uint8, np.uint16, np.float16, np.float32, np.float64, float] and (
        1 < num_channels <= 3
    ):
        # color images
        volume_type = "image"
        shader = color
    elif dtype in [np.uint8, np.uint16, np.float16, np.float32, np.float64, float] and (
        num_channels > 3
    ):
        # too many colors
        volume_type = "image"
        shader = color
    else:
        volume_type = "segmentation"
        shader = None

    # make the array compatible with neuroglancer
    make_neuroglancer_compatible(array)

    dimensions, affine_transform = create_coordinate_space(array)

    layer = neuroglancer.LocalVolume(
        data=array.data,
        dimensions=dimensions,
        volume_type=volume_type,
    )

    if volume_type == "segmentation":
        context.layers[name] = neuroglancer.SegmentationLayer(
            source=[
                neuroglancer.LayerDataSource(
                    layer,
                    transform=neuroglancer.CoordinateSpaceTransform(
                        output_dimensions=dimensions,
                        matrix=affine_transform,
                    ),
                )
            ],
        )
    else:
        context.layers[name] = neuroglancer.ImageLayer(
            source=[
                neuroglancer.LayerDataSource(
                    layer,
                    transform=neuroglancer.CoordinateSpaceTransform(
                        output_dimensions=dimensions,
                        matrix=affine_transform,
                    ),
                )
            ],
            **({"shader": shader} if shader is not None else {}),
        )
