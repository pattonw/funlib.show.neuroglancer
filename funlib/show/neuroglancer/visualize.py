#!/usr/bin/env python

from funlib.persistence import Array
import neuroglancer
import os
import webbrowser

from .add_array import add_layer


def visualize(
    arrays: dict[str, Array],
    bind_address: str = "0.0.0.0",
    port: int = 0,
    browser: bool = True,
):
    neuroglancer.set_server_bind_address(bind_address, bind_port=port)
    viewer = neuroglancer.Viewer()

    with viewer.txn() as s:
        for name, array in arrays.items():
            add_layer(s, array, name)

    if all((array.dims - array.channel_dims) <= 2 for array in arrays.values()):
        with viewer.txn() as s:
            # set view to xy plane for 2D arrays
            s.layout = "xy"

    url = str(viewer)
    print(url)
    if os.environ.get("DISPLAY") and browser:
        webbrowser.open_new(url)

    print("Press ENTER to quit")
    input()
