#!/usr/bin/env python

import click
import glob
import os
import sys
import numpy as np
from pathlib import Path

from funlib.persistence import open_ds
# Assuming visualize is available in the same package
from funlib.show.neuroglancer import visualize

# Initialize S3 filesystem once
try:
    import s3fs
    s3 = s3fs.S3FileSystem()
except ImportError:
    s3 = None


def parse_slice(slice_str):
    """Safely parses a slice string into a numpy slice object."""
    if not slice_str:
        return None
    if not slice_str.startswith("[") and not slice_str.startswith("("):
        slice_str = f"[{slice_str}]"
    try:
        return eval(f"np.s_{slice_str}")
    except Exception as e:
        raise click.BadParameter(f"Invalid slice format '{slice_str}': {e}")

def expand_paths(path_str):
    """
    Handles wildcard expansion for both Local and S3 paths.
    Returns a list of valid path strings.
    """
    # 1. Handle S3 Paths
    if path_str.startswith("s3://"):
        assert s3 is not None, "s3fs failed to import"
        # If there is a wildcard, ask s3fs to glob it
        if glob.has_magic(path_str):
            # s3fs.glob returns paths like 'bucket/path/file', so we add s3:// back
            return [f"s3://{p}" for p in s3.glob(path_str)]
        return [path_str]
    
    # 2. Handle Local Paths
    else:
        expanded = glob.glob(path_str)
        if not expanded and not glob.has_magic(path_str):
            # If no magic chars, it might be a direct file that glob missed 
            # or simply doesn't exist yet (but we return it for the opener to try)
            return [path_str]
        return expanded

def find_pyramid_scales(ds_path):
    """
    Looks for 's0', 's1', etc. subdirectories.
    Returns sorted list of full paths or empty list.
    """
    # 1. Handle S3
    if ds_path.startswith("s3://"):
        # s3.glob returns 'bucket/key/s0', need to prepend s3://
        # We look for path/s*
        assert s3 is not None, "s3fs failed to import"
        scales = sorted(s3.glob(f"{ds_path}/s*"))
        return [f"s3://{s}" for s in scales]

    # 2. Handle Local
    else:
        return sorted(glob.glob(f"{ds_path}/s*"))

@click.command(context_settings=dict(ignore_unknown_options=True))
@click.option(
    "--dataset",
    "-d",
    multiple=True,
    help="Path to a dataset (Local or S3).",
)
@click.option(
    "--slice",
    "-s",
    "slices",
    multiple=True,
    help="Slice to apply.",
)
@click.option(
    "--no-browser",
    "-n",
    is_flag=True,
    help="If set, do not open a browser.",
)
@click.option(
    "--bind-address",
    "-b",
    default="0.0.0.0",
    help="Bind address.",
)
@click.option(
    "--port",
    type=int,
    default=0,
    help="The port to bind to.",
)
@click.argument("extra_datasets", nargs=-1)
def main(dataset, slices, no_browser, bind_address, port, extra_datasets):
    """
    Visualizes datasets in Neuroglancer (Local or S3).
    
    Example:
    neuroglancer -d s3://my-bucket/data.zarr/{raw,labels} -b localhost
    """
    
    all_dataset_inputs = list(dataset) + list(extra_datasets)

    if not all_dataset_inputs:
        click.echo("No datasets specified.")
        sys.exit(0)

    combined_inputs = []
    for i, ds_pattern in enumerate(all_dataset_inputs):
        s = slices[i] if i < len(slices) else None
        combined_inputs.append((ds_pattern, s))

    arrays_to_show = {}

    for path_input, slice_str in combined_inputs:
        
        current_slice = parse_slice(slice_str)
        
        # Expand wildcards (S3 aware)
        paths = expand_paths(path_input)

        for ds_path in paths:
            if not ds_path: 
                continue

            # Determine a display name (Last part of the path)
            layer_name = Path(ds_path).name

            # --- Attempt Open ---
            try:
                click.echo(f"Opening {ds_path}...")
                # Note: open_ds handles s3:// strings if s3fs is installed
                array_obj = open_ds(ds_path)
                array_pyramid = [array_obj]
            except Exception as e:
                # Fallback: Check for multi-resolution pyramid
                scales = find_pyramid_scales(ds_path)
                
                if not scales:
                    click.echo(f"Skipping {ds_path}: {e}")
                    continue
                
                click.echo(f"Found scales for {layer_name}: {[Path(s).name for s in scales]}")
                try:
                    array_pyramid = [open_ds(scale) for scale in scales]
                except Exception as pyramid_e:
                    click.echo(f"Failed to open scales for {ds_path}: {pyramid_e}")
                    continue

            # --- Apply Slice ---
            if current_slice is not None:
                click.echo(f"Applying slice {slice_str} to {layer_name}")
                for arr in array_pyramid:
                    arr.lazy_op(current_slice)

            # --- Add to Dict ---
            final_obj = array_pyramid if len(array_pyramid) > 1 else array_pyramid[0]
            
            unique_name = layer_name
            count = 1
            while unique_name in arrays_to_show:
                unique_name = f"{layer_name}_{count}"
                count += 1
            
            arrays_to_show[unique_name] = final_obj

    if not arrays_to_show:
        click.echo("No arrays successfully loaded.")
        sys.exit(1)

    click.echo(f"Visualizing {len(arrays_to_show)} layers...")
    
    visualize(
        arrays=arrays_to_show,
        graphs=None,
        bind_address=bind_address,
        port=port,
        browser=not no_browser
    )

if __name__ == "__main__":
    main()