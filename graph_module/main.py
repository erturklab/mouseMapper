import os
import json5 as json
from extract_merge_graph import patchify_voxel, process_crop, merge_graphs
from concurrent.futures import ProcessPoolExecutor, as_completed
import zarr
import numpy as np
from tqdm import tqdm

import argparse

os.environ['ZARR_V3_EXPERIMENTAL_API'] = '1'
os.environ['ZARR_V3_SHARDING'] = '1'

print(f"ZARR version {zarr.__version__}")
# For zarr v3, try disabling locking
if int(zarr.__version__.split(".")[0]) > 2: 
    zarr.config.set({'array.order': 'C', 'codec_pipeline.batch_size': 1})


class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)


def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./config.json", help="Path to config JSON")
    args = parser.parse_args()
    
    

    path_config = args.path 

    config = {}

    print(f"Reading {path_config}")

    with open(path_config, "rb") as file:
        config = json.load(file)

    print(f"Config:\n{config}")

    config["output_dir"] = os.path.join(config["workdir"],"voreen_output/")
    config["tempdir"] = os.path.join(config["workdir"], "voreen_tmpdir/")
    config["cachedir"] = os.path.join(config["workdir"], "voreen_cachedir/")
    config = dict2obj(config)

    print(f"Opening in_file {config.in_file} ({os.listdir(config.in_file)})")
    # Open the Zarr file
    zarr_data = zarr.open(config.in_file, mode='r', synchronizer=None)
    shape_ = zarr_data.shape

    span = tuple(np.array(config.patch_size)-2*np.array(config.pad))
    seg_patch_list, start_ind, seq_ind = patchify_voxel(shape_, config.patch_size, config.pad)

############################################# Extract Graphs #############################################
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.tempdir, exist_ok=True)
    os.makedirs(config.cachedir, exist_ok=True)
    with ProcessPoolExecutor(max_workers=config.num_threads) as executor:
        futures = [
            executor.submit(
                process_crop,
                crop_idx, start_, seq_, zarr_data, config
            )
            for crop_idx, start_, seq_ in zip(seg_patch_list, start_ind, seq_ind)
        ]

        # Use `as_completed` to update the progress bar as tasks complete
        with tqdm(total=len(futures), ncols=110) as pbar:
            for future in as_completed(futures):
                pbar.update(1)

############################################# Merge Graphs #############################################
    # list all .vtp files in the vtp_path
    vtp_files = [f for f in os.listdir(config.output_dir) if f.endswith('.vtp')]

    graph = merge_graphs(vtp_files, config.output_dir, span)

    # save the graph
    graph.save(config.out_file)
