import argparse
import os
import shutil

import tifffile
import zarr
import dask.array as da
from multiprocessing import Pool, freeze_support
import numpy as np
from dask.diagnostics import ProgressBar

   
    
def main(args):
    output_folder = args.output_folder
    idx_out = args.idx
                                   
    outp_path = os.path.join(output_folder, f"inference_output_{idx_out}.zarr")
    cmap_path = os.path.join(output_folder, f"count_map_{idx_out}.zarr")
    
    outp_da = da.from_zarr(outp_path)
    cmap_da = da.from_zarr(cmap_path)
    
    dataset_shape = cmap_da.shape
    chunk_size = outp_da.chunksize
    result= outp_da / cmap_da
    result[result>=0.5]=1
    result[result<0.5]=0
    result_da= result.astype(np.uint8)
    
    # generate a path for binary outputs    
    binary_out_path = os.path.join(output_folder, f"binaries_{idx_out}.zarr")

    # create and save binary output
    binary_out = zarr.open(binary_out_path, mode='w', 
                            shape=dataset_shape, 
                            dtype=np.uint8, 
                            chunks=chunk_size
                           )
    with ProgressBar():
        result_da.to_zarr(binary_out, compute=True)
    del result_da
    del outp_da
    del cmap_da
    
    os.makedirs(os.path.join(output_folder, f"binary_{idx_out}"), exist_ok=True)
    
    result_da = da.from_zarr(os.path.join(output_folder, f"binaries_{idx_out}.zarr"))
    for z in range(result_da.shape[2]):
        img = result_da[0,0,z, :, :].compute()
        final_path = os.path.join(os.path.join(output_folder, f"binary_{idx_out}"), f"{z:04}.tif")
        tifffile.imwrite(final_path, img.astype(np.uint8), compression='lzw')
    # Delete intermediate Zarr files
    try:
        shutil.rmtree(outp_path)
        shutil.rmtree(cmap_path)
        print(f"Deleted intermediate files: {outp_path}, {cmap_path} and {binary_out_path}")
    except Exception as e:
        print(f"Error deleting intermediate files: {e}")
                            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', "--output_folder",  help="dir of output inference and count map zarr files", required=True)
    parser.add_argument('-id', "--idx",  help="Index identifier for the output binary tiff slice folder", required=True)
    args = parser.parse_args()
    
    freeze_support()
       
    main(args)
