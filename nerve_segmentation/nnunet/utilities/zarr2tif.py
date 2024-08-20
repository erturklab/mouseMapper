import argparse
import zarr
import os
import numpy as np
from multiprocessing import Pool, freeze_support
#from dask.distributed import Client
import dask.array as da
import tifffile
from dask.diagnostics import ProgressBar
import datetime
from nnunet.inference.predict_simple_dask import parse_plans, get_shape


def main(args):
    tiff_path = args.input_raw_folder
    output_folder = args.output_folder
    plans_file_path = args.plans_path
    dataset_shape = get_shape(tiff_path)
    
    plans = parse_plans(plans_file_path=plans_file_path)
    chunk_size = [1, 1] + list(plans['patch_size'])
    with Pool(processes=4) as pool:
        output_image = zarr.open(os.path.join(output_folder, "inference_output.zarr"), mode='r', 
                                    shape=dataset_shape, 
                                    dtype=np.float16, 
                                    chunks=chunk_size)
    with Pool(processes=4) as pool:    
        count_map = zarr.open(os.path.join(output_folder,  "count_map.zarr"), mode='r', 
                                    shape=dataset_shape, 
                                    dtype=np.float16, 
                                    chunks=chunk_size
                                    )
    
    outp_da = da.from_zarr(output_image)
    cmap_da = da.from_zarr(count_map)
    result= outp_da / cmap_da
    result[result>=0.5]=1
    result[result<0.5]=0
    result_da= result.astype(np.uint8)
    
    print(f"{datetime.datetime.now()} : Creating binarized output")
    # generate a path for binary outputs
    binary_out_path = os.path.join(output_folder, "binaries.zarr")
    os.makedirs(binary_out_path, exist_ok=True)
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
    
    os.makedirs(os.path.join(output_folder, 'binary'), exist_ok=True)
    print(binary_out.shape)
    for z in range(binary_out.shape[2]):
        img = np.array(binary_out[0,0,z])
        final_path = os.path.join(os.path.join(output_folder, 'binary'), f"{z:04}.tif")
        print(np.unique(img))
        tifffile.imwrite(final_path, img.astype(np.uint8))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input_raw_folder", help="directory of raw slices", required=True)
    parser.add_argument('-pl', "--plans_path", help="Path of plans.pkl for the model", required=True)
    parser.add_argument('-o', "--output_folder",  help="path of output folders of model inference", required=True)
    args = parser.parse_args()
    
    freeze_support()
    
    #client = Client(timeout='60s')
    
    main(args)