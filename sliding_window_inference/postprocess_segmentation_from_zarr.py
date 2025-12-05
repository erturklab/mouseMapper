import os                                                                                                                                                                                                                                                                                                                                                                                                                                 
import dask.array as da                                                                                                                                                                                                                                                                                                                                                                                                                   
from dask.array.image import imread                                                                                                                                                                                                                                                                                                                                                                                                       
from pathlib import Path                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
from dask.diagnostics import ProgressBar                                                                                                                                                                                                                                                                                                                                                                                                  
import dask
import numpy as np
import cv2
from tqdm import tqdm
import argparse
from pathlib import Path
import sys
import shutil

def generate_tiff_from_tissue_pred_zarr(input_folder, output_folder,save_as_zarr):
    fld_in=input_folder
    im_1 = dask.array.from_zarr(fld_in+'/count_map.zarr')
    im_2 = dask.array.from_zarr(fld_in+'/inference_output.zarr')

    fld_out = output_folder
    os.makedirs(fld_out, exist_ok=True)

    result = im_2*255/im_1 

    if save_as_zarr:
        zarr_out_path = os.path.join(fld_out, 'segmentation.zarr')
        os.makedirs(zarr_out_path, exist_ok=True)
        result_zarr = zarr.open(zarr_out_path, mode='w', 
                                shape=result.shape, 
                                dtype=np.uint8, 
                                chunks=result.chunksize
                               )
        with ProgressBar():
            result.to_zarr(result_zarr, compute=True)
        print(f"Saved segmentation as Zarr at {zarr_out_path}")

    for j in tqdm(range(result.shape[2])):
        output_path = Path(fld_out) / f"segmentation{j:04d}.tif"
        
        if not os.path.isfile(output_path):
            output =result[:,:,j].compute()#.reshape((result.shape[3], result.shape[4]))>0.5).astype(np.uint8)

            output = np.argmax(output,axis=1).reshape((result.shape[3], result.shape[4])) 

            cv2.imwrite(str(output_path), (output).astype(np.uint8))
            if (j%100==0):
                print(f"Processed slice {j} and saved to {output_path}")
    print(f"Processed {result.shape[2]} slices and saved to {fld_out}")


def main():       
    parser = argparse.ArgumentParser(description="Convert TIFF stack to chunked Zarr format")
    parser.add_argument("-i", '--input_prediction_path', help="Directory of raw tiff slices", required=True)
    parser.add_argument("-o", '--output_tiff_path', help="Output zarr path", required=True)
    parser.add_argument("-as_zarr", '--as_zarr', action='store_true', help="If set,also saves output as zarr and tiff", required=False, default=False)
    args = parser.parse_args()
    
    
    zarr_path = args.input_prediction_path
    tiff_dir = args.output_tiff_path
    save_as_zarr = args.as_zarr

    generate_tiff_from_tissue_pred_zarr(zarr_path, tiff_dir,save_as_zarr)

if __name__ == '__main__':  
    main()