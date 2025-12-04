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

def generate_tiff_from_tissue_pred_zarr(input_folder, output_folder):
    fld_in=input_folder
    im_1 = dask.array.from_zarr(fld_in+'/count_map.zarr')
    im_2 = dask.array.from_zarr(fld_in+'/inference_output.zarr')

    fld_out = output_folder
    os.makedirs(fld_out, exist_ok=True)

    result = im_2*255/im_1 

    for j in tqdm(range(result.shape[2])):
        output_path = Path(fld_out) / f"tissue_map{j:04d}.tif"
        
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
    args = parser.parse_args()
    
    
    zarr_path = args.input_prediction_path
    tiff_dir = args.output_tiff_path

    generate_tiff_from_tissue_pred_zarr(zarr_path, tiff_dir)

if __name__ == '__main__':  
    main()