import numpy as np
import os
import pandas as pd
import tifffile 
import argparse
from collections import defaultdict

def quantification_wholebody(path_nerveseg_slice, path_wholebodymask_slice, path_output_csv):
    nerveseg_slicelist = sorted(os.listdir(path_nerveseg_slice))
    wholebodymask_slicelist = sorted(os.listdir(path_wholebodymask_slice))
    
    result_voxel = {}
    result_voxel.update({'nerve_voxel': 0})
    result_voxel.update({'wholebody_voxel': 0})
    for z, z_slice in enumerate(nerveseg_slicelist):
        print(z_slice, wholebodymask_slicelist[z])
        img_nerveseg = tifffile.imread(os.path.join(path_nerveseg_slice, z_slice))
        img_nerveseg = np.squeeze(img_nerveseg)
        if len(np.unique(img_nerveseg))==1:
            continue
       
        img_wholebody = tifffile.imread(os.path.join(path_wholebodymask_slice, wholebodymask_slicelist[z]))
        img_wholebody = np.squeeze(img_wholebody)
        

        result_voxel['nerve_voxel'] += np.sum(img_nerveseg)
        result_voxel['wholebody_voxel'] += np.sum(img_wholebody)

    df = pd.DataFrame.from_dict(result_voxel, orient='index', columns=['DateValue'])    
    df.to_csv(path_output_csv)

def quantification_tissue(path_nerveseg_slice, path_tissuemask_slice, path_output_csv):
    nerveseg_slicelist = sorted(os.listdir(path_nerveseg_slice))
    tissuemask_slicelist = sorted(os.listdir(path_tissuemask_slice))

    tissue_voxel = defaultdict(int)
    nerve_tissue_voxel = defaultdict(int)
    for z, z_slice in enumerate(nerveseg_slicelist):
        print(z_slice, tissuemask_slicelist[z])
        img_nerveseg = tifffile.imread(os.path.join(path_nerveseg_slice, z_slice))
        if np.all(img_nerveseg==0):
            continue

        img_tissue = tifffile.imread(os.path.join(path_tissuemask_slice, tissuemask_slicelist[z]))
        if np.all(img_tissue==0):
            continue
        tissue_ids = np.unique(img_tissue)
        tissue_ids = tissue_ids[tissue_ids > 0]

        # Vectorized count of tissue voxels
        counts = np.bincount(img_tissue.ravel())
        # Vectorized count of nerve-tissue overlap
        nerve_counts = np.bincount(img_tissue.ravel(), weights=img_nerveseg.ravel())

        for t in tissue_ids:
            tissue_voxel[f'tissue_{t}'] += counts[t]
            nerve_tissue_voxel[f'nerve_tissue_{t}'] += nerve_counts[t]
    #print(result_voxel)
    result_voxel = {**tissue_voxel, **nerve_tissue_voxel}
    df = pd.DataFrame.from_dict(result_voxel, orient='index', columns=['DateValue'])    
    df.to_csv(path_output_csv)
    
def quantification_organ(path_nerveseg_slice, path_organmask_slice, path_output_csv):
    nerveseg_slicelist = sorted(os.listdir(path_nerveseg_slice))
    organmask_slicelist = sorted(os.listdir(path_organmask_slice))

    organ_voxel = defaultdict(int)
    nerve_organ_voxel = defaultdict(int)
    for z, z_slice in enumerate(nerveseg_slicelist):
        print(z_slice, organmask_slicelist[z])
        img_nerveseg = tifffile.imread(os.path.join(path_nerveseg_slice, z_slice))
        if np.all(img_nerveseg == 0):
            continue
        img_organ = tifffile.imread(os.path.join(path_organmask_slice, organmask_slicelist[z]))
        if np.all(img_organ == 0):
            continue
        organ_ids = np.unique(img_organ)
        organ_ids = organ_ids[organ_ids > 0]

        counts = np.bincount(img_organ.ravel())
        # Vectorized count of nerve-tissue overlap
        nerve_counts = np.bincount(img_organ.ravel(), weights=img_nerveseg.ravel())

        for t in organ_ids:
            organ_voxel[f'organ_{t}'] += counts[t]
            nerve_organ_voxel[f'nerve_organ_{t}'] += nerve_counts[t]

    result_voxel = {**organ_voxel, **nerve_organ_voxel}
    df = pd.DataFrame.from_dict(result_voxel, orient='index', columns=['DateValue'])    
    df.to_csv(path_output_csv)

def main():       
    parser = argparse.ArgumentParser()
    parser.add_argument("-nerve_seg", '--input_nerve_seg', help="directory of nerve segmentation tiff slices", required=True)
    parser.add_argument("-head_mask", '--input_head_mask', default = None, help="directory of head mask tiff slices")
    parser.add_argument("-wb_mask", '--input_wholebody_mask', default = None, help="directory of wholebody mask tiff slices")
    parser.add_argument("-organ_mask", '--input_organ_mask', default = None, help="directory of dilated organ mask tiff slices")
    parser.add_argument('-tissue_mask', "--input_tissue_mask",  default = None, help="directory of tissue mask tiff slices")
    parser.add_argument('-o', "--output_dir",  help="directory of quantification results in csv files", required=True)
    args = parser.parse_args()
    
    path_nerveseg_slice = args.input_nerve_seg
    path_tissuemask_slice = args.input_tissue_mask
    path_organmask_slice = args.input_organ_mask
    path_headmask_slice = args.input_head_mask
    path_wbmask_slice = args.input_wholebody_mask
    dir_output = args.output_dir
    
    if path_wbmask_slice is not None:
        path_wboutput = os.path.join(dir_output, 'wholebody_quantification.csv')
        quantification_wholebody(path_nerveseg_slice, path_wbmask_slice, path_wboutput)
    if path_tissuemask_slice is not None:
        path_tissueoutput = os.path.join(dir_output, 'tissue_quantification.csv')
        quantification_tissue(path_nerveseg_slice, path_tissuemask_slice, path_tissueoutput)
    
    if path_organmask_slice is not None:
        path_organoutput = os.path.join(dir_output, 'organ_quantification.csv')
        quantification_organ(path_nerveseg_slice, path_organmask_slice, path_organoutput)
    if path_headmask_slice is not None:
        path_headoutput = os.path.join(dir_output, 'head_quantification.csv')
        quantification_organ(path_nerveseg_slice, path_headmask_slice, path_headoutput)
    

if __name__ == "__main__":
    main()



