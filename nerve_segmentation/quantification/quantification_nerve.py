import numpy as np
import os
import pandas as pd
import cv2 
import argparse
def quantification_tissue(path_nerveseg_slice, path_tissuemask_slice, path_output_csv):
    nerveseg_slicelist = sorted(os.listdir(path_nerveseg_slice))
    tissuemask_slicelist = sorted(os.listdir(path_tissuemask_slice))

    result_voxel = {}
    for z, z_slice in enumerate(nerveseg_slicelist):
        img_nerveseg = cv2.imread(os.path.join(path_nerveseg_slice, z_slice), -1)
        img_nerveseg = np.squeeze(img_nerveseg)
        print(np.unique(img_nerveseg))

        img_tissue = cv2.imread(os.path.join(path_tissuemask_slice, tissuemask_slicelist[z]), -1)
        img_tissue = np.squeeze(img_tissue)

        tissue_uniques = np.unique(img_tissue)
        tissue_uniques = tissue_uniques[tissue_uniques > 0]
        print(tissue_uniques)
        if len(tissue_uniques)==0:
            continue
        for t_unique in tissue_uniques:
            if 'tissue_' + str(t_unique) not in result_voxel.keys():
                result_voxel.update({'tissue_' + str(t_unique): 0})
            if 'nerve_tissue_' + str(t_unique) not in result_voxel.keys():
                result_voxel.update({'nerve_tissue_' + str(t_unique): 0})
            result_voxel['nerve_tissue_' + str(t_unique)] += np.sum(img_nerveseg * (img_tissue==t_unique))
            result_voxel['tissue_' + str(t_unique)] += np.sum(img_tissue==t_unique)
    print(result_voxel)
    df = pd.DataFrame.from_dict(result_voxel, orient='index', columns=['DateValue'])    
    df.to_csv(path_output_csv)
    
def quantification_organ(path_nerveseg_slice, path_organmask_slice, path_output_csv):
    nerveseg_slicelist = sorted(os.listdir(path_nerveseg_slice))
    organmask_slicelist = sorted(os.listdir(path_organmask_slice))

    result_voxel = {}
    for z, z_slice in enumerate(nerveseg_slicelist):
        print(z_slice, organmask_slicelist[z])
        img_nerveseg = cv2.imread(os.path.join(path_nerveseg_slice, z_slice), -1)
        img_nerveseg = np.squeeze(img_nerveseg)
        print(np.unique(img_nerveseg))

        img_organ = cv2.imread(os.path.join(path_organmask_slice, organmask_slicelist[z]), -1)
        img_organ = np.squeeze(img_organ)
        
        organ_uniques = np.unique(img_organ)
        organ_uniques = organ_uniques[organ_uniques > 0]
        print(organ_uniques)
        if len(organ_uniques)==0:
            continue
        for t_unique in organ_uniques:
            if 'organ_' + str(t_unique) not in result_voxel.keys():
                result_voxel.update({'organ_' + str(t_unique): 0})
            if 'nerve_organ_' + str(t_unique) not in result_voxel.keys():
                result_voxel.update({'nerve_organ_' + str(t_unique): 0})
            result_voxel['nerve_organ_' + str(t_unique)] += np.sum(img_nerveseg * (img_organ==t_unique))
            result_voxel['organ_' + str(t_unique)] += np.sum(img_organ==t_unique)
    print(result_voxel)
    df = pd.DataFrame.from_dict(result_voxel, orient='index', columns=['DateValue'])    
    df.to_csv(path_output_csv)

parser = argparse.ArgumentParser()
parser.add_argument("-nerve_mask", '--input_nerve_seg', help="directory of nerve segmentation tiff slices", required=True)
parser.add_argument("-head_mask", '--input_head_mask', help="directory of head mask tiff slices", required=True)
parser.add_argument("-organ_mask", '--input_organ_mask', help="directory of dilated organ mask tiff slices", required=True)
parser.add_argument('-tissue_mask', "--input_tissue_mask",  help="directory of tissue mask tiff slices", required=True)
parser.add_argument('-o_dir', "--output_dir",  help="directory of quantification results in csv files", required=True)
args = parser.parse_args()

path_nerveseg_slice = args.input_nerve_seg
path_tissuemask_slice = args.input_tissue_mask
path_organmask_slice = args.input_organ_mask
path_headmask_slice = args.input_head_mask
dir_output = args.output_dir

path_tissueoutput = os.path.join(dir_output, 'tissue_quantification.csv')
quantification_tissue(path_nerveseg_slice, path_tissuemask_slice, path_tissueoutput)
path_organoutput = os.path.join(dir_output, 'organ_quantification.csv')
quantification_organ(path_nerveseg_slice, path_organmask_slice, path_organoutput)
path_headoutput = os.path.join(dir_output, 'head_quantification.csv')
quantification_organ(path_nerveseg_slice, path_headmask_slice, path_headoutput)