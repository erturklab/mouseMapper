import os
import nibabel as nib
import numpy as np
import scipy 
from libtiff import TIFFimage
import cv2
from skimage.segmentation import expand_labels
import argparse

def readNifti(path,reorient=None):
    '''
    volume = readNifti(path)
    
    Reads in the NiftiObject saved under path and returns a Numpy volume.
    This function can also read in .img files (ANALYZE format).
    '''
    if(path.find('.nii')==-1 and path.find('.img')==-1):
        path = path + '.nii'
    print(path)
    if(os.path.isfile(path)):    
        NiftiObject = nib.load(path)
    elif(os.path.isfile(path + '.gz')):
        NiftiObject = nib.load(path + '.gz')
    else:
        raise Exception("No file found at: "+path)
    # Load volume and adjust orientation from (x,y,z) to (y,x,z)
    volume = np.swapaxes(NiftiObject.dataobj,0,1)
    if(reorient=='uCT_Rosenhain' and path.find('.img')):
        # Only perform this when reading in raw .img files
        # from the Rosenhain et al. (2018) dataset
        #    y = from back to belly
        #    x = from left to right
        #    z = from toe to head
        volume = np.swapaxes(volume,0,2) # swap y with z
        volume = np.flip(volume,0) # head  should by at y=0
        volume = np.flip(volume,2) # belly should by at x=0
    return volume

def writeNifti(path,volume,compress=False):
    '''
    writeNifti(path,volume)
    
    Takes a Numpy volume, converts it to the Nifti1 file format, and saves it to file under
    the specified path. 
    '''
    if(path.find('.nii')==-1 and compress==False):
        path = path + '.nii'
    if(path.find('.nii.gz')==-1 and compress==True):
        path = path + '.nii.gz'
    folderpath = '/'.join([folder for folder in path.split('/')[0:-1]])
    if(os.path.isdir(folderpath) == False):
        os.makedirs(folderpath) # create folder(s) if missing so far.
    # Save volume with adjusted orientation
    # --> Swap X and Y axis to go from (y,x,z) to (x,y,z)
    # --> Show in RAI orientation (x: right-to-left, y: anterior-to-posterior, z: inferior-to-superior)
    affmat = np.eye(4)
    affmat[0,0] = affmat[1,1] = -1
    NiftiObject = nib.Nifti1Image(np.swapaxes(volume,0,1), affine=affmat)
    nib.save(NiftiObject,path)

def grow_head_mask(path_organ_mask, path_grow_output, niter = 115):
    
    np_organ_mask = readNifti(path_organ_mask)
    np_head_mask = np.zeros_like(np_organ_mask)
    np_head_mask[np_organ_mask==5]=1
    del np_organ_mask
    print(np_head_mask.shape)
    dilate_rate_x = 1 
    dilate_rate_y = 1
    dilate_rate_z = 3
    struct_elem = np.zeros((2 * dilate_rate_x + 1, 2 * dilate_rate_y + 1, 2 * dilate_rate_z + 1), dtype=bool)
    struct_elem[dilate_rate_x, :, :] = 1
    struct_elem[:, dilate_rate_y, :] = 1
    struct_elem[:, :, dilate_rate_z] = 1
    print(struct_elem)
    print(struct_elem.shape)
    np_head_mask = scipy.ndimage.binary_dilation(np_head_mask, structure=struct_elem, iterations=niter).astype(np.uint8)
    writeNifti(path_grow_output, np_head_mask, compress=True)

def grow_organ_mask(path_organ_mask, path_grow_output):
    np_organ_mask = readNifti(path_organ_mask)

    # grow the mask for organs without brain
    np_organ_mask[np_organ_mask==5]=0
    np_organ_mask = expand_labels(np_organ_mask, distance=15)
    
    writeNifti(path_grow_output, np_organ_mask, compress=True)

parser = argparse.ArgumentParser()
parser.add_argument("-i_mask", '--input_organ_mask', help="path of organ mask", required=True)
parser.add_argument('-o_mask', "--output_dilated_masks",  help="directory of output dilated masks", required=True)
args = parser.parse_args()

path_organ_mask = args.input_organ_mask
dir_dilated_masks = args.output_dilated_masks

path_dilatedhead_output = os.path.join(dir_dilated_masks, 'headmask_dilate_xy5z5.nii.gz')
grow_head_mask(path_organ_mask, path_dilatedhead_output)

path_dilatedorgan_output = os.path.join(dir_dilated_masks, 'organmask_dilate_xy5z5.nii.gz')
grow_organ_mask(path_organ_mask, path_dilatedorgan_output)