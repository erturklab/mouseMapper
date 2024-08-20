import os
import numpy as np
import cv2
import argparse

def update_intensity_percent(voxel_intensity, update_intensity_low, update_intensity_high, percent):
    voxel_intensity = voxel_intensity.flatten()
    voxel_intensity = voxel_intensity[voxel_intensity>0]
    if len(voxel_intensity)==0:
        return update_intensity_low, update_intensity_high
    voxel_intensity = np.sort(voxel_intensity)
    print('intensity range:', np.min(voxel_intensity), np.max(voxel_intensity))
    
    cur_intensity_low = np.percentile(voxel_intensity, percent)
    
    cur_intensity_high = np.percentile(voxel_intensity, 100-percent)
    
    if cur_intensity_low < update_intensity_low:
        update_intensity_low = cur_intensity_low 
    if cur_intensity_high > update_intensity_high:
        update_intensity_high = cur_intensity_high
    print('Updated intensity low and high:', update_intensity_low, update_intensity_high)
    return update_intensity_low, update_intensity_high
    
def stat_intensity_range_nozero(raw_slice_path, percent):
    raw_slices = sorted(os.listdir(raw_slice_path))
    z_dim = len(raw_slices)
    x_dim, y_dim = cv2.imread(raw_slice_path + raw_slices[0], -1).shape
    
    update_intensity_low = 65535
    update_intensity_high = 0
    for z, z_slice in enumerate(raw_slices):
        img = cv2.imread(raw_slice_path+ z_slice, -1)
        update_intensity_low, update_intensity_high = update_intensity_percent(img, update_intensity_low, update_intensity_high, percent)

    return update_intensity_low, update_intensity_high

parser = argparse.ArgumentParser()
parser.add_argument("-i", '--input_folder', help="directory of raw slices", required=True)
parser.add_argument('-p', "--percent",  type=float, required=True, help="percitile ")
args = parser.parse_args()

raw_patches_path = args.input_folder
(Intensity_low, Intensity_high) = stat_intensity_range_nozero(raw_slice_path=raw_patches_path, percent=args.percent)
print((Intensity_low, Intensity_high))

