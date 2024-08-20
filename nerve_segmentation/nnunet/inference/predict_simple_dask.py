import pickle
import os
import numpy as np
from torch import nn
import torch
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from functools import partial
import SimpleITK as sitk
import glob
from os.path import join
from sliding_window_inferer_zarr import SlidingWindowInferer
import argparse
import datetime
import tifffile
import zarr
import dask.array as da
from multiprocessing import Pool, freeze_support
from dask.distributed import Client
from batchgenerators.utilities.file_and_folder_operations import *


def parse_plans(plans_file_path):
    args = {}
    plans=load_pickle(plans_file_path)
    
    stage = list(plans['plans_per_stage'].keys())[0]
    stage_plans = plans['plans_per_stage'][stage]
    args['batch_size'] = stage_plans['batch_size']
    args['net_pool_per_axis'] = stage_plans['num_pool_per_axis']
    args['patch_size']= np.array(stage_plans['patch_size']).astype(int)
    args['do_dummy_2D_aug'] = stage_plans['do_dummy_2D_data_aug']
    if 'pool_op_kernel_sizes' not in stage_plans.keys():
        assert 'num_pool_per_axis' in stage_plans.keys()
        net_num_pool_op_kernel_sizes = []
        for i in range(max(args['net_pool_per_axis'])):
            curr = []
            for j in args['net_pool_per_axis']:
                if (max(args['net_pool_per_axis']) - j) <= i:
                    curr.append(2)
                else:
                    curr.append(1)
            net_num_pool_op_kernel_sizes.append(curr)
    else:
        net_num_pool_op_kernel_sizes = stage_plans['pool_op_kernel_sizes']
    args['net_num_pool_op_kernel_sizes'] = net_num_pool_op_kernel_sizes
    if 'conv_kernel_sizes' not in stage_plans.keys():
        net_conv_kernel_sizes = [[3] * len(args['net_pool_per_axis'])] * (max(args['net_pool_per_axis']) + 1)
    else:
        net_conv_kernel_sizes = stage_plans['conv_kernel_sizes']

    args['net_conv_kernel_sizes']=net_conv_kernel_sizes

    args['pad_all_sides'] = None  
    args['intensity_properties'] = plans['dataset_properties']['intensityproperties']
    args['normalization_schemes'] = plans['normalization_schemes']
    args['base_num_features'] = plans['base_num_features']
    args['num_input_channels'] = plans['num_modalities']
    args['num_classes'] = plans['num_classes'] + 1 
    args['classes'] = plans['all_classes']
    args['use_mask_for_norm'] = plans['use_mask_for_norm']
    args['only_keep_largest_connected_component'] = plans['keep_only_largest_region']
    args['min_region_size_per_class'] = plans['min_region_size_per_class']
    args['min_size_per_class'] = None 

    if plans.get('transpose_forward') is None or plans.get('transpose_backward') is None:
        print("WARNING! You seem to have data that was preprocessed with a previous version of nnU-Net. "
                "You should rerun preprocessing. We will proceed and assume that both transpose_foward "
                "and transpose_backward are [0, 1, 2]. If that is not correct then weird things will happen!")
        plans['transpose_forward'] = [0, 1, 2]
        plans['transpose_backward'] = [0, 1, 2]
    args['transpose_forward'] = plans['transpose_forward']
    args['transpose_backward'] = plans['transpose_backward']
    if len(args['patch_size']) == 2:
        args['threeD'] = False
    elif len(args['patch_size']) == 3:
        args['threeD'] = True
    else:
        raise RuntimeError("invalid patch size in plans file: %s" % str(args['patch_size']))
    
    if "conv_per_stage" in plans.keys():  # this ha sbeen added to the plans only recently
        args['conv_per_stage'] = plans['conv_per_stage']
    else:
        args['conv_per_stage'] = 2

    if args['threeD']:
        args['conv_op'] = nn.Conv3d
        args['dropout_op'] = nn.Dropout3d
        args['norm_op'] = nn.InstanceNorm3d
    else:
        args['conv_op'] = nn.Conv2d
        args['dropout_op'] = nn.Dropout2d
        args['norm_op'] = nn.InstanceNorm2d
    args['num_pool'] = len(net_num_pool_op_kernel_sizes)

    args['norm_op_kwargs'] = {'eps': 1e-5, 'affine': True}
    args['dropout_op_kwargs'] = {'p': 0, 'inplace': True}
    args['net_nonlin'] = nn.LeakyReLU
    args['net_nonlin_kwargs'] = {'negative_slope': 1e-2, 'inplace': True}

    return args

def get_shape(file_location):
    input_list = sorted([x for x in os.listdir(file_location) if ".tif" in x])
    z_ = len(input_list)
    store = tifffile.imread(os.path.join(file_location) + input_list[0], aszarr=True)
    yx = zarr.open(store).shape
    yx = list(yx)
    img_shape = [1, 1, z_]
    img_shape.extend(yx)
    return img_shape

def main(args):
    tiff_path = args.input_slice_folder
    output_folder = args.output_folder
    threshold = args.threshold
    normalize_min =  args.norm_min
    normalize_max =  args.norm_max
    start_l = args.start_l
    batch_size = args.batch_size
    overlap_ratio = args.overlap_ratio
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    model_path = args.model_path
    plans_file_path = args.plans_path
    plans = parse_plans(plans_file_path=plans_file_path)
    model = Generic_UNet(plans['num_input_channels'], plans['base_num_features'], plans['num_classes'], plans['num_pool'], 
                        num_conv_per_stage=plans['conv_per_stage'], feat_map_mul_on_downscale=2, conv_op=plans['conv_op'],
                        norm_op=plans['norm_op'], norm_op_kwargs=plans['norm_op_kwargs'], dropout_op=plans['dropout_op'], 
                        dropout_op_kwargs=plans['dropout_op_kwargs'],nonlin=plans['net_nonlin'], 
                        nonlin_kwargs=plans['net_nonlin_kwargs'], deep_supervision=False, dropout_in_localization=False,
                        weightInitializer=InitWeights_He(1e-2), 
                        pool_op_kernel_sizes=plans['net_num_pool_op_kernel_sizes'], conv_kernel_sizes=plans['net_conv_kernel_sizes'], 
                        upscale_logits=False, convolutional_pooling=True, convolutional_upsampling=True)
    checkpoint = torch.load(model_path, map_location="cpu")
    torch.cuda.empty_cache()

    sw_batch_size = batch_size
    print("using batch size: ",sw_batch_size)
    
    patch_size = [256, 256, 256]
    inferer = SlidingWindowInferer(
            roi_size=patch_size,
            sw_batch_size=sw_batch_size,
            sw_device= torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            device=torch.device('cpu'),
            threshold_inference = threshold,
            normalize_min = normalize_min,
            normalize_max = normalize_max,
            overlap = overlap_ratio,
            mode = "gaussian",
            padding_mode = "replicate",
        )
    model = model.to(device)
    model.load_state_dict(checkpoint['state_dict'])
    
    
    ### DATA PREP ###
    print(f"{datetime.datetime.now()} : Loading Data")
    
    #load dataset
    dataset_shape = get_shape(tiff_path)
    
    #create output folder if not already present:
    os.makedirs(os.path.join(output_folder), exist_ok=True)
    
    chunk_size = [1, 1] + patch_size
    if start_l == 0:
        with Pool(processes=4) as pool:
            os.makedirs(os.path.join(output_folder, "inference_output.zarr"), exist_ok=True)
            output_image = zarr.open(os.path.join(output_folder, "inference_output.zarr"), mode='w', 
                                    shape=dataset_shape, 
                                    dtype=np.float16, 
                                    chunks=chunk_size
                                    )
        with Pool(processes=4) as pool:    
            os.makedirs(os.path.join(output_folder, "count_map.zarr"), exist_ok=True)
            count_map = zarr.open(os.path.join(output_folder,  "count_map.zarr"), mode='w', 
                                    shape=dataset_shape, 
                                    dtype=np.float16, 
                                    chunks=chunk_size
                                    )
    else:
        with Pool(processes=4) as pool:
            output_image = zarr.open(os.path.join(output_folder, "inference_output.zarr"), mode='r+', 
                                    shape=dataset_shape, 
                                    dtype=np.float16, 
                                    chunks=chunk_size)
        with Pool(processes=4) as pool:    
            count_map = zarr.open(os.path.join(output_folder,  "count_map.zarr"), mode='r+', 
                                    shape=dataset_shape, 
                                    dtype=np.float16, 
                                    chunks=chunk_size
                                    )
        
    print("output_image shape",output_image.shape)
    print("count_map shape",count_map.shape)
    
    
    # eval
    with torch.no_grad():
        model.eval()    
        print(f"Running inferrer with {tiff_path} {type(output_image)} {type(count_map)}")
        outputi = output_image
        inferer(input_path=tiff_path, network=model, output_image = outputi, count_map = count_map, start_slice = start_l)                    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input_slice_folder", help="directory of raw slices", required=True)
    parser.add_argument('-o', "--output_folder",  help="path of output inference npy file", required=True)
    parser.add_argument('-thresh', "--threshold",  type=float, required=True, help="lower boundary for prediction")
    parser.add_argument('-min', "--norm_min",  type=float, required=True, help="lower boundary for normalization")
    parser.add_argument('-max', "--norm_max",  type=float, required=True, help="upper boundary for normalization")
    parser.add_argument('-mdl', "--model_path",  help="Path of trained model", required=True)
    parser.add_argument('-pl', "--plans_path", help="Path of plans.pkl for the model", required=True)

    parser.add_argument('-start', "--start_l",  type=int, default= 0, help="starting batch indix")
    parser.add_argument('-bs', "--batch_size",  type=int, default= 2, help="batch size")
    parser.add_argument('-overlap', "--overlap_ratio",  type=float, default= 0.5, help="overlap ration for inference")
    args = parser.parse_args()
    
    freeze_support()
    
    #client = Client(timeout='60s')
    
    main(args)
     

