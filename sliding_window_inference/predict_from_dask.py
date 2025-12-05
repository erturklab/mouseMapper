import os
import numpy as np
import torch
from torch import nn

import argparse
import datetime
import tifffile
import zarr
import dask.array as da
from multiprocessing import Pool, freeze_support
import hydra
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from batchgenerators.utilities.file_and_folder_operations import load_json
from sliding_window_inferer_zarr_binary import SlidingWindowInferer


import sys
import pickle
import os
import numpy as np
from torch import nn
import torch
from functools import partial
import SimpleITK as sitk
import glob
from os.path import join

import argparse
import datetime
import tifffile
import zarr
import dask.array as da
from multiprocessing import Pool, freeze_support
from dask.distributed import Client
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from batchgenerators.utilities.file_and_folder_operations import load_json
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0


class vesselFM(nn.Module):
    def __init__(self, _deep_supervision=True):
        super().__init__()
        network_cfg = {"_target_": "monai.networks.nets.DynUNet",
                        "in_channels": 1,
                        "out_channels": 2,
                        "spatial_dims": 3,
                        "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                        "kernel_size": [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                        "upsample_kernel_size": [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                        "filters": [32, 64, 128, 256, 320, 320],
                        "deep_supervision": True,
                        "deep_supr_num": 4, 
                        "res_block": True}
        self.network = hydra.utils.instantiate(network_cfg)
        self._deep_supervision = _deep_supervision
        
    def forward(self, x):
        if self._deep_supervision:
            self.network.training = True
            outputs = self.network(x)
            return [outputs[:,i] for i in range(outputs.shape[1])]
        else:
            self.network.training = False
            return self.network(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)


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
    input_folder = args.input_folder
    output_folder = args.output_folder
    threshold = args.threshold
    normalize_min =  args.norm_min
    normalize_max =  args.norm_max
    normalization_mode = args.normalization_mode
    start_l = args.start_l
    end_l = args.end_l
    batch_size = args.batch_size
    overlap_ratio = args.overlap_ratio
    model_path = args.model_path
   
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    
    
    ## For vesselFM tuned models
    #model = vesselFM(_deep_supervision=False)
    
    #checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    results_dir = os.environ.get("nnUNet_results")+'/Dataset17268_CD68/nnUNetTrainer__nnUNetPlans__3d_fullres/'
    plans_file = load_json(os.path.join(results_dir, 'plans.json'))
    dataset_json = load_json(os.path.join(results_dir, 'dataset.json'))
    if model_path is None:
        model_path = results_dir+'/fold_0/checkpoint_best.pth'

    plans_manager = PlansManager(plans_file)
    configuration_manager = plans_manager.get_configuration('3d_fullres')
    
    # Get actual input channels from nnUNet plans
    num_input_channels_model = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
    label_manager = plans_manager.get_label_manager(dataset_json)
    num_output_classes = label_manager.num_segmentation_heads # Get number of output classes
    print(f"nnUNet model expects {num_input_channels_model} input channels and produces {num_output_classes} output classes.")
    

    model = get_network_from_plans(
        configuration_manager.network_arch_class_name,
        configuration_manager.network_arch_init_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        num_input_channels_model, # Pass the correct number of input channels for the model
        2, # Pass the correct number of output classes
        allow_init=True,
        deep_supervision=False)

    #model = vesselFM(_deep_supervision=False)
    
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    torch.cuda.empty_cache()
    

    sw_batch_size = batch_size
    print("using batch size: ",sw_batch_size)
    
    patch_size = [128, 128, 128]
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
            normalization_mode= normalization_mode, # 'zscore' or 'minmax'
            padding_mode = "replicate",
        )
    model = model.to(device)
    model.load_state_dict(checkpoint["network_weights"])
    model.eval()
    
    
    ### DATA PREP ###
    print(f"{datetime.datetime.now()} : Loading Data")
    
    image_dask = da.from_zarr(input_folder)
    
    
    dataset_shape = image_dask.shape
    
    print('Raw image shape:', dataset_shape)
    #create output folder if not already present:
    os.makedirs(os.path.join(output_folder), exist_ok=True)

    # Determine actual channels and spatial shape from the loaded Zarr array
    if image_dask.ndim == 4: # Assuming (C, D, H, W)
        num_channels_in_zarr = image_dask.shape[0]
        spatial_dataset_shape = list(image_dask.shape[1:]) # (D, H, W)
    elif image_dask.ndim == 3: # Assuming (D, H, W) for single channel (fallback, though user specified 2-channel)
        num_channels_in_zarr = 1
        spatial_dataset_shape = list(image_dask.shape)
        print("Warning: Input Zarr is 3D. Assuming single channel. If it's multi-channel, it might need to be reshaped or saved as 4D (C,D,H,W).")
    else:
        raise ValueError(f"Unsupported Zarr shape: {image_dask.shape}. Expected 3D (D,H,W) or 4D (C,D,H,W).")

    if num_channels_in_zarr != num_input_channels_model:
        print(f"WARNING: Zarr file has {num_channels_in_zarr} channels, but nnUNet model expects {num_input_channels_model} channels. This might cause issues.")

    # Final dataset_shape for Zarr output (N, C_out, D, H, W)
    # Note: the N for output is 1 (batch), C_out is num_output_classes, spatial_dataset_shape is D, H, W
    output_zarr_full_shape = [1, num_output_classes] + spatial_dataset_shape
    print(f"Full output Zarr shape (N, C_out, D, H, W): {output_zarr_full_shape}")
    
    # Input path to inferer is just the single Zarr file path
    input_for_inferer = input_folder 

    # Create output folder if not already present:
    os.makedirs(os.path.join(output_folder), exist_ok=True)
    print(f"Output folder created at: {output_folder}")

    # Chunk size for the output Zarr arrays needs to match (N, C_out, D, H, W)
    output_chunk_size = [1, num_output_classes] + patch_size 


    output_zarr_path = os.path.join(output_folder, "inference_output.zarr")
    count_map_zarr_path = os.path.join(output_folder, "count_map.zarr")


    if os.path.exists(output_zarr_path) and os.path.exists(count_map_zarr_path):
        print(f"Output Zarr files already exist. Resuming inference from slice {start_l} to {end_l}.")

        with Pool(processes=4) as pool:
            
            output_image = zarr.open(output_zarr_path, mode='r+',
                                        shape=output_zarr_full_shape, # Shape must match existing Zarr for r+
                                        dtype=np.float16,
                                        chunks=output_chunk_size # Chunks must match existing Zarr for r+
                                        )
            count_map = zarr.open(count_map_zarr_path, mode='r+',
                                    shape=output_zarr_full_shape, # Shape must match existing Zarr for r+
                                    dtype=np.float16,
                                    chunks=output_chunk_size # Chunks must match existing Zarr for r+
                                    )
    else:
        print("Creating new output Zarr files for inference.")

        with Pool(processes=4) as pool:
            
            # No need for os.makedirs(output_zarr_path) here, zarr.open will create parent dirs.
            # But the documentation example used it, so keeping for consistency if it's expected for some reason.
            # zarr.open creates the directory for the .zarr file.
            
            output_image = zarr.open(output_zarr_path, mode='w',
                                        shape=output_zarr_full_shape, # Use the correctly derived shape
                                        dtype=np.float16,
                                        chunks=output_chunk_size
                                        )
            
            count_map = zarr.open(count_map_zarr_path, mode='w',
                                    shape=output_zarr_full_shape, # Use the correctly derived shape
                                    dtype=np.float16,
                                    chunks=output_chunk_size
                                    )           
        
    print(f"output_image shape: {output_image.shape}")
    print(f"count_map shape: {count_map.shape}")
    
    
    # eval
    
     ### Inference
    with torch.no_grad():
        model.eval()
        print(f"Running inferrer with input: {input_for_inferer}, output_image type: {type(output_image)}, count_map type: {type(count_map)}")
        # Pass the single input_folder string
        inferer(input_path=input_for_inferer, network=model, output_image=output_image, count_map=count_map, start_slice=start_l, end_slice=end_l,normalization_mode = normalization_mode)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input_folder', help="Path to the single-channel Zarr input file (e.g., /path/to/my_image.zarr)",
                        required=True)
    parser.add_argument('-o', "--output_folder", help="path of output inference npy file",
                        required=True)
    parser.add_argument('-thresh', "--threshold", type=float, default = 0.5, required=False, help="lower boundary for prediction")
    parser.add_argument('-min', "--norm_min", type=float, default = 100, required=False, help="lower boundary for normalization")
    parser.add_argument('-max', "--norm_max", type=float, default = 60000, required=False, help="upper boundary for normalization")
    parser.add_argument('-normalization', "--normalization_mode", type=str, default = 'zscore', required=False, help="normalization type: minmax or zscore")
    parser.add_argument('-start', "--start_l", type=int, default= 0, help="starting batch indix")
    parser.add_argument('-end', "--end_l", type=int, default= -1, help="Ending batch indix")
    parser.add_argument('-bs', "--batch_size", type=int, default= 4, help="batch size")
    parser.add_argument('-overlap', "--overlap_ratio", type=float, default= 0.25, help="overlap ration for inference")
    parser.add_argument('-model',"--model_path", type=str, required=False, default = None, help="path of binary model")
    args = parser.parse_args()
 
    
    freeze_support()
    
    main(args)
     
