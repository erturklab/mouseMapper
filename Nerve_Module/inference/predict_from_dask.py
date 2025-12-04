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

from sliding_window_inferer_zarr import SlidingWindowInferer


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
    start_l = args.start_l
    end_l = args.end_l
    batch_size = args.batch_size
    overlap_ratio = args.overlap_ratio
    model_path = args.model
    idx_out = args.idx
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    
    
    ## For vesselFM tuned models
    model = vesselFM(_deep_supervision=False)
    
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    torch.cuda.empty_cache()

    sw_batch_size = batch_size
    print("using batch size: ",sw_batch_size)
    patch_size = [256, 256, 256]
    #patch_size = [128, 128, 128]
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
    model.load_state_dict(checkpoint["network_weights"])
    model.eval()
    
    
    ### DATA PREP ###
    print(f"{datetime.datetime.now()} : Loading Data")
    
    #load dataset
    if input_folder.find(".zarr") != -1:
        image = da.from_zarr(input_folder)
        dataset_shape = image.shape
        dataset_shape = [1,1,dataset_shape[0], dataset_shape[1], dataset_shape[2]]
    else:
        dataset_shape = get_shape(input_folder)
    print('Raw image shape:', dataset_shape)
    #create output folder if not already present:
    os.makedirs(os.path.join(output_folder), exist_ok=True)
    
    chunk_size = [1, 1] + patch_size
    if start_l == 0:
        with Pool(processes=4) as pool:
            #os.makedirs(os.path.join(output_folder, f"inference_output_{idx_out}.zarr"), exist_ok=True)
            output_image = zarr.open(os.path.join(output_folder, f"inference_output_{idx_out}.zarr"), mode='w', 
                                    shape=dataset_shape, 
                                    dtype=np.float16, 
                                    chunks=chunk_size, 
                                    compressor=None
                                    )
        with Pool(processes=4) as pool:    
            #os.makedirs(os.path.join(output_folder, "count_map.zarr"), exist_ok=True)
            count_map = zarr.open(os.path.join(output_folder,  f"count_map_{idx_out}.zarr"), mode='w', 
                                    shape=dataset_shape, 
                                    dtype=np.float16, 
                                    chunks=chunk_size, 
                                    compressor=None
                                    )
    else:
        with Pool(processes=4) as pool:
            output_image = zarr.open(os.path.join(output_folder, f"inference_output_{idx_out}.zarr"), mode='r+', 
                                    shape=dataset_shape, 
                                    dtype=np.float16, 
                                    chunks=chunk_size, 
                                    compressor=None)
        with Pool(processes=4) as pool:    
            count_map = zarr.open(os.path.join(output_folder,  f"count_map_{idx_out}.zarr"), mode='r+', 
                                    shape=dataset_shape, 
                                    dtype=np.float16, 
                                    chunks=chunk_size, 
                                    compressor=None)
        
    print("output_image shape",output_image.shape)
    print("count_map shape",count_map.shape)
    
    
    # eval
    
    with torch.no_grad():
        model.eval()    
        print(f"Running inferrer with {input_folder} {type(output_image)} {type(count_map)}")
        outputi = output_image
        inferer(input_path=input_folder, network=model, output_image = outputi, count_map = count_map, start_slice = start_l, end_slice = end_l)                    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input_folder', help="directory of raw input", required=True)
    parser.add_argument('-o', "--output_folder",  help="path of output inference npy file", required=True)
    parser.add_argument('-m', "--model",  help="path of model pth file", required=True)
    parser.add_argument('-id', "--idx",  help="indicator for saving the output", required=True)
    parser.add_argument('-thresh', "--threshold",  type=float, required=True, help="lower boundary for prediction")
    parser.add_argument('-min', "--norm_min",  type=float, required=True, help="lower boundary for normalization")
    parser.add_argument('-max', "--norm_max",  type=float, required=True, help="upper boundary for normalization")
    parser.add_argument('-start', "--start_l",  type=int, default= 0, help="starting batch indix")
    parser.add_argument('-end', "--end_l",  type=int, default= -1, help="Ending batch indix")
    parser.add_argument('-bs', "--batch_size",  type=int, default= 2, help="batch size")
    parser.add_argument('-overlap', "--overlap_ratio",  type=float, default= 0.5, help="overlap ration for inference")
    args = parser.parse_args()
    
    freeze_support()
    
    main(args)
     
