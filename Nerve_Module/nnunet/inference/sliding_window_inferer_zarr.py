#Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any, Callable, List, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
import datetime

from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.utils import BlendMode, PytorchPadMode, fall_back_tuple, look_up_option
from monai.inferers.inferer import Inferer
from monai.transforms import RandGaussianNoise

import tifffile
import zarr
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
from dask.array.image import imread as imread_dask
from dask.distributed import Client, progress
from pathlib import Path

import datetime

__all__ = ["sliding_window_inference_zarr"]

def read_patch(location, shape : [int], z :int, y:int, x:int) -> np.array:
    """Returns a numpy array from a folder of TIFF files using lazy loading
    Args:
        - location: path to folder of TIFFs
        - shape: shape of the resulting array
        - x, y, z: coordinates at (0,0,0) for the array
    """
    start = datetime.datetime.now()
    images_path = Path(location, "*.tif")
    images = imread_dask(str(images_path))
    patch = images[z:z + shape[0],
                y:y + shape[1],
                x:x + shape[2]].compute()
    # return patch
    patch = da.expand_dims(patch, axis=0)
    patch = da.expand_dims(patch, axis=0)
    delta = datetime.datetime.now() - start
    print(f"Patch at {z} {y} {x} took {delta}")

    return patch


def read_patch_dask(image, shape : [int], location : [int]) -> np.array:
    """Read a single patch from a dask image.
    Args:
        - image: the dask image
        - shape: shape of the resulting array
        - location: [z, y, x] coordinates at (0,0,0) for the array
    """
    patch = image[location[0]:location[0] + shape[0],
                  location[1]:location[1] + shape[1],
                  location[2]:location[2] + shape[2]]
    return patch

def _get_scan_interval(
    image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: float
) -> Tuple[int, ...]:
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.

    """
    if len(image_size) != num_spatial_dims:
        raise ValueError("image coord different from spatial dims.")
    if len(roi_size) != num_spatial_dims:
        raise ValueError("roi coord different from spatial dims.")

    scan_interval = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - overlap))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)


def initialize_dask(location : str, chunk_shape : [int], image_type : str = "zarr", cache : bool = True, cache_path = None) -> Tuple[dask.array.core.Array, dask.distributed.Client]:
    """Initialize a dask array for lazy reading and rechunking it for easier use
    Args:
        - location: path to folder of tiff slices or path to zarr directory
        - chunk_shape: sliding window size, zyx
        - image_type: filetype of the data to read, should be zarr, tif, or tiff
    Returns:
        - image: dask image for lazy reading in read_patch
        - client: dask client
    """
    if image_type == "zarr" or location.endswith(".zarr"):
        zarr_location = Path(location) # removes extra "/" at the end
        zarr_location = (zarr_location.parent / (zarr_location.name + ".zarr")) if not zarr_location.name.endswith(".zarr") else zarr_location
        image = da.from_zarr(location)
        
    else:
        if cache:
            if cache_path and cache_path.end:
                cache_path = Path(cache_path)
            else:
                cache_path = Path(location).parent / (Path(location).name + ".zarr")
                
            if cache_path.exists():
                image = da.from_zarr(str(cache_path))
            else:
                images_path = Path(location, f"*.{image_type}")
                im = imread_dask(str(images_path))
                im = im.rechunk(chunk_shape)
                print("Caching as zarr...")
                with ProgressBar():
                    da.to_zarr(im, str(cache_path))
                print("Done!")
                image = da.from_zarr(str(cache_path))
        else:   
            images_path = Path(location, f"*.{image_type}")
            image = imread_dask(str(images_path))
  
    client = Client()
    return image, client

def sliding_window_inference_zarr(
    input_path: str,
    roi_size: Union[Sequence[int], int],
    sw_batch_size: int,
    predictor: Callable[..., torch.Tensor],
    overlap: float = 0.25,
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    sigma_scale: Union[Sequence[float], float] = 0.125,
    padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: Union[torch.device, str, None] = None,
    device: Union[torch.device, str, None] = None,
    window_data_threshold: int = 0,
    normalize_min: int = 0,
    normalize_max: int = -1,
    start_slice: int = 0,
    end_slice: int = -1,
    SIGMOID : bool = False,
    output_image: zarr.core.Array = None,
    count_map: zarr.core.Array = None,
    tta: bool = None,
    flip_dim: int = None,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Sliding window inference on `inputs` with `predictor`.

    When roi_size is larger than the inputs' spatial size, the input image are padded during inference.
    To maintain the same spatial sizes, the output image will be cropped to the original input size.

    Args:
        inputs: input image to be processed (assuming NCHW[D])
        roi_size: the spatial window size for inferences.
            When its components have None or non-positives, the corresponding inputs dimension will be used.
            if the components of the `roi_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        sw_batch_size: the batch size to run window slices.
        predictor: given input tensor `patch_data` in shape NCHW[D], `predictor(patch_data)`
            should return a prediction with the same spatial shape and batch_size, i.e. NMHW[D];
            where HW[D] represents the patch spatial size, M is the number of output channels, N is `sw_batch_size`.
        overlap: Amount of overlap between scans.
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        sigma_scale: the standard deviation coefficient of the Gaussian window when `mode` is ``"gaussian"``.
            Default: 0.125. Actual window sigma is ``sigma_scale`` * ``dim_size``.
            When sigma_scale is a sequence of floats, the values denote sigma_scale at the corresponding
            spatial dimensions.
        padding_mode: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}
            Padding mode for ``inputs``, when ``roi_size`` is larger than inputs. Defaults to ``"constant"``
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        cval: fill value for 'constant' padding mode. Default: 0
        sw_device: device for the window data.
            By default the device (and accordingly the memory) of the `inputs` is used.
            Normally `sw_device` should be consistent with the device where `predictor` is defined.
        device: device for the stitched output prediction.
            By default the device (and accordingly the memory) of the `inputs` is used. If for example
            set to device=torch.device('cpu') the gpu memory consumption is less and independent of the
            `inputs` and `roi_size`. Output is on the `device`.
        window_data_threshold: Threshold value for skipping the inference, 0 in case masking was used
        normalize_min: Minimum value for minmax normalization.
        normalize_max: Maximum value for minmax normalization.
        args: optional args to be passed to ``predictor``.
        kwargs: optional keyword args to be passed to ``predictor``.

    Note:
        - input must be channel-first and have a batch dim, supports N-D sliding window.

    """
    print(f"Inside kwargs: {kwargs.keys()}")
    image, client = initialize_dask(input_path, roi_size, os.listdir(input_path)[10].split(".")[-1])
    print(f"Image shape {image.shape}")
    print(f"Dask client {client}")

    img_shape = [1, 1]
    img_shape.extend(image.shape)

    #TODO Always 3?
    num_spatial_dims = len(img_shape) - 2

    if overlap < 0 or overlap >= 1:
        raise AssertionError("overlap must be >= 0 and < 1.")

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    image_size = img_shape[2:]
    batch_size = img_shape[0]
    roi_size = fall_back_tuple(roi_size, image_size)
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size[i], roi_size[i]) for i in range(num_spatial_dims))

   
    pad_size = []
    for k in range(len(img_shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - img_shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
 
    np_pad = []
    for i, j in enumerate(img_shape):
         np_pad.append((pad_size[i], pad_size[i]))
    
    np_pad = tuple(np_pad)

    
    #with (64,64,32) roi_size this works out to (32,32,16)
    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows
    print(f"Total slices {total_slices}")
    
    # Create window-level importance map
    importance_map = compute_importance_map(get_valid_patch_size(image_size, roi_size), mode='constant', sigma_scale=sigma_scale)
    importance_map = importance_map.to(torch.float16).to(device).numpy()

    
            
    #calculate the amount of slices in this batch (for "inferring... x / 142" progress report)
    slice_l = len(list(range(0, total_slices, sw_batch_size)))
    

    #TODO:
    # Start and end point for inference, this way this can be distributed to multiple nodes
    #split total slices into number of sw_batches (slice_i) and number of slices within the total_slices (slice_g), 
    #then run inference on each of those slices 
    if end_slice < 0:
        end_slice = total_slices


    # Perform predictions
    print(f"Inferring from {start_slice} to {end_slice} (sw_batch_size {sw_batch_size}); in total {(end_slice - start_slice)/sw_batch_size}...")
    for slice_i, slice_g in enumerate(range(start_slice, end_slice, sw_batch_size)):
        print(f"{slice_g}/{total_slices}",end="\r",flush=True)
        slice_start_time = datetime.datetime.now()
        #print("slice_g",slice_g)
    
        #create the range of slice numbers within current sw_batch
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))

        unravel_slice = []
        for idx in slice_range:
            try:
                slice_block = [[list([each_slice.start , each_slice.stop]) for each_slice in slices[idx % num_win]]]
                unravel_slice += slice_block
            except:
                print("skipped window: ",idx, " of ",num_win," windows")
                pass
                #this try-except block fixes an off-by-one error in the slice generation that leads to output shifting
        

        #load the data as subset from the main input dataset (that should be a tensor by now) and concatenate all sw_batch slices into one tensor
        #load data 
        data_to_load = []
        # print("started gathering data_to_load...")
        for win_id, win_slice in enumerate(unravel_slice):
            #TODO Add bounding box here:
            slice_offset = [win_slice[0][0], win_slice[1][0], win_slice[2][0]]
            slice_size = [win_slice[0][1] - slice_offset[0],
                          win_slice[1][1] - slice_offset[1],
                          win_slice[2][1] - slice_offset[2]]
            
            single_slice_load = read_patch_dask(image, slice_size, slice_offset)
            
            data_to_load.append(single_slice_load)
        # concatenate all win_slices to a single batch (influenced by sw_batch_size, set according to on the graphics card VRAM)
        
        # print("started loading data_to_load...")
        window_data = da.stack(data_to_load).compute().astype(np.int32) # offloading tasks from dask
        window_data = np.expand_dims(window_data, axis=1) # add channel dimension      
        window_data = torch.IntTensor(window_data)
        
        #skip computing if this tile is background (as filtered and set to 0 by the mask_detection step)
        if window_data.max() <= window_data_threshold:
            seg_prob=torch.zeros_like(window_data)
            #cast the results back to float16 
            seg_prob = seg_prob.to(torch.float16).to(device).numpy()
            #print('Skipped orediction')
            #print(window_data.min(), window_data.max())
            seg_prob = seg_prob[:, 0]
            #print(seg_prob.shape)
        #if this tile contains data, process it: 
        else: 
            #cast as 32-bit float and send to graphics card 
            window_data = window_data.type(torch.float32)     
            if normalize_max > 0:
                window_data = torch.clamp(window_data, min= normalize_min, max = normalize_max)
                window_data = (window_data - normalize_min) / (normalize_max-normalize_min)    
            window_data = window_data.cuda()
            window_data.to(sw_device)
            #print(window_data.min(), window_data.max())
            #add noise if running with test-time augmentation
            if tta:
                #window_data = RandGaussianNoise(prob=1.0, std=0.001)(window_data)
                #window_data[:,0,:,:,:] = window_data[:,0,:,:,:] + (0.00001**0.5)*torch.randn(size=window_data[:,0,:,:,:].shape,out=window_data[:,0,:,:,:],dtype=torch.float32,device=sw_device)
                window_data = RandGaussianNoise(prob=1.0, mean=0.0, std=0.001)(window_data)

            # if the data needs to be flipped, do this here (flip_dim: 2 = z, 3 = y, 4 = x) 
            if flip_dim is not None:
                window_data = torch.flip(window_data,dims=[flip_dim])

            #run the actual prediction 
            #seg_prob = predictor(window_data, *args, **kwargs).to(device)  # batched patch segmentation
            #TODO Sort out this kwargs mess - state all arguments unless they need to be passed down!!!
            seg_prob = predictor(window_data)#, *args, **kwargs)  # batched patch segmentation
            
            # flip data back if previously flipped 
            if flip_dim is not None:
                seg_prob = torch.flip(seg_prob,dims=[flip_dim])
            
            #cast the results back to float16 
            seg_prob = seg_prob.to(torch.float16).to(device).numpy()
            #print('Using model to predict')
            seg_prob = seg_prob[:, 1]
            #print(seg_prob.shape)
        #print('Inference range:', (seg_prob.min(), seg_prob.max()))
        slice_delta = datetime.datetime.now() - slice_start_time 
        print('slice_delta:', slice_delta) 
        print(f"Inferred: {slice_g}/{total_slices} [{(slice_g / total_slices)*100:.2f} %]",end="\r",flush=True)

        # store the result in the proper location of the full output. Apply weights from importance map. (skip this if it is all background) 
        for idx, original_idx in zip(slice_range, unravel_slice):
            output_image[0,0,original_idx[0][0]:original_idx[0][1],original_idx[1][0]:original_idx[1][1],original_idx[2][0]:original_idx[2][1]] += importance_map * seg_prob[idx - slice_g]
            count_map[0,0,original_idx[0][0]:original_idx[0][1],original_idx[1][0]:original_idx[1][1],original_idx[2][0]:original_idx[2][1]] += importance_map  
        window_data = 0
        seg_prob = 0

    print(f"{datetime.datetime.now()} : Inference run finished")

    
class SlidingWindowInferer(Inferer):
    """
    Sliding window method for model inference,
    with `sw_batch_size` windows for every model.forward().
    Usage example can be found in the :py:class:`monai.inferers.Inferer` base class.

    Args:
    roi_size: the window size to execute SlidingWindow evaluation.
        If it has non-positive components, the corresponding `inputs` size will be used.
        if the components of the `roi_size` are non-positive values, the transform will use the
        corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
        to `(32, 64)` if the second spatial dimension size of img is `64`.
    sw_batch_size: the batch size to run window slices.
    overlap: Amount of overlap between scans.
    mode: {``"constant"``, ``"gaussian"``}
        How to blend output of overlapping windows. Defaults to ``"constant"``.

        - ``"constant``": gives equal weight to all predictions.
        - ``"gaussian``": gives less weight to predictions on edges of windows.

    sigma_scale: the standard deviation coefficient of the Gaussian window when `mode` is ``"gaussian"``.
        Default: 0.125. Actual window sigma is ``sigma_scale`` * ``dim_size``.
        When sigma_scale is a sequence of floats, the values denote sigma_scale at the corresponding
        spatial dimensions.
    padding_mode: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}
        Padding mode when ``roi_size`` is larger than inputs. Defaults to ``"constant"``
        See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
    cval: fill value for 'constant' padding mode. Default: 0
    sw_device: device for the window data.
        By default the device (and accordingly the memory) of the `inputs` is used.
        Normally `sw_device` should be consistent with the device where `predictor` is defined.
    device: device for the stitched output prediction.
        By default the device (and accordingly the memory) of the `inputs` is used. If for example
        set to device=torch.device('cpu') the gpu memory consumption is less and independent of the
        `inputs` and `roi_size`. Output is on the `device`.
    threshold_inference: threshold for which inference starts, everything < threshold is skipped.
    normalize_min: Minimum value for minmax normalization.
    normalize_max: Maximum value for minmax normalization.
    start_slice: Parallelization feature, indicates starting slice for distributed computing
    end_slice: Ending slice for distributed computing

    Note:
        ``sw_batch_size`` denotes the max number of windows per network inference iteration,
        not the batch size of inputs.

    """

    def __init__(
        self,
        roi_size: Union[Sequence[int], int],
        sw_batch_size: int = 1,
        overlap: float = 0.25,
        mode: Union[BlendMode, str] = BlendMode.CONSTANT,
        sigma_scale: Union[Sequence[float], float] = 0.125,
        padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
        cval: float = 0.0,
        sw_device: Union[torch.device, str, None] = None,
        device: Union[torch.device, str, None] = None,
        threshold_inference: int = 0,
        normalize_min: int = 0,
        normalize_max: int = -1,
        start_slice: int = 0,
        end_slice: int = -1
    ) -> None:
        Inferer.__init__(self)
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.mode: BlendMode = BlendMode(mode)
        self.sigma_scale = sigma_scale
        self.padding_mode = padding_mode
        self.cval = cval
        self.sw_device = sw_device
        self.device = device
        self.threshold_inference = threshold_inference
        self.normalize_min = normalize_min
        self.normalize_max = normalize_max

    def __call__(
        self, input_path: str, network: Callable[..., torch.Tensor], *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """

        Args:
            inputs: model input data for inference.
            network: target model to execute inference.
                supports callables such as ``lambda x: my_torch_model(x, additional_config)``
            args: optional args to be passed to ``network``.
            kwargs: optional keyword args to be passed to ``network``.

        """
        print(f"Calling SWI with {kwargs.keys()}")
        print(f"Output image {type(kwargs['output_image'])} {kwargs['output_image'].shape}")

        return sliding_window_inference_zarr(
            input_path,
            self.roi_size,
            self.sw_batch_size,
            network,
            self.overlap,
            self.mode,
            self.sigma_scale,
            self.padding_mode,
            self.cval,
            self.sw_device,
            self.device,
            self.threshold_inference,
            self.normalize_min,
            self.normalize_max,
            *args,
            **kwargs,
        )