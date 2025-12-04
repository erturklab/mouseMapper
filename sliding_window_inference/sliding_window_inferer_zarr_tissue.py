#Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
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
from dask.distributed import Client, progress, LocalCluster
from pathlib import Path
import matplotlib.pyplot as plt


dask.config.set({"optimization.fuse.active": False})

__all__ = ["sliding_window_inference_zarr"]

#  read_patch_dask to handle single multi-channel Dask array ---
def read_patch_dask(image: da.Array, shape: List[int], location: List[int]) -> np.array:
    """Read a single patch from a dask image (potentially multi-channel).
    Args:
        - image: The dask image array (expected shape: (C, D, H, W) or (D, H, W)).
        - shape: Spatial shape of the resulting patch [D, H, W].
        - location: [z, y, x] coordinates at (0,0,0) for the patch.
    """
    # Assuming Zarr is loaded as (C, D, H, W) or (D, H, W)
    # The [:, ...] slice will select all channels if the channel dim exists,
    # or effectively do nothing if it's a 3D array (D, H, W).
    # dask will automatically add a channel dimension if needed later by stack.
    patch = image[...,
                  location[0]:location[0] + shape[0],
                  location[1]:location[1] + shape[1],
                  location[2]:location[2] + shape[2]]
    return patch

# --- MODIFIED: initialize_dask to return a single Dask array and its channel count ---
def initialize_dask(location: str, chunk_shape: List[int], image_type: str = "zarr", cache: bool = True, cache_path=None) -> Tuple[dask.array.core.Array, int, dask.distributed.Client]:
    """Initialize a dask array for lazy reading and rechunking it for easier use.
    Handles single 2-channel Zarr input.
    Args:
        - location: path to the Zarr directory.
        - chunk_shape: sliding window size, zyx (spatial dims only).
        - image_type: filetype of the data to read, should be "zarr". Other types removed as per new requirement.
    Returns:
        - image: dask image for lazy reading.
        - num_input_channels_actual: The number of channels detected in the Zarr file.
        - client: dask client.
    """
    client = Client()

    if not location.endswith(".zarr"):
        raise ValueError("Expected a .zarr input file path for multi-channel Zarr.")

    print(f'Reading multi-channel Zarr from {location}...')
    image = da.from_zarr(location)

    # Determine number of input channels based on Zarr structure
    # Assumption: If image.ndim == 4, it's (C, D, H, W). If image.ndim == 3, it's (D, H, W) (single channel).
    if image.ndim == 4: # Assuming (C, D, H, W)
        num_input_channels_actual = image.shape[0]
        # The image will be treated as (C, D, H, W) in read_patch_dask
        # No rechunking here as it's a single Zarr.
        spatial_shape_for_padding = image.shape[1:] # D, H, W
    elif image.ndim == 3: # Assuming (D, H, W) for a single channel
        num_input_channels_actual = 1
        image = da.expand_dims(image, axis=0) # Add a channel dimension for consistency (1, D, H, W)
        spatial_shape_for_padding = image.shape[1:] # D, H, W
    else:
        raise ValueError(f"Unsupported Zarr shape: {image.shape}. Expected 3D (D,H,W) or 4D (C,D,H,W).")

    print(f"Initialized Dask image with shape: {image.shape} (C,D,H,W), detected channels: {num_input_channels_actual}")
    
    return image, num_input_channels_actual, client



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


def sliding_window_inference_zarr(
    input_path: str, # Modified to accept single string path for Zarr
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
    window_data_threshold: int = 1000,
    normalize_min: int = 0,
    normalize_max: int = -1,
    SOFTMAX: bool = False,
    num_classes: int = 5, # Number of output classes
    output_image: zarr.array = None,
    count_map: zarr.array = None,
    tta: bool = None,
    flip_dim: int = None,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Sliding window inference on `inputs` with `predictor`.

    Args:
        input_path: Input image path (expected to be a 2-channel Zarr file).
        roi_size: the spatial window size for inferences.
        sw_batch_size: the batch size to run window slices.
        predictor: given input tensor `patch_data` in shape NCHW[D], `predictor(patch_data)`
            should return a prediction with the same spatial shape and batch_size, i.e. NMHW[D];
            where HW[D] represents the patch spatial size, M is the number of output channels, N is `sw_batch_size`.
        overlap: Amount of overlap between scans.
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.
        sigma_scale: the standard deviation coefficient of the Gaussian window when `mode` is ``"gaussian"``.
        padding_mode: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}
            Padding mode for ``inputs``, when ``roi_size`` is larger than inputs. Defaults to ``"constant"``
        cval: fill value for 'constant' padding mode. Default: 0
        sw_device: device for the window data.
        device: device for the stitched output prediction.
        window_data_threshold: Threshold value for skipping the inference, 0 in case masking was used
        normalize_min: Minimum value for minmax normalization.
        normalize_max: Maximum value for minmax normalization.
        SOFTMAX: If True, apply softmax to the model output. This is typically used for multi-class segmentation
                 when the model outputs logits.
        num_classes: The number of output classes for multi-class segmentation.
        args: optional args to be passed to ``predictor``.
        kwargs: optional keyword args to be passed to ``predictor``.

    Note:
        - input must be channel-first and have a batch dim, supports N-D sliding window.

    """
    start_slice = kwargs.get('start_slice', 0)
    end_slice = kwargs.get('end_slice', -1)
    normalization = kwargs.get('normalization', 'minmax')
    print(f"Inside kwargs: {kwargs.keys()}")

    # --- MODIFIED: Call initialize_dask for single Zarr input ---
    image_dask_array, num_input_channels_actual, client = initialize_dask(input_path, roi_size, "zarr")
    print(f"Loaded Dask image with shape: {image_dask_array.shape}, actual input channels: {num_input_channels_actual}")
    print(f"Dask client {client}")

    # img_shape now reflects (N=1, C_in, D, H, W) where C_in is num_input_channels_actual
    img_shape = [1, num_input_channels_actual]
    img_shape.extend(image_dask_array.shape[1:]) # Spatial dims from the Dask array (D, H, W)

    num_spatial_dims = len(img_shape) - 2 # N C D H W -> D H W (3 spatial dims)

    if overlap < 0 or overlap >= 1:
        raise AssertionError("overlap must be >= 0 and < 1.")

    image_size = img_shape[2:] # Spatial dimensions (D, H, W)
    batch_size = img_shape[0] # Should typically be 1 for single image inference

    roi_size = fall_back_tuple(roi_size, image_size)
    image_size = tuple(max(image_size[i], roi_size[i]) for i in range(num_spatial_dims))

    pad_size = []
    for k in range(len(img_shape) - 1, 1, -1): # Iterate spatial dimensions (from W to D)
        diff = max(roi_size[k - 2] - img_shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    
    np_pad = []
    for i in range(len(img_shape)):
        if i >= 2: # Spatial dimensions (D, H, W)
            idx = (i - 2) * 2
            np_pad.append((pad_size[idx], pad_size[idx+1]))
        else: # Non-spatial dimensions (N, C)
            np_pad.append((0, 0))
    np_pad = tuple(np_pad)

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)
    total_slices = num_win * batch_size
    print(f"Total slices {total_slices}")
    
    # Calculate importance map for a single patch size (D_patch, H_patch, W_patch)
    # The output of compute_importance_map will be (D_patch, H_patch, W_patch)
    spatial_patch_size_for_importance = get_valid_patch_size(image_size, roi_size)
    importance_map = compute_importance_map(spatial_patch_size_for_importance, mode=mode, sigma_scale=sigma_scale)
    
    # Correctly expand dimensions to (1, C_out, D_patch, H_patch, W_patch)
    # 1. Add a batch dimension at index 0 (becomes (1, D_patch, H_patch, W_patch))
    # 2. Add a channel dimension at index 1 (becomes (1, 1, D_patch, H_patch, W_patch))
    # 3. Repeat along the channel dimension to match num_classes
    importance_map = importance_map.unsqueeze(0).unsqueeze(1).repeat(1, num_classes, 1, 1, 1)
    
    importance_map = importance_map.to(torch.float16).to(device).numpy() # Convert to NumPy for Zarr writing

    slice_l = len(list(range(0, total_slices, sw_batch_size)))
    
    if end_slice < 0 or end_slice > total_slices:
        end_slice = total_slices

    print(f"Inferring from {start_slice} to {end_slice} (sw_batch_size {sw_batch_size}); in total {(end_slice - start_slice)/sw_batch_size:.2f} batches...")
    for slice_i, slice_g in enumerate(range(start_slice, end_slice, sw_batch_size)):
        print(f"{slice_g}/{total_slices}", end="\r", flush=True)
        slice_start_time = datetime.datetime.now()
        
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))

        unravel_slice = []
        for idx in slice_range:
            try:
                slice_block = [[list([each_slice.start , each_slice.stop]) for each_slice in slices[idx % num_win]]]
                unravel_slice += slice_block
            except IndexError:
                print(f"skipped window: {idx} of {num_win} windows (IndexError)")
                pass
                
        data_to_load = []
        for win_id, win_slice in enumerate(unravel_slice):
            slice_offset = [win_slice[0][0], win_slice[1][0], win_slice[2][0]]
            slice_size = [win_slice[0][1] - slice_offset[0],
                          win_slice[1][1] - slice_offset[1],
                          win_slice[2][1] - slice_offset[2]]
            
            #  Call read_patch_dask with the single Dask array ---
            # read_patch_dask now returns (C_in, D, H, W) for a single patch
            single_slice_load = read_patch_dask(image_dask_array, slice_size, slice_offset)
            data_to_load.append(single_slice_load)
        
        # Stack all sw_batch_size patches. Resulting shape: (B, C_in, D, H, W)
        window_data = da.stack(data_to_load).compute().astype(np.float32)
        window_data = torch.from_numpy(window_data)
        
        if window_data.max() <= window_data_threshold:
            # For multi-class, initialize with num_classes channels
            print("Skipping due to low values")
            seg_prob = torch.zeros((window_data.shape[0], num_classes, *window_data.shape[2:]), dtype=torch.float16, device=device).numpy()
        else:
            window_data = window_data.type(torch.float32)
            if normalize_max > 0:
                window_data = torch.clamp(window_data, min= normalize_min, max = normalize_max)
            
                if normalization == 'minmax':
                    #rint('Using min-max normalization, ', normalize_min , normalize_max)
                    window_data = (window_data - normalize_min) / (normalize_max-normalize_min)
                elif normalization == 'zscore':
                    #print('Using z-score normalization')
                    # ---  Calculate mean and std per channel ---
                    # window_data shape is (B, C, D, H, W)
                    # We want mean/std over D, H, W for each B and C
                    mean = torch.mean(window_data, dim=[2, 3, 4], keepdim=True)
                    std = torch.std(window_data, dim=[2, 3, 4], keepdim=True)
                    
                    # Ensure std is not zero to prevent division by zero, add epsilon
                    window_data = (window_data - mean) / (std + 1e-8) # Added 1e-8 to std directly

                    
            window_data = window_data.cuda()
            window_data.to(sw_device)

            if tta:
                window_data = RandGaussianNoise(prob=1.0, mean=0.0, std=0.001)(window_data)

            # Note: flip_dim now applies to (N, C, D, H, W) input.
            # D, H, W correspond to dims 2, 3, 4 respectively.
            if flip_dim is not None:
                window_data = torch.flip(window_data,dims=[flip_dim])

            seg_prob = predictor(window_data)
            
            if flip_dim is not None:
                seg_prob = torch.flip(seg_prob,dims=[flip_dim])
            
            if SOFTMAX:
                #print('Using Softmax')
                seg_prob = F.softmax(seg_prob, dim=1)

            seg_prob = seg_prob.to(torch.float16).to(device).numpy()

        slice_delta = datetime.datetime.now() - slice_start_time
        print(f"Inferred: {slice_g}/{total_slices} [{(slice_g / total_slices)*100:.2f} %] slice_delta: {slice_delta}", end="\r", flush=False)

        for idx, original_idx in zip(slice_range, unravel_slice):

            output_image[0,:,original_idx[0][0]:original_idx[0][1],original_idx[1][0]:original_idx[1][1],original_idx[2][0]:original_idx[2][1]] += importance_map[0] * seg_prob[idx - slice_g]
            count_map[0,:,original_idx[0][0]:original_idx[0][1],original_idx[1][0]:original_idx[1][1],original_idx[2][0]:original_idx[2][1]] += importance_map[0]

            try:
                lhs_slice_shape = output_image[0, :, 
                                               original_idx[0][0]:original_idx[0][1], 
                                               original_idx[1][0]:original_idx[1][1], 
                                               original_idx[2][0]:original_idx[2][1]
                                              ].shape

            except Exception as e:
                print(f"DEBUG: Failed to get LHS slice shape: {e}")
        window_data = 0
        seg_prob = 0

    print(f"\n{datetime.datetime.now()} : Inference run finished")

    
class SlidingWindowInferer(Inferer):
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
        threshold_inference: int = 1000,
        normalize_min: int = 0,
        normalize_max: int = -1,
        num_classes: int = 5,
        SOFTMAX: bool = False,
        start_slice: int = 0,
        end_slice: int = -1,
        normalization :str = 'minmax',  # 'minmax' or 'zscore'
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
        self.num_classes = num_classes
        self.SOFTMAX = SOFTMAX
        self.normalization = normalization

    def __call__(
        self, input_path: str, network: Callable[..., torch.Tensor], *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        print(f"Calling SWI with {kwargs.keys()}")
        print(f"Output image type: {type(kwargs['output_image'])}, shape: {kwargs['output_image'].shape}")

        return sliding_window_inference_zarr(
            input_path, # Single string path now
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
            SOFTMAX=self.SOFTMAX,
            num_classes=self.num_classes,
            normalization=self.normalization,
            *args,
            **kwargs,
        )