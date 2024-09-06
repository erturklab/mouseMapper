# Peripheral nerves segmentation
<img src="nerve_seg_example.png" width="500">

## Requirements
* Linux system with GPU (at least 10 GB GPU RAM) and CPU (at least 10 cores), and with 100 GB RAM.  
* Raw image data saved as a series of 16-bit TIFF files (.tif), one per z-plane. 
  
## Installation
* Install [CUDA](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn).
* Install [Anaconda](https://www.anaconda.com/download#downloads) to create and control virtual environments.
* Install Python 3.9 or higher version by Anaconda.
  ```
    conda create -n env python=3.9
	conda activate env
	```
* Install [pytorch](https://pytorch.org/get-started/locally/).
* Install additional required libraries:
     ```
     cd {dir_whole_repo}/Nerve_Module
     pip install -e .
	```
  `dir_whole_repo` refers to the directory where the entire repository is stored on your local device.
* Create folders `nnUNet_raw_data_base`, `nnUNet_preprocessed`, `RESULTS_FOLDER` under [this folder](../Nerve_Module/) for storing raw data, preprocessed data and trained models, and set the three folders as environment variables:
  ```
   export nnUNet_raw_data_base="{dir_whole_repo}/Nerve_Module/nnUNet_raw_data_base"
   export nnUNet_preprocessed="{dir_whole_repo}/Nerve_Module/nnUNet_preprocessed"
   export RESULTS_FOLDER="{dir_whole_repo}/Nerve_Module/RESULTS_FOLDER"
	```  
  
## Nerve segmentation
* Download trained models and the plan file from [here](../models/nerve_segmentation), then put them in the created folder `RESULTS_FOLDER`.
* Download the [example data](../Nerve_Module/example_data) for testing, which is part of one trigeminal nerve. The example data includes two folders: one containing raw tiff slices named `Image`, and a folder of annotation tiff slices named `Mask`.
* Compute the `Intensity_low` and `Intensity_high` for percentile normalization during the next inference step:
  ```
  cd {dir_whole_repo}/Nerve_Module/nnunet/utilities/
  python stat_intensity_range_nozero.py -i RAW_TIFF_FOLDER -p 0.1
	``` 
  `RAW_TIFF_FOLDER` refers to the folder containing raw tiff slices.
* Run nerve segmentation inference for a whole-body scan or part of the scan, saved as a series of 16-bit TIFF files (.tif):
  ```
  cd {dir_whole_repo}/Nerve_Module/nnunet/inference/
  CUDA_VISIBLE_DEVICES=0 python predict_simple_dask.py -i RAW_TIFF_FOLDER -o OUTPUT_NERVE_SEGMENTATION_FOLDER -min Intensity_low -max Intensity_low -mdl {dir_whole_repo}/Nerve_Module/RESULTS_FOLDER/model_final_checkpoint.model -pl {dir_whole_repo}/Nerve_Module/RESULTS_FOLDER/plans.pkl -bs 1 -overlap 0.75
	```  
  The nerve segmentation results will be saved as zarr files under the selected `OUTPUT_NERVE_SEGMENTATION_FOLDER`.
  For the [example data](../Nerve_Module/example_data) whose size is 960×776*504, the inference time is around half an hour. 
* Convert infered segmentation results to tiff slices:
  ```
  cd {dir_whole_repo}/Nerve_Module/nnunet/utilities/
  python zarr2tif.py -i RAW_TIFF_FOLDER -o OUTPUT_NERVE_SEGMENTATION_FOLDER -pl {dir_whole_repo}/Nerve_Module/RESULTS_FOLDER/plans.pkl
	```  
  The nerve segmentation results will be converted to tiff slices in a folder named `binary` under the selected `OUTPUT_NERVE_SEGMENTATION_FOLDER`.
## Nerve quantification
* Get the [organ segmentation](../Tissue_Module/Organ_Segmentation.ipynb) and [tissue segmentation](../Tissue_Module/Tissue_segmentation.ipynb) for each mouse scan with a downsampling factor of 5 in both xy and z directions.
* Dilated the organ mask for head and the rest organs:
  ```
  cd {dir_whole_repo}/Nerve_Module/quantification/
  python grow_organ_mask.py -i_mask PATH_ORGAN_SEGMENTATION -o_mask OUT_DILATEDMASK_FOLDER
	``` 
  `PATH_ORGAN_SEGMENTATION` is the path for the organ mask NIfTI file (.nii.gz). `OUT_DILATEDMASK_FOLDER` is the path for the resulted dilated organ mask NIfTI file (.nii.gz).
* Upsample the dilated head mask and save it as tiff slices to a folder `HEAD_MASK_FOLDER`. The mask of the rest organs is upsampled and saved as tiff slices to a folder `ORGAN_MASK_FOLDER`, and the tissue mask to a folder `TISSUE_MASK_FOLDER` following [instructions](../Tissue_Module/Organ_Segmentation.ipynb).
* Quantify nerve voxels/density in every organ/tissue:
  ```
  python quantification_nerve.py -nerve_mask OUTPUT_NERVE_SEGMENTATION_FOLDER -head_mask HEAD_MASK_FOLDER -organ_mask ORGAN_MASK_FOLDER -tissue_mask TISSUE_MASK_FOLDER -o_dir QUANTIFICATION_OUTPUTS_FOLDER
	```  
  The quantification results will be saved in csv files under the selected `QUANTIFICATION_OUTPUTS_FOLDER`.
