# Tissue (Internal Organs + Soft tissue) Module

![Example](./assets/VR_fig.png)

## Requirements
* Linux system with GPU (at least 24 GB GPU RAM) and CPU (at least 10 cores), and with 100 GB RAM.  
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
Install [nnUNETv2](https://github.com/MIC-DKFZ/nnUNet/tree/master) following the instructions on their repository.
* Install additional required libraries:
    opencv-python==4.5.5.64
    pylibtiff==0.5.1
    connected-components-3d==3.10.5 
    nibabel==3.2.2
* Create folders `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results`.
   
  
## Organ segmentation
* Download trained model zip from [here](TODO), and select model 6802 . For setting it up, follow the instructions form [here] (https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md#how-to-deploy-and-run-inference-with-your-pretrained-models)
* Run organ downsampling. You can find the necessary code and an exampl in [Organ_Segmentation.ipynp](./Organ_Segmentation.ipynb)
* Run organ segmentation inference for a whole-body scan:
  ```
  nnUNetv2_predict -d 68802 -i path_ouput_preprocessing -o folder_out_pred -c 3d_fullres -tr nnUNetTrainer 
	```  
* Prediction on a downsampled mouse is expected to take up to 10 minutes. Postprocess the predictions
* Upsample the resulting masks
* Optional : create masked non-organ slices for the downstream tissue segmentation

![Example](./assets/organ_map_1.png)
  

## Tissue segmentation
* Download trained model zip from [here](TODO), and select model 310 . For setting it up, follow the instructions form [here] (https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md#how-to-deploy-and-run-inference-with-your-pretrained-models)
* Crop your tissue data with your solution of choice. The smaller the crops, the faster each patch will be segmented, and the less resources (RAM and CPU) you can get away with.
* Run organ segmentation inference for a whole-body scan:
  ```
  nnUNetv2_predict -d 310 -i path_ouput_preprocessing -o folder_out_pred -c 3d_fullres -tr nnUNetTrainer 
	```  
* Prediction on a 500x500x500 sized patch is expected to take around 13 minutes. Merge the resulting patches back to the original resolution

![Example](./assets/mouse_map.png)

![Example](./assets/pipeline.png)

