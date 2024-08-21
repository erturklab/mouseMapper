# Internal Organ Segmentation segmentation

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
* Install additional required libraries:
     ```
     cd ./nerve_segmentation
     pip install -e .
	```
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
* Postprocess the predictions
* Upsample the resulting masks
* Optional : create masked non-organ slices for the downstream tissue segmentation
  
