# MouseMapper: Obesity-induced changes in peripheral nerves and immune cell dynamics on the whole body scale

![GA](./HFD_GA%20(2).png)

Official repository for [Deep Learning and 3D Imaging Reveal Whole-Body Alterations in Obesity](https://doi.org/10.1101/2024.08.18.608300)


Many diseases, such as obesity, have systemic effects that impact multiple organ systems throughout the body. However, tools for comprehensive, high-resolution analysis of disease-associated changes at the whole-body scale have been lacking. Here, we developed a suite of deep learning-based image analysis algorithms (MouseMapper) and integrated it with tissue clearing and light-sheet microscopy to enable a comprehensive analysis of diseases impacting diverse systems across the mouse body. This approach enables the quantitative analysis of cellular and structural changes across the entire mouse body at unprecedented resolution and scale, including tracking nerves over several centimeters in whole animal bodies. To demonstrate its power, we applied MouseMapper to study nervous and immune systems in high-fat diet induced obesity. We uncovered widespread changes in both immune cell distribution and nerve structures, including alterations in the trigeminal nerve characterized by a reduced number of nerve endings in obese mice. These structural abnormalities were associated with functional deficits of whisker sensing and proteomic changes in the trigeminal ganglion, primarily affecting pathways related to axon growth and the complement system. Additionally, we found heterogeneity in obesity-induced whole-body inflammation across different tissues and organs. Our study demonstrates MouseMapper's capability to discover and quantify pathological alterations at the whole-body level, offering a powerful approach for investigating the systemic impacts of various diseases.

[![YouTube](http://i.ytimg.com/vi/yjxQJOu3trE/hqdefault.jpg)](https://www.youtube.com/watch?v=yjxQJOu3trE)

Please explore the image data yourself:
* [Whole Body UCHL1 High fat diet mouse](https://wilddisco.translatum.xyz/datasets/erturk_lab/UCHL1_HFD_788/view#3718,7491,1848,0,1.3)
* [Whole Body CD68 CHOW mouse](https://wilddisco.translatum.xyz/datasets/erturk_lab/CD68_chow_7790/view#2336,4181,2299,0,17.043)

## Modules

* [Nerve Module](./nerve_segmentation/) - AI-driven module developed for the segmentation and quantification of the peripheral nervous system
* [Immune Module](./Inflammation_Module/) - AI-driven module for the detection, localization and quantification of immune markers
* [Tissue Module](./Tissue_Module/) - Ensemble of AI models for the segmentation of 20 internal organs, along with tissue types such as muscle, fat, bone, and bone marrow
* [Neuron Module](./Graph_Module/) - Graph extraction for whole body nerve segmentations


## Prerequisites 

For running the analysis presented in this codebase, we recommend a workstation equipped with Linux and a GPU of at least 24 GB VRAM, and at least 64 GB RAM (depending on the size of the data to be analysed), and minimum 6 CPU cores. The code presented here is supported for Linux. System has been tested on Ubuntu 20.04 and Linux Rocky 9.4.

The organ, tissue, and immune module require the prior installation of nnUNetv2. You can find that codebase here: [nnUNetv2](https://github.com/MIC-DKFZ/nnUNet/tree/master).
The nerve module uses nnUNetv1, and you can find all necessary code [here](./nerve_segmentation/).

Graph extraction requires the installation of [pi2/itl2](https://github.com/arttumiettinen/pi2). After installation, please change the config file to the corresponding pi2 location.

## Usage & Installation

As our solution is highly modular, we invite the users to follow the instructions described within each module. Installation for each module can take around 30 minutes - 1 hour, with most time being taken up by the pytorch installations.

In general, the code in this repository is designed to have 16 bit tiff zstacks as input and output.



