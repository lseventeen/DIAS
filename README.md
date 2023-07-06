# PHTrans: Parallelly Aggregating Global and Local Representations for Medical Image Segmentation
This repository is the official PyTorch code for the paper [PHTrans: Parallelly Aggregating Global and Local Representations for Medical Image Segmentation](https://link.springer.com/chapter/10.1007/978-3-031-16443-9_23).
<!-- This repository is the official PyTorch code for the paper '[PHTrans: Parallelly Aggregating Global and Local Representations for Medical Image Segmentation](https://arxiv.org/abs/2203.04568)' (Wentao Liu, Tong Tian, Weijin Xu, Huihua Yang, and Xipeng Pan) -->

<div align="center">
  <img src="PHTrans/PHTrans.png" width="70%">
</div>

## News
2022.9.24: The solution of Two-stage PHTrans combined with self-training won Winner Finalist Award at [MICCAI FLARE2022](https://flare22.grand-challenge.org/awards/) ([CODE](https://github.com/lseventeen/FLARE22-TwoStagePHTrans) & [PAPER](https://arxiv.org/abs/2207.11512)).

## Prerequisites
 

 
Download our repo:
```
git clone https://github.com/lseventeen/PHTrans.git
cd PHTrans
```
Install packages
```
cd nnUNet
pip install -e .

cd PHTrans
pip install -e .
```
 
## Datasets processing
Download datasets [BCV](https://www.dropbox.com/sh/z4hbbzqai0ilqht/AAARqnQhjq3wQcSVFNR__6xNa?dl=0https://www.dropbox.com/sh/z4hbbzqai0ilqht/AAARqnQhjq3wQcSVFNR__6xNa?dl=0) and [ACDC](https://acdc.creatis.insa-lyon.fr/description/databases.html). Type this in the terminal to perform dataset partitioning followed by [nnFormer](https://github.com/282857341/nnFormer). 
 
```
PHTrans_BCV -dataset_path DATASET_PATH
PHTrans_ACDC -dataset_path DATASET_PATH
```
Preprocess the BCV and ACDC datasets according to the uploaded nnUNet package
## Training
Type this in terminal to run train
 
```
PHTrans_train -task 17OR27 --fold 0
```
## Test
Type this in terminal to test:
 
```
PHTrans_train -task 17OR27 -eei EXPERIMENT_ID -val
```
To replicate the results in the paper, we have prepared the [download link](https://drive.google.com/drive/folders/1lzj8SJgwGQG-lP9D-pzB1y6RNa6pzVsv?usp=sharing) of pre-trained models.

## Acknowledgements


The 3D Swin Transformer block of PHTrans refers to the source code repository of [Swin Transformer](https://github.com/microsoft/Swin-Transformer) and part of codes are reused from the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet). Thanks to Liu Ze and Fabian Isensee for the open source code.

 
