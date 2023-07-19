# ConvFormer
This repo is the official implementation for:\
[MICCAI2023] ConvFormer: Plug-and-Play CNN-Style Transformers for Improving Medical Image Segmentation.\
(The details of our ConvFormer can be found at the models directory in this repo or in the paper. We take SETR for example.)

## Requirements
* python 3.6
* pytorch 1.8.0
* torchvision 0.9.0
* more details please see the requirements.txt

## Datasets
* The ACDC dataset could be acquired from [here](https://www.creatis.insa-lyon.fr/Challenge/acdc/). The slice-level ACDC dataset preprocessed by us can be downloaded from [here](https://drive.google.com/file/d/18W_d8ho0Tl7TgPQXczOXZK5OUxtYkQdc/view?usp=share_link).
* The ISIC 2018 dataset could be acquired from [here](https://challenge.isic-archive.com/data/).

## Training
Commands for training
```
python train.py
```
## Testing
Commands for testing
``` 
python test.py
```
