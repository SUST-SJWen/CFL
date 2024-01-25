## Title
Pixel-level semi-supervision: Co-frequency learning for labeled and unlabeled data for semi-supervised medical image segmentation
## Requirements
* [Pytorch]
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy, Medpy ......

## Usage

1. In this project, we use the ACDC dataset for training demonstration, and the training process for the AbdomenCT-1K and ISIC2018 datasets is similar to that of ACDC. Download the pre-processed data and put the data in `../data/ACDC`. You can download the ACDC dataset with the list of labeled training, unlabeled training, validation, and testing slices as following:
ACDC from [Google Drive Link](https://drive.google.com/file/d/1F3JzBSIURtFJkfcExBcT6Hu7Ar5_f8uv/view?usp=sharing), or [Baidu Netdisk Link](https://pan.baidu.com/s/1LS6VHujD8kvuQikbydOibQ) with passcode: 'kafc'.

2. Train the model

```
cd code
```

You can choose model(unet/vnet/pnet...), dataset(ACDC/AbdomenCT-1K/ISIC2018), experiment name(the path of saving your model weights and inference), iteration number, batch size and etc in your command line, or leave it with default option.

CFL
```
python train.py --root_path ../data/ACDC --exp ACDC/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 4 --labeled_num XXX
```

3. Test the model
```
python test.py -root_path ../data/ACDC --exp ACDC/XXX -model XXX --num_classes 4 --labeled_num XXX
```
Check trained model and inference
```
cd model
```


## Acknowledgement

This code is mainly based on [SSL4MIS](https://github.com/HiLab-git/SSL4MIS).

Some of the other code is from [SegFormer](https://github.com/NVlabs/SegFormer), [SwinUNet](https://github.com/HuCaoFighting/Swin-Unet), [Segmentation Models](https://github.com/qubvel/segmentation_models.pytorch), [UAMT](https://github.com/yulequan/UA-MT), [nnUNet](https://github.com/MIC-DKFZ/nnUNet).