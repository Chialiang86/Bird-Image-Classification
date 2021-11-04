---
tags: CVDL
---

# CodaLab Competetion : Bird Images Classification

> contest link : https://competitions.codalab.org/competitions/35668?secret_key=09789b13-35ec-4928-ac0f-6c86631dda07

## Environment
- Ubuntu 20.04.1
- python 3.8.10

## Requirements
* numpy==1.21.3
* torch==1.10.0
* torchvision==0.11.1
* Pillow==8.4.0
* matplotlib==3.4.3
* efficientnet-pytorch==0.7.1
* scipy==1.7.1
* ml_collections==0.1.0

### Download requirements
```shell 
$ pip install -r requirements.txt
```

## Preprocessing Approach
#### 1.  All the 6,033 given images are cropped to form a new data set
For example :
- before cropping

![](https://i.imgur.com/wbKZqYX.jpg)

- after cropping 

![](https://i.imgur.com/b8BgRxJ.jpg)

Run the image cropper tool 
```python
$ python crop.py
```
1. First, click the left-top position of the bird patch, and then click the right-bottom position of the bird patch
2. Press space key to select the next image until all the images are cropped

#### 2. Use torchvision.transforms to do basic augmentation
```python 
    transform_train = T.Compose([
        T.Resize([args.img_size, args.img_size]),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=(-10, 10), translate=(0, 0)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
```

#### 3. Double the number of images of the given dataset
Each image will generate two images through transform methods

## Training

#### Download pretrained weights for transFG
- GitHub repo : [link](https://github.com/TACJu/TransFG)
- ViT-B_16 pretrained weight : [link](https://drive.google.com/file/d/1GOnXkRCrAQgctRJI4eIsl5no_1yg1GQF/view?usp=sharing)
- ViT-B_32 pretrained weight : [link](https://drive.google.com/file/d/1r_zZ5awqyHadAxTqlslyztkzA1vchDg-/view?usp=sharing)
#### Command line options
- img_size : image size of the given image, ex 320 -> (320, 320)
- aug_num : magnification of the original number of dataset (3000 for training set),default 2
- batch_size : default 6
- epoch : default 100
- model : [resnet18, resnet50, efficientnet-b0, efficientnet-b1, efficientnet-b3, efficientnet-b4, efficientnet-b5, transFG], default transFG (fine-grained approach)
- fg_type : type of fine-grained, [ViT-B_16, ViT-B_32], default ViT-B_16
- patience : early stop threshold number of the val_loss, default 2
- lr : learning rate, default 1e-5
- weight_decay : default 0.01

#### Example
```python 
$ python train.py --img_size [default=320] --aug_num [default=2] --batch_size [default=16] --epoch [default=100] --model [default=transFG] --fg_type [default=ViT-B_16] --patience [default=2] --lr [default=0.00005] --weight_decay [default=0.01]
```

## Generate answer.txt

```python 
$ python submit.py --model_path [default='result/transFG/07/bird_0.89677.pth'] --img_size [default=320]
```

## Reproduce the best submission file

1. Download the best weight 
-> google drive link : [bird_0.89677.pth](https://drive.google.com/file/d/1BGkMcoTOT5U24ufnyyMOjQtN74NASkJW/view?usp=sharing)
2. Drag the weight file to the root of the project folder
3. Run `python inference.py` to get the answer.txt in the root of the project folder

## Best Result


| User        | Team Name |Score  | Submission Date     |
| ------------|-----------|-------| ------------------- |
| Chialiang86 | 310552027 |0.87834| 11/01/2021 13:53:52 |


## References
- TransFG : https://github.com/TACJu/TransFG
- EfficientNet : https://github.com/lukemelas/EfficientNet-PyTorch
- ResNet : https://pytorch.org/hub/pytorch_vision_resnet/