import glob
import cv2
import torch
import torchvision.models as models
from torchvision import datasets, transforms as T
from PIL import Image
import numpy as np 

f_class = open('data/classes.txt', 'r')
class_list = [str(c).strip() for c in f_class.readlines()]

f_training_info = open('data/training_labels.txt', 'r')
training_info = [str(c).strip('\n') for c in f_training_info.readlines()]


preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_x_raw = []
train_y = []
train_img_path = 'data/training_images/'
test_img_path = 'data/testing_images/'

print('preprocessing data...')
train_x = torch.stack([preprocess(Image.open('{}{}'.format(train_img_path, info.split()[0]))) for info in training_info])
train_y = torch.Tensor([info.split()[1] for info in training_info])
print(train_x)

model = models.resnet18(pretrained=True)
model.eval()
