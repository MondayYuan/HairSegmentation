import os
from torch.utils.data import Dataset
import sys
import cv2
import glob
import torch
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import random
from torchvision.transforms import ToTensor, Compose, Normalize, ColorJitter
from .transform import ToIndex, RandomCrop, RandomRotate, RandomHorizontallyFlip, CenterCrop, Rescale
from .transform import Compose as ComposeX

class Figaro(Dataset):
    def __init__(self, datadir, mode, argument=True):
        self.input_images_dir = os.path.join(datadir, 'Originals')
        self.masks_dir = os.path.join(datadir, 'Ground_Truth')

        self.mode = mode
        self.argument = argument

        if self.mode == 'train':
            list_path = 'train.txt'
        elif self.mode == 'val':
            list_path = 'val.txt'
        elif self.mode == 'trainval':
            list_path = 'trainval.txt'
        else:
            raise Warning("mode must be train/val/trainval")

        list_path = os.path.join(datadir, list_path)

        with open(list_path, 'r') as f:
            imgs_list = f.readlines()

        self.images_list = [x.strip() for x in imgs_list]

        if self.argument == False:
            self.image_transforms = Compose([
                ToTensor(),
                Normalize([.485, .456, .406], [.229, .224, .225])
            ])
            self.mask_transforms = ToIndex(dataset='figaro')
            self.joint_transforms = None
        else:
             self.image_transforms = Compose([
                    ColorJitter(0.05, 0.05, 0.05),
                    ToTensor(),
                    Normalize([.485, .456, .406], [.229, .224, .225])
                ])
             self.mask_transforms = ToIndex(dataset='figaro')

             if self.mode == 'train' or self.mode == 'trainval':
                 self.joint_transforms = ComposeX([
                    Rescale(200),
                    RandomRotate(5),
                    RandomCrop((218, 178)),
                    RandomHorizontallyFlip()
                 ])
             elif self.mode == 'val':
                 self.joint_transforms = ComposeX([
                     Rescale(200),
                     CenterCrop((218, 178)),
                 ])

    def __getitem__(self,idx):
        image_name = self.images_list[idx]
        img_path = os.path.join(self.input_images_dir, image_name + '.jpg')
        mask_path = os.path.join(self.masks_dir, image_name + '.pbm')

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        if self.joint_transforms is not None:
            img, mask = self.joint_transforms(img, mask)

        if self.image_transforms is not None:
            img = self.image_transforms(img)

        if self.mask_transforms is not None:
            mask = self.mask_transforms(mask)

        return img, mask, img_path

    def __len__(self):
        return len(self.images_list)
        # return 32


class CelebA(Dataset):
    def __init__(self, datadir, mode, n_classes=2, argument=True):
        self.input_images_dir = os.path.join(datadir, 'images')
        self.masks_dir = os.path.join(datadir, 'masks')

        self.mode = mode
        self.n_classes = n_classes
        self.argument = argument

        if self.mode == 'train':
            list_path = 'train.txt'
        elif self.mode == 'val':
            list_path = 'val.txt'
        elif self.mode == 'trainval':
            list_path = 'trainval.txt'
        else:
            raise Warning("mode must be train/val/trainval")

        list_path = os.path.join(datadir, list_path)

        with open(list_path, 'r') as f:
            imgs_list = f.readlines()

        self.images_list = [x.strip() for x in imgs_list]

        if self.argument == False:
            self.image_transforms = Compose([
                ToTensor(),
                Normalize([.485, .456, .406], [.229, .224, .225])
            ])
            self.mask_transforms = ToIndex(dataset='celeba', n_classes=self.n_classes)
            self.joint_transforms = None
        else:
             self.image_transforms = Compose([
                    ColorJitter(0.05, 0.05, 0.05),
                    ToTensor(),
                    Normalize([.485, .456, .406], [.229, .224, .225])
                ])
             self.mask_transforms = ToIndex(dataset='celeba', n_classes=self.n_classes)

             if self.mode == 'train' or self.mode == 'trainval':
                 self.joint_transforms = ComposeX([
                    # RandomCrop(160),
                    # RandomRotate(10),
                    RandomHorizontallyFlip()
                 ])
             elif self.mode == 'val':
                 self.joint_transforms = None


    def __getitem__(self, item):
        image_name = self.images_list[item]
        image_path = os.path.join(self.input_images_dir, image_name + '.png')
        mask_path = os.path.join(self.masks_dir, image_name + '.bmp')

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.joint_transforms:
            image, mask = self.joint_transforms(image, mask)
        image = self.image_transforms(image)
        mask = self.mask_transforms(mask)

        return image, mask, image_name

    def __len__(self):
        return len(self.images_list)
        # return 32


