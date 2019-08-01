import os
from torch.utils.data import Dataset
import sys
import cv2
import glob
import torch
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import random
from torchvision.transforms import ToTensor
from torchvision import transforms

class CelebA(Dataset):
    def __init__(self, datadir, mode, base_size=240, crop_size=200, argument=True):
        self.input_images_dir = os.path.join(datadir, 'images')
        self.masks_dir = os.path.join(datadir, 'masks')

        self.mode = mode
        self.base_size = base_size
        self.crop_size = crop_size
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

        # self.image_pairs = [(os.path.join(self.input_images_dir, x+'.png'), os.path.join(self.masks_dir, x+'.bmp')) for x in imgs_list]

        # self.image_pairs = self.find_corresponding_images(self.masks_dir, self.input_images_dir)

    def __getitem__(self, item):
        image_name = self.images_list[item]
        image_path = os.path.join(self.input_images_dir, image_name + '.png')
        mask_path = os.path.join(self.masks_dir, image_name + '.bmp')

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.mode == 'train' or self.mode == 'trainval':
            image, mask = self._train_sync_transform(image, mask)
        elif self.mode == 'val':
            image, mask = self._val_sync_transform(image, mask)
        return image, mask, image_name

    def _train_sync_transform(self, img, mask):
        '''
        :param image:  PIL input image
        :param gt_image: PIL input gt_image
        :return:
        '''
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if self.argument:
            crop_size = self.crop_size
            # random scale (short edge)
            short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
            w, h = img.size
            if h > w:
                ow = short_size
                oh = int(1.0 * h * ow / w)
            else:
                oh = short_size
                ow = int(1.0 * w * oh / h)
            img = img.resize((ow, oh), Image.BILINEAR)
            mask = mask.resize((ow, oh), Image.NEAREST)
            # pad crop
            if short_size < crop_size:
                padh = crop_size - oh if oh < crop_size else 0
                padw = crop_size - ow if ow < crop_size else 0
                img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
                mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
            # random crop crop_size
            w, h = img.size
            x1 = random.randint(0, w - crop_size)
            y1 = random.randint(0, h - crop_size)
            img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
            mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))

        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _val_sync_transform(self, img, mask):
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, image):
        image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        image = image_transforms(image)
        return image

    def _mask_transform(self, gt_image):
        # target = self._class_to_index(np.array(gt_image).astype('int32'))
        target = np.asarray(np.array(gt_image) // 127, dtype=np.int32)
        target = torch.from_numpy(target).long()

        return target

    def __len__(self):
        return len(self.images_list)
        # return 32


