import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor, Compose, Normalize, ColorJitter
from .transform import ToIndex, RandomCrop, RandomRotate, RandomHorizontallyFlip, CenterCrop, Resize
from .transform import Compose as ComposeX
import cv2
import torch

class Figaro(Dataset):
    def __init__(self, datadir, resize, crop_size, argument=True):
        self.input_images_dir = os.path.join(datadir, 'Originals')
        self.masks_dir = os.path.join(datadir, 'Ground_Truth')

        self.pairlist = self.search_pair()
        self.argument = argument

        if resize > 0:
            f_resize = Resize(resize)
        else:
            f_resize = None

        if crop_size is not None:
            f_crop = RandomCrop(crop_size)
        else:
            f_crop = None

        if self.argument == False:
            self.image_transforms = Compose([
                ToTensor(),
                Normalize([.485, .456, .406], [.229, .224, .225])
            ])
            self.mask_transforms = ToIndex(dataset='figaro')

            self.joint_transforms = ComposeX([
                f_resize,
                f_crop
            ])
        else:
             self.image_transforms = Compose([
                    ColorJitter(0.1, 0.1, 0.1),
                    ToTensor(),
                    Normalize([.485, .456, .406], [.229, .224, .225])
                ])
             self.mask_transforms = ToIndex(dataset='figaro')

             self.joint_transforms = ComposeX([
                f_resize,
                # RandomRotate(5),
                f_crop,
                RandomHorizontallyFlip()
             ])

    def search_pair(self):
        images_list = []
        masks_list = []
        for dirname in os.listdir(self.input_images_dir):
            totaldir = os.path.join(self.input_images_dir, dirname)
            if not os.path.isdir(totaldir):
                continue
            for filename in os.listdir(totaldir):
                if not filename.endswith('.jpg'):
                    continue
                images_list.append(os.path.join(dirname, filename[:-4]))

        for dirname in os.listdir(self.masks_dir):
            totaldir = os.path.join(self.masks_dir, dirname)
            if not os.path.isdir(totaldir):
                continue
            for filename in os.listdir(totaldir):
                if not filename.endswith('.pbm'):
                    continue
                masks_list.append(os.path.join(dirname, filename[:-4]))

        return list(set(images_list) & set(masks_list))

    def __getitem__(self,idx):
        image_name = self.pairlist[idx]

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
        return len(self.pairlist)
        # return 32


class CelebA(Dataset):
    def __init__(self, datadir, resize, argument=True):
        self.input_images_dir = os.path.join(datadir, 'images')
        self.masks_dir = os.path.join(datadir, 'masks')

        self.argument = argument
        self.pairlist = self.search_pairs()

        if self.argument == False:
            self.image_transforms = Compose([
                ToTensor(),
                Normalize([.485, .456, .406], [.229, .224, .225])
            ])
            self.mask_transforms = ToIndex(dataset='celeba')
            if resize > 0:
                self.joint_transforms = ComposeX([
                    Resize(resize),
                ])
            else:
                self.joint_transforms = None
        else:
            self.image_transforms = Compose([
                ColorJitter(0.05, 0.05, 0.05),
                ToTensor(),
                Normalize([.485, .456, .406], [.229, .224, .225])
            ])
            self.mask_transforms = ToIndex(dataset='celeba')

            if resize > 0:
                self.joint_transforms = ComposeX([
                    Resize(resize),
                    RandomHorizontallyFlip()
                ])
            else:
                self.joint_transforms = RandomHorizontallyFlip()

    def search_pairs(self):
        images_list = []
        masks_list = []
        for file in os.listdir(self.input_images_dir):
            if file.endswith('.png'):
                images_list.append(file[:-4])
        for file in os.listdir(self.masks_dir):
            if file.endswith('.bmp'):
                masks_list.append(file[:-4])

        return list(set(images_list) & set(masks_list))

    def __getitem__(self, item):
        image_name = self.pairlist[item]
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
        return len(self.pairlist)
        # return 32

class OurDataset(Dataset):
    def __init__(self, datadir, resize, argument=True):
        self.datadir = datadir
        self.pairlist = self.search_pairs()

        if argument == False:
            self.image_transforms = Compose([
                ToTensor(),
                Normalize([.485, .456, .406], [.229, .224, .225])
            ])
            self.mask_transforms = ToIndex(dataset='our')
            if resize > 0:
                self.joint_transforms = ComposeX([
                    Resize(resize),
                ])
            else:
                self.joint_transforms = None
        else:
            self.image_transforms = Compose([
                    ColorJitter(0.05, 0.05, 0.05),
                    ToTensor(),
                    Normalize([.485, .456, .406], [.229, .224, .225])
            ])

            self.mask_transforms = ToIndex(dataset='our')

            if resize > 0:
                self.joint_transforms = ComposeX([
                    Resize(resize),
                    RandomHorizontallyFlip()
                ])
            else:
                self.joint_transforms = RandomHorizontallyFlip()

    def search_pairs(self):
        jpg_list = []
        png_list = []
        for file in os.listdir(self.datadir):
            if file.endswith('.jpg'):
                jpg_list.append(file[:-4])
            elif file.endswith('_mask.png'):
                png_list.append(file[:-9])

        return list(set(jpg_list) & set(png_list))

    def __getitem__(self, idx):
        image_name = self.pairlist[idx]
        img_path = os.path.join(self.datadir, image_name + '.jpg')
        mask_path = os.path.join(self.datadir, image_name + '_mask.png')

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
        return len(self.pairlist)
        # return 32

class CelebAMaskHQ(Dataset):
    def __init__(self, datadir, resize, argument=True):
        self.img_dir = os.path.join(datadir, 'CelebA-HQ-img')
        self.mask_dir = os.path.join(datadir, 'CelebAMask-HQ-mask-anno')

        if resize < 0:
            self.resize = 512
        else:
            self.resize = resize

        if argument == False:
            self.image_transforms = Compose([
                ToTensor(),
                Normalize([.485, .456, .406], [.229, .224, .225])
            ])
            self.mask_transforms = ToIndex(dataset='celebamask-hq')
            self.joint_transforms = Resize(self.resize)

        else:
            self.image_transforms = Compose([
                    ColorJitter(0.05, 0.05, 0.05),
                    ToTensor(),
                    Normalize([.485, .456, .406], [.229, .224, .225])
            ])

            self.mask_transforms = ToIndex(dataset='celebamask-hq')
            self.joint_transforms = ComposeX([
                Resize(self.resize),
                RandomHorizontallyFlip()
            ])

    def __getitem__(self, item):
        img_path = os.path.join(self.img_dir, f'{item}.jpg')
        mask_path = os.path.join(f'{item // 2000}', '{:0>5d}_hair.png'.format(item))
        mask_path = os.path.join(self.mask_dir, mask_path)

        img = Image.open(img_path)
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('1')
        else:
            mask = Image.fromarray(np.zeros((self.resize, self.resize))).convert('1')

        if self.joint_transforms is not None:
            img, mask = self.joint_transforms(img, mask)

        if self.image_transforms is not None:
            img = self.image_transforms(img)

        if self.mask_transforms is not None:
            mask = self.mask_transforms(mask)

        return img, mask, img_path

    def __len__(self):
        return 30000
        # return 32

class SpringFace(Dataset):
    def __init__(self, datadir, resize, argument=True):
        self.datadir = datadir
        self.pairlist = self.search_pairs()

        if argument == False:
            self.image_transforms = Compose([
                ToTensor(),
                Normalize([.485, .456, .406], [.229, .224, .225])
            ])
            self.mask_transforms = ToIndex(dataset='springface')
            if resize > 0:
                self.joint_transforms = ComposeX([
                    Resize(resize),
                ])
            else:
                self.joint_transforms = None
        else:
            self.image_transforms = Compose([
                    ColorJitter(0.05, 0.05, 0.05),
                    ToTensor(),
                    Normalize([.485, .456, .406], [.229, .224, .225])
            ])

            self.mask_transforms = ToIndex(dataset='springface')

            if resize > 0:
                self.joint_transforms = ComposeX([
                    Resize(resize),
                    RandomHorizontallyFlip()
                ])
            else:
                self.joint_transforms = RandomHorizontallyFlip()

    def search_pairs(self):
        masks_list = []
        images_list = []
        for file in os.listdir(self.datadir):
            if file.endswith('_mask.png'):
                masks_list.append(file[:-9])
            elif file.endswith('.png'):
                images_list.append(file[:-4])

        return list(set(masks_list) & set(images_list))

    def __getitem__(self, item):
        image_name = self.pairlist[item]
        image_path = os.path.join(self.datadir, image_name + '.png')
        mask_path = os.path.join(self.datadir, image_name + '_mask.png')

        img = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')

        if self.joint_transforms is not None:
            img, mask = self.joint_transforms(img, mask)

        if self.image_transforms is not None:
            img = self.image_transforms(img)

        if self.mask_transforms is not None:
            mask = self.mask_transforms(mask)

        return img, mask, image_name

    def __len__(self):
        return len(self.pairlist)

class SpringHair(Dataset):
    def __init__(self, datadir, resize, crop_size, argument=True):
        self.datadir = datadir
        self.pairlist = self.search_pairs()

        if resize > 0:
            f_resize = Resize(resize)
        else:
            f_resize = None

        if crop_size is not None:
            f_crop = RandomCrop(crop_size)
        else:
            f_crop = None

        if argument == False:
            self.image_transforms = Compose([
                ToTensor(),
                Normalize([.485, .456, .406], [.229, .224, .225])
            ])
            self.mask_transforms = ToIndex(dataset='springhair')
            self.joint_transforms = ComposeX([
                f_resize,
                f_crop
            ])
        else:
            self.image_transforms = Compose([
                    ColorJitter(0.1, 0.1, 0.1),
                    ToTensor(),
                    Normalize([.485, .456, .406], [.229, .224, .225])
            ])

            self.mask_transforms = ToIndex(dataset='springhair')

            self.joint_transforms = ComposeX([
                f_resize,
                f_crop,
                RandomHorizontallyFlip()
            ])

    def search_pairs(self):
        masks_list = []
        images_list = []
        for file in os.listdir(self.datadir):
            if file.endswith('.jpg'):
                masks_list.append(file[:-4])
            elif file.endswith('.png'):
                images_list.append(file[:-4])

        return list(set(masks_list) & set(images_list))

    def __getitem__(self, item):
        image_name = self.pairlist[item]
        image_path = os.path.join(self.datadir, image_name + '.jpg')
        mask_path = os.path.join(self.datadir, image_name + '.png')

        img = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')

        if self.joint_transforms is not None:
            img, mask = self.joint_transforms(img, mask)

        if self.image_transforms is not None:
            img = self.image_transforms(img)

        if self.mask_transforms is not None:
            mask = self.mask_transforms(mask)

        return img, mask, image_name

    def __len__(self):
        return len(self.pairlist)


