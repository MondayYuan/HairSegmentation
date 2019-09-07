from torchvision.transforms import ToTensor
import random
from PIL import Image, ImageOps
import torch
import numpy as np
from .type_conversion import PIL2opencv
import cv2

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST)

class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask

# class Resize(object):
#     def __init__(self, size):
#         self.w = 0
#         self.h = 0
#         if isinstance(size, int):
#             self.w = size
#             self.h = size
#         elif isinstance(size, tuple) and len(size) == 2:
#             if isinstance(size[0], int) and isinstance(size[1], int):
#                 self.w = size[0]
#                 self.h = size[1]
#             else:
#                 raise ValueError
#         else:
#             raise ValueError
#
#     def __call__(self, img, mask):
#         return (img.resize((self.w, self.h), Image.NEAREST),
#                     mask.resize((self.w, self.h), Image.BILINEAR))


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))

class ToIndex(object):
    def __init__(self, dataset='celeba'):
        self.dataset = dataset

    def __call__(self, mask):
        if self.dataset == 'celeba':
            target = np.asarray(np.array(mask) // 255, dtype=np.int32)
            return torch.from_numpy(target).long()
        elif self.dataset == 'figaro':
            return torch.from_numpy(np.asarray(mask, dtype=np.int))
        elif self.dataset == 'our':
            mask = np.asarray(mask, dtype=np.int)
            tmp_mask = np.zeros(mask.shape, dtype=np.int)
            tmp_mask[mask == 3] = 1
            return torch.from_numpy(tmp_mask)
        elif self.dataset == 'aisegment':
            return torch.from_numpy(np.asarray(mask) / 255)
        elif self.dataset == 'celebamask-hq':
            return torch.from_numpy(np.asarray(mask, dtype=int))
        elif self.dataset == 'springface':
            mask = PIL2opencv(mask)
            red = [0, 0, 255]
            mask_tmp = (mask[:,:, 0] == red[0]) & (mask[:, :, 1] == red[1]) & (mask[:, :, 2] == red[2])
            return torch.from_numpy(np.asarray(mask_tmp, dtype=int))
        elif self.dataset == 'springhair':
            mask = PIL2opencv(mask)
            magenta = [255, 0, 255]
            mask_tmp = (mask[:,:, 0] == magenta[0]) & (mask[:, :, 1] == magenta[1]) & (mask[:, :, 2] == magenta[2])
            return torch.from_numpy(np.asarray(mask_tmp, dtype=int))
        else:
            print('dataset should be figaro or celeba')
            raise TypeError

class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            if t is None:
                continue
            img, mask = t(img, mask)
        return img, mask

class Resize(object):
    def __init__(self, short_size):
        self.short_size = short_size

    def __call__(self, img, mask):
        n_size = []
        for tmp_img in [img, mask]:
            w, h = tmp_img.size

            if w > h:
                nw, nh =  w * self.short_size // h, self.short_size
            else:
                nw, nh = self.short_size, self.short_size * h // w

            n_size.append((nw, nh))

        return img.resize((n_size[0][0], n_size[0][1]), Image.BILINEAR), mask.resize((n_size[1][0], n_size[1][1]), Image.NEAREST)

