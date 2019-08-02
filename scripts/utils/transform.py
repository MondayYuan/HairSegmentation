from torchvision.transforms import ToTensor
import random
from PIL import Image, ImageOps
import torch
import numpy as np

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

class Resize(object):
    def __init__(self, size):
        self.w = 0
        self.h = 0
        if isinstance(size, int):
            self.w = size
            self.h = size
        elif isinstance(size, tuple) and len(size) == 2:
            if isinstance(size[0], int) and isinstance(size[1], int):
                self.w = size[0]
                self.h = size[1]
            else:
                raise ValueError
        else:
            raise ValueError

    def __call__(self, img, mask):
        return (img.resize((self.w, self.h), Image.NEAREST),
                    mask.resize((self.w, self.h), Image.BILINEAR))


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
    def __init__(self, dataset='celeba', n_classes=2):
        self.dataset = dataset
        self.n_classes = n_classes

    def __call__(self, mask):
        if self.dataset == 'celeba':
            if self.n_classes == 3:
                target = np.asarray(np.array(mask) // 127, dtype=np.int32)
                return torch.from_numpy(target).long()

            elif self.n_classes == 2:
                target = np.asarray(np.array(mask) // 255, dtype=np.int32)
                return torch.from_numpy(target).long()
            else:
                print('n_classes should be 2 or 3')
                raise ValueError
        elif self.dataset == 'figaro':
            return torch.from_numpy(np.asarray(mask, dtype=np.int))
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
            img, mask = t(img, mask)
        return img, mask

class Rescale(object):
    def __init__(self, short_size):
        self.short_size = short_size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size

        if w > h:
            nw, nh =  w * self.short_size // h, self.short_size
        else:
            nw, nh = self.short_size, self.short_size * h // w

        return img.resize((nw, nh), Image.BILINEAR), mask.resize((nw, nh), Image.NEAREST)
