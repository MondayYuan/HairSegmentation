import torch
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor, ToPILImage
import cv2

def PIL2tensor(img):
    return ToTensor()(img)

def tensor2PIL(img):
    return ToPILImage()(img.cpu())

def PIL2opencv(img):
    return cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)

def opencv2PIL(img):
    return Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

def opencv2tensor(img):
    return PIL2tensor(opencv2PIL(img))

def tensor2opencv(img):
    return PIL2opencv(tensor2PIL(img))

def skimage2opencv(img):
    return cv2.cvtColor((img * 255).astype(int), cv2.COLOR_RGB2BGR)

def opencv2skimage(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(float) / 255

def skimage2PIL(img):
    return opencv2PIL(skimage2opencv(img))

def PIL2skimage(img):
    return opencv2skimage(PIL2opencv(img))

def Index2Gray(mask, n_class=2):
    if n_class == 2:
        return  mask * 255
    else:
        return mask * 127 + 1

def Gray2Index(mask, n_class=2):
    if n_class == 2:
        return mask // 255
    else:
        return mask // 127

