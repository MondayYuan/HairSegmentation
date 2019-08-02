import numpy as np
import cv2
import torch
from PIL import Image
from .type_conversion import *

hair_color = (0, 0, 255)
face_color = (255, 0, 0)

def overlay_mask_with_color(img, seg_mask, color):
    color_img = np.zeros(img.shape, img.dtype)
    color_img[:, :] = color
    # print(img.shape, seg_mask.shape)
    color_mask = cv2.bitwise_and(color_img, color_img, mask=seg_mask)
    display_image = cv2.addWeighted(color_mask, 0.3, img, 0.7, 0)
    return display_image


def overlay_segmentation(img, mask, show_skin=False):
    """ Overlays the hair-face segmentation over the input image
    :param img: input bgr-image containing the color image
    :param mask: input greyscale image containing the segmentation mask
                (0 - background pixel,
                1 - skin pixel,
                2 - hair pixel)
    :return: a "pretty" view of the segmentation (the segmentation mask: mask is super-imposed over the input image: img)
    """
    if show_skin:
        hair_mask = np.zeros(mask.shape, dtype=np.uint8)
        hair_mask[mask == 2] = 255
        segmentation_color = overlay_mask_with_color(img, hair_mask, hair_color)


        skin_mask = np.zeros(mask.shape, dtype=np.uint8)
        skin_mask[mask == 1] = 255
        segmentation_color = overlay_mask_with_color(segmentation_color, skin_mask, face_color)
    else:
        hair_mask = np.zeros(mask.shape, dtype=np.uint8)
        hair_mask[mask == 1] = 255
        segmentation_color = overlay_mask_with_color(img, hair_mask, hair_color)

    return segmentation_color

def overlay_segmentation_mask(img, mask, inmode='opencv', outmode='opencv'):
        if inmode=='tensor':
            tmp_img = tensor2opencv(img)
        elif inmode=='PIL':
            tmp_img = PIL2opencv(img)
        elif inmode == 'opencv':
            tmp_img = img.copy()

        tmp_mask = mask.squeeze(0).cpu().numpy()
        img_masked = overlay_segmentation(tmp_img, tmp_mask)

        if outmode == 'tensor':
            return opencv2tensor(img_masked)
        elif outmode == 'opencv':
            return img_masked
        elif outmode == 'PIL':
            return opencv2PIL(img_masked)
