from skimage import morphology
import torch
import numpy as np

def remove_small_area(mask, n_classes=2, minhair=5000, minface=5000):
    if n_classes == 2:
        hair_class = 1
    else:
        hair_class = 2
        face_class = 1
    mask = mask.cpu().numpy()

    hair_mask = (mask == hair_class)
    hair_mask = morphology.remove_small_objects(hair_mask, min_size=minhair, connectivity=1)
    hair_mask = morphology.remove_small_holes(hair_mask, min_size=100, connectivity=1)

    if n_classes == 3:
        face_mask = (mask == face_class)
        face_mask = morphology.remove_small_objects(face_mask, min_size=minface, connectivity=1)
        face_mask = morphology.remove_small_holes(face_mask, min_size=100, connectivity=1)

    mask = np.zeros(mask.shape)

    mask[hair_mask] = hair_class
    if n_classes == 3:
        mask[face_mask] = face_class

    return torch.from_numpy(mask)
