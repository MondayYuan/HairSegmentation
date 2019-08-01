from skimage import morphology
import torch
import numpy as np

def remove_small_area(mask, minhair=5000, minface=5000):
    mask = mask.cpu().numpy()

    hair_mask = (mask == 2)
    hair_mask = morphology.remove_small_objects(hair_mask, min_size=minhair, connectivity=1)
    hair_mask = morphology.remove_small_holes(hair_mask, min_size=100, connectivity=1)

    face_mask = (mask == 1)
    face_mask = morphology.remove_small_objects(face_mask, min_size=minface, connectivity=1)
    face_mask = morphology.remove_small_holes(face_mask, min_size=100, connectivity=1)

    mask = np.zeros(mask.shape)
    mask[hair_mask] = 2
    mask[face_mask] = 1

    return torch.from_numpy(mask)
