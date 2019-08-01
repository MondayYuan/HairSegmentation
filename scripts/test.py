import cv2
import torch
from utils.view_segmentation import overlay_segmentation_mask
import numpy as np
from PIL import Image
from torchvision.transforms import Normalize, ToTensor, Compose
from utils.postprocess import remove_small_area

def resize(img, short_size):
    w, h = img.size
    if w < h:
        nw, nh = short_size, int(w * short_size / h)
    else:
        nw, nh = int(h * short_size / w), short_size

    return img.resize((nh, nw))


def test(args, model):
    model.eval()

    device = torch.device("cuda" if args.gpu else "cpu")

    # image = cv2.imread(args.image)
    image = Image.open(args.image).convert('RGB')

    if args.resize:
        image = resize(image, 300)


    ori_image = image.copy()
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = torch.from_numpy(image).float().permute(2, 0, 1).div(255).to(device)
    input_transform = Compose([
        ToTensor(),
        Normalize([.485, .456, .406], [.229, .224, .225])]
    )
    image = input_transform(image).to(device)

    mask = torch.argmax(model(image.unsqueeze(0)), 1)

    if args.remove_small_area:
        mask = remove_small_area(mask)

    img_masked = overlay_segmentation_mask(ori_image, mask, inmode='PIL', outmode='PIL')

    if args.save:
        # cv2.imwrite(args.save, img_masked)
        img_masked.save(args.save)

    img_masked.show()
    ori_image.show()

    # cv2.imshow('image', ori_image)
    # cv2.imshow('image_masked', img_masked)

    # cv2.waitKey()
