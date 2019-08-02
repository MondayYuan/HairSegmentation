from torchvision.transforms import Compose, ToTensor, Normalize
from .preprocess import *
from .postprocess import remove_small_area
from .view_segmentation import overlay_segmentation_mask


class Segmenter(object):
    def __init__(self, model, device, detector=None, mode=None):
        self.model = model
        self.detector = detector
        self.mode = mode

        self.input_transform = Compose([
            ToTensor(),
            Normalize([.485, .456, .406], [.229, .224, .225])]
        )

        self.device = device

    def segment(self, img, is_remove_small_area=True):
        crop_img, box = face_detect(self.detector, img, mode=self.mode)

        tmp_img = opencv2PIL(crop_img)
        tmp_img = self.input_transform(tmp_img).to(self.device)

        mask = torch.argmax(self.model(tmp_img.unsqueeze(0)), 1)

        if is_remove_small_area:
            mask = remove_small_area(mask, minhair=2000)

        img_masked = overlay_segmentation_mask(crop_img, mask, inmode='opencv', outmode='opencv')

        return self.replace(img, img_masked, box)

    def replace(self, big, small, box):
        if box is None:
            return small.copy()

        big_tmp = big.copy()
        l, r, t, b = box
        big_tmp[t:b, l:r, :] = small
        return big_tmp
