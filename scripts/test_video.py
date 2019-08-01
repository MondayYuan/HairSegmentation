import cv2
import torch
from PIL import Image
from torchvision.transforms import Normalize, ToTensor, Compose
from argparse import ArgumentParser
from nets.unet import UNet
from torch import nn
from utils.view_segmentation import overlay_segmentation_mask
import dlib
from utils.type_conversion import *
from utils.preprocess import *
from utils.postprocess import remove_small_area
from utils.face_detector import FaceDetectorFaceboxes

short_size = 300

def segmentation(model, img, args):

    device = torch.device("cuda" if args.gpu else "cpu")
    tmp_img = opencv2PIL(img)

    input_transform = Compose([
        ToTensor(),
        Normalize([.485, .456, .406], [.229, .224, .225])]
    )
    tmp_img = input_transform(tmp_img).to(device)

    mask = torch.argmax(model(tmp_img.unsqueeze(0)), 1)


    if args.remove_small_area:
        mask = remove_small_area(mask, minhair=2000, minface=2000)

    img_masked = overlay_segmentation_mask(img, mask, inmode='opencv', outmode='opencv')

    return img_masked

def replace(big, small, box):
    if box is None:
        return small

    l, r, t, b = box
    big[t:b, l:r, :] = small
    return big

def test_video(model, args):
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(0)

    w_win = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_win = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if w_win > h_win:
        nw, nh = short_size, int(w_win * short_size / h_win)
    else:
        nw, nh = int(h_win * short_size / w_win), short_size

    detector = None
    if args.detector == 'dlib':
        detector = dlib.get_frontal_face_detector()
    elif args.detector == 'faceboxes':
        MODEL_PATH = 'model/faceboxes.pb'
        detector = FaceDetectorFaceboxes(MODEL_PATH, gpu_memory_fraction=0.25, visible_device_list='0')

    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(args.save, fourcc, 20, (nh, nw), True)

    while True:
        frame = cap.read()[1]

        if frame is None:
            break


        frame = cv2.resize(frame, (nh, nw))

        box = None
        if args.detector == 'dlib':
            tmp_frame, box = face_detect(detector, frame, mode='dlib')
        elif args.detector == 'faceboxes':
            tmp_frame, box = face_detect(detector, frame, mode='faceboxes')

        if box is None:
            tmp_frame = frame.copy()

        result = segmentation(model, tmp_frame, args)

        frame = replace(frame, result, box)

        if args.save:
            out.write(frame)


        cv2.imshow('image', frame)
        # cv2.imshow('result', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if args.save:
        out.release()

def main(args):
    model = None
    if args.model == 'unet':
        model = UNet(3)

    assert model is not None, f'model {args.model} not available'

    if args.gpu:
        model = model.cuda()

    if args.double_gpus:
        model = nn.DataParallel(model, [0, 1])

    model.load_state_dict(torch.load(args.load_path))

    test_video(model, args)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--double-gpus', action='store_true', default=False)
    parser.add_argument('--model', required=True)
    parser.add_argument('--load-path', required=True)
    parser.add_argument('--video')
    parser.add_argument('--detector')
    parser.add_argument('--save')
    parser.add_argument('--remove-small-area', action='store_true', default=False)

    args = parser.parse_args()

    main(args)
