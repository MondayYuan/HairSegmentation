import cv2
import torch
from PIL import Image
from utils.segmenter import Segmenter
from utils.type_conversion import *

def resize(img, short_size):
    w, h = img.size
    if w < h:
        nw, nh = short_size, int(w * short_size / h)
    else:
        nw, nh = int(h * short_size / w), short_size

    return img.resize((nh, nw))

def test_image(args, model):
    if args.detector == 'dlib':
        import dlib
    elif args.detector == 'faceboxes':
        from utils.face_detector import FaceDetectorFaceboxes

    model.eval()

    device = torch.device("cuda" if args.gpu else "cpu")

    image = Image.open(args.image).convert('RGB')

    if args.resize > 0:
        image = resize(image, args.resize)

    detector = None
    if args.detector == 'dlib':
        detector = dlib.get_frontal_face_detector()
    elif args.detector == 'faceboxes':
        MODEL_PATH = 'model/faceboxes.pb'
        detector = FaceDetectorFaceboxes(MODEL_PATH, gpu_memory_fraction=0.25, visible_device_list='0')

    segmenter = Segmenter(model, device, detector, mode=args.detector)

    result = segmenter.segment(PIL2opencv(image), args.remove_small_area)
    result = opencv2PIL(result)

    if args.save:
        result.save(args.save)

    if not args.unshow:
        result.show()
        image.show()

def test_video(args, model):
    if args.video == '0':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.video)

    w_win = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_win = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(w_win, h_win)

    if args.resize > 0:
        short_size = args.resize
        if w_win > h_win:
            nw, nh = short_size, int(w_win * short_size / h_win)
        else:
            nw, nh = int(h_win * short_size / w_win), short_size
    else:
        nw, nh = w_win, h_win

    detector = None
    if args.detector == 'dlib':
        detector = dlib.get_frontal_face_detector()
    elif args.detector == 'faceboxes':
        MODEL_PATH = 'model/faceboxes.pb'
        detector = FaceDetectorFaceboxes(MODEL_PATH, gpu_memory_fraction=0.25, visible_device_list='0')

    device = torch.device("cuda" if args.gpu else "cpu")
    segmenter = Segmenter(model, device, detector, mode=args.detector)

    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(args.save, fourcc, 20, (nh, nw), True)

    while True:
        frame = cap.read()[1]

        if frame is None:
            break

        frame = cv2.resize(frame, (nh, nw))

        result = segmenter.segment(frame, args.remove_small_area)

        if args.save:
            out.write(result)


        if not args.unshow:
            cv2.imshow('image', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if args.save:
        out.release()

