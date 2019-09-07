import cv2
from .type_conversion import *

def centercrop(img, shape):
    ow, oh = img.shape[0], img.shape[1]
    nw, nh = shape[0], shape[1]

    cx, cy = ow // 2, oh // 2

    l = max(0, cx - nw // 2)
    r = min(ow , cx + nw //2)
    t = min(0, cy - nh // 2)
    b = max(oh , cy + nh // 2)

    return  img[:, l:r, t:b]

def face_detect(detector, frame, mode='dlib'):
    if mode == 'dlib':
        dets = detector(frame, 1)

        if len(dets) > 0:
            box = dets[0]
            # cv2.rectangle(frame, (box.left(), box.top()), (box.right(), box.bottom()), (255, 0, 0))

            w, h = box.right() - box.left(), box.bottom() - box.top()
            nl = max(0, box.left() - w)
            nr = min(frame.shape[1], box.right() + w)
            nt = max(0, box.top() - h )
            nb = min(frame.shape[0], box.bottom() + h // 2)

            # cv2.rectangle(frame, (nl, nt), (nr, nb), (0, 255, 0))

            return frame[nt:nb, nl:nr, :], (nl, nr, nt, nb)
        else:
            return frame.copy(), None

    elif mode == 'faceboxes':
        boxes, scores = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), score_threshold=0.8)

        if len(boxes) > 0:
            t, l, b, r = [int(x) for x in boxes[0]]

            # cv2.rectangle(frame, (l, t), (r, b), (255, 0, 0))

            w, h = r - l, b - t
            nl = max(0, l - w // 2)
            nr = min(frame.shape[1], r + w // 2)
            nt = max(0, t - h)
            nb = min(frame.shape[0], b + h // 5)

            # cv2.rectangle(frame, (nl, nt), (nr, nb), (0, 255, 0))

            return frame[nt:nb, nl:nr, :], (nl, nr, nt, nb)
        else:
            return frame.copy(), None

    else:
        return frame.copy(), None


