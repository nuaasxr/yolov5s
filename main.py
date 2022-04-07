import time
import yaml
import cv2
from pathlib import Path
import numpy as np
from numpy import random
from models.common import *
from utils.general import non_max_suppression, scale_coords
from utils.augmentations import letterbox
from yolov5s import MyYolo

if __name__ == "__main__":
    cfg = './config/yolov5s.yaml'
    yaml_file = Path(cfg).name
    with open(cfg, encoding='ascii', errors='ignore') as f:
        d = yaml.safe_load(f)  # model dict
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    namess = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
              'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
              'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
              'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
              'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
              'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
              'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
              'hair drier', 'toothbrush']
    names = [str(namess[i]) for i in range(80)]
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    myyolo = MyYolo(nc=nc, anchors=anchors)
    myyolo.load_state_dict(torch.load('./weights/myyolov5s.pth', map_location=None))

    for name, value in myyolo.named_parameters():
        value.requires_grad = False
    params_conv = filter(lambda p: p.requires_grad, myyolo.parameters())

    imgname = 'zidane.jpg'

    with torch.no_grad():
        myyolo = myyolo.to(device)
        myyolo.fuse().eval().half()
        for _ in range(1):
            img0 = cv2.imread('./data/'+imgname)  # BGR
            # Padded resize
            img = letterbox(img0, 640, stride=32, auto=True)[0]
            # Convert
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
            im = torch.from_numpy(img).to(device)
            im = im.half()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t1 = time.time()
            d = myyolo(im)
            y = d[0]
            pred = non_max_suppression(y, conf_thres=0.30, iou_thres=0.45, classes=None, agnostic=True,
                                       max_det=100)  # pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det
            for i, det in enumerate(pred):  # per image
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img0.shape).round()
                    for *x, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = names[int(cls)] + ' ' + str(conf.cpu().numpy())
                        color = colors[int(cls)]
                        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
                        cv2.rectangle(img0, c1, c2, color, thickness=3, lineType=cv2.LINE_AA)
                        cv2.putText(img0, label, (c1[0], c1[1] - 5), 0, 1, color, thickness=3, lineType=cv2.LINE_AA)
            t2 = time.time()
            print(t2 - t1)
            cv2.imshow('result.jpg', img0)
            cv2.waitKey(0)
            cv2.imwrite('./output/'+'new'+imgname, img0)