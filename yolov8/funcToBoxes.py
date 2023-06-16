import cv2
import numpy as np


def DivideImageIntoBox(H, W):
    XAxis = np.linspace(0, W, 4, dtype=np.int32)
    YAxis = np.linspace(0, H, 4, dtype=np.int32)
    b1 = np.array((XAxis[0], YAxis[0], XAxis[1], YAxis[1]))  # TL
    b2 = np.array((XAxis[1], YAxis[0], XAxis[2], YAxis[1]))  # T
    b3 = np.array((XAxis[2], YAxis[0], XAxis[3], YAxis[1]))  # TR
    b4 = np.array((XAxis[0], YAxis[1], XAxis[1], YAxis[2]))  # L
    b5 = np.array((XAxis[1], YAxis[1], XAxis[2], YAxis[2]))  # C
    b6 = np.array((XAxis[2], YAxis[1], XAxis[3], YAxis[2]))  # R
    b7 = np.array((XAxis[0], YAxis[2], XAxis[1], YAxis[3]))  # BL
    b8 = np.array((XAxis[1], YAxis[2], XAxis[2], YAxis[3]))  # B
    b9 = np.array((XAxis[2], YAxis[2], XAxis[3], YAxis[3]))  # BR
    BB = np.array((b1, b2, b3, b4, b5, b6, b7, b8, b9))
    return BB


def showDivideImageIntoBox(image):
    img = image
    H, W = img.shape[:2]
    for i in DivideImageIntoBox(H, W):
        p1, p2 = (i[:2], i[2:])
        cv2.rectangle(img, p1, p2, (0, 0, 0), 0, cv2.LINE_AA)
    cv2.imshow("DivideImageIntoBox", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def box_label(image, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255)):
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[
            0
        ]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            image,
            label,
            (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
            0,
            lw / 3,
            txt_color,
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def PlotBoxes(image, boxes, labels=[], colors=[], score=True, conf=None):
    # plot each boxes
    for box in boxes:
        # add score in label if score=True
        if score:
            label = (
                labels[int(box[-1]) + 1]
                + " "
                + str(round(100 * float(box[-2]), 1))
                + "%"
            )
        else:
            label = labels[int(box[-1]) + 1]
        # filter every box under conf threshold if conf threshold setted
        if conf:
            if box[-2] > conf:
                color = colors[int(box[-1])]
                box_label(image, box, label, color)
        else:
            color = colors[int(box[-1])]
            box_label(image, box, label, color)
    return image


def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2

    Arguments:
    box1 -- first box, list object with coordinates (box1_x1, box1_y1, box1_x2, box_1_y2)
    box2 -- second box, list object with coordinates (box2_x1, box2_y1, box2_x2, box2_y2)
    """
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_width = max((xi2 - xi1), 0)
    inter_height = max((yi2 - yi1), 0)
    inter_area = inter_height * inter_width
    box1_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
    box2_area = (box2[3] - box2[1]) * (box2[2] - box2[0])
    union_area = (box1_area + box2_area) - inter_area
    iou = inter_area / union_area
    return iou


def Text(image, Box, cls, label, side):
    H, W = image.shape[:2]
    BB = DivideImageIntoBox(H, W)
    IOU = []
    text = []
    c = 0
    for box in Box:
        for b in BB:
            IOU.append(iou(box, b))
        text.append("a " + label[cls[c] + 1] + " " + side[np.argmax(np.array(IOU))])
        c += 1
        IOU = []
    return text
