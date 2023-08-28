from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from ultralytics import YOLO
import base64
import cv2


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


def iou(box1, box2):
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


def Text(image, Box, cls):
    side = [
        "on the top left",
        "on the top",
        "on the top right",
        "on the left",
        "on the middle",
        "on the right",
        "on the bottom left",
        "on the bottom",
        "on the bottom right",
    ]
    label = {
        0: "__backgrond__",
        1: "person",
        2: "bicycle",
        3: "car",
        4: "motorcycle",
        5: "airplane",
        6: "bs",
        7: "train",
        8: "trck",
        9: "boat",
        10: "traffic light",
        11: "fire hydrant",
        12: "stop sign",
        13: "parking meter",
        14: "bench",
        15: "bird",
        16: "cat",
        17: "dog",
        18: "horse",
        19: "sheep",
        20: "cow",
        21: "elephant",
        22: "bear",
        23: "zebra",
        24: "giraffe",
        25: "backpack",
        26: "mbrella",
        27: "handbag",
        28: "tie",
        29: "sitcase",
        30: "frisbee",
        31: "skis",
        32: "snowboard",
        33: "sports ball",
        34: "kite",
        35: "baseball bat",
        36: "baseball glove",
        37: "skateboard",
        38: "srfboard",
        39: "tennis racket",
        40: "bottle",
        41: "wine glass",
        42: "cp",
        43: "fork",
        44: "knife",
        45: "spoon",
        46: "bowl",
        47: "banana",
        48: "apple",
        49: "sandwich",
        50: "orange",
        51: "broccoli",
        52: "carrot",
        53: "hot dog",
        54: "pizza",
        55: "dont",
        56: "cake",
        57: "chair",
        58: "coch",
        59: "potted plant",
        60: "bed",
        61: "dining table",
        62: "toilet",
        63: "tv",
        64: "laptop",
        65: "mose",
        66: "remote",
        67: "keyboard",
        68: "cell phone",
        69: "microwave",
        70: "oven",
        71: "toaster",
        72: "sink",
        73: "refrigerator",
        74: "book",
        75: "clock",
        76: "vase",
        77: "scissors",
        78: "teddy bear",
        79: "hair drier",
        80: "toothbrsh",
    }
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


app = FastAPI()

model = YOLO("yolov8x.pt")

# Configure CORS
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"blind guide api"}


@app.websocket("/detect-image/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("the connection accepted")
    cnt = 0
    while True:
        try:
            encoded_image = await websocket.receive_text()
            decoded_image = base64.b64decode(encoded_image)
            img = cv2.imdecode(np.frombuffer(decoded_image, np.uint8), cv2.IMREAD_COLOR)
            conf = 0.7
            device = "cpu"
            results = model(img, conf=conf, device=device)
            boxes_list = results[0].boxes.xyxyn.numpy().tolist()
            scores_list = results[0].boxes.conf.numpy().tolist()
            classes_list = results[0].boxes.cls.numpy().tolist()
            text = Text(img, results[0].boxes.xyxy, classes_list)
            cnt += 1
            await websocket.send_json(
                {
                    "text": text,
                    "boxes_list": boxes_list,
                    "scores_list": scores_list,
                    "classes_list": classes_list,
                }
            )
            print("sent")
        except:
            pass
            break