from ultralytics import YOLO
from . import funcToBoxes as pb


class YOLOv8:
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(YOLOv8, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.labels = {
            0: "__background__",
            1: "a person",
            2: "a bicycle",
            3: "a car",
            4: "a motorcycle",
            5: "a airplane",
            6: "a bus",
            7: "a train",
            8: "a truck",
            9: "a boat",
            10: "a traffic light",
            11: "a fire hydrant",
            12: "a stop sign",
            13: "a parking meter",
            14: "a bench",
            15: "a bird",
            16: "a cat",
            17: "a dog",
            18: "a horse",
            19: "a sheep",
            20: "a cow",
            21: "a elephant",
            22: "a bear",
            23: "a zebra",
            24: "a giraffe",
            25: "a backpack",
            26: "a umbrella",
            27: "a handbag",
            28: "a tie",
            29: "a suitcase",
            30: "a frisbee",
            31: "a skis",
            32: "a snowboard",
            33: "a sports ball",
            34: "a kite",
            35: "a baseball bat",
            36: "a baseball glove",
            37: "a skateboard",
            38: "a surfboard",
            39: "a tennis racket",
            40: "a bottle",
            41: "a wine glass",
            42: "a cup",
            43: "a fork",
            44: "a knife",
            45: "a spoon",
            46: "a bowl",
            47: "a banana",
            48: "a apple",
            49: "a sandwich",
            50: "a orange",
            51: "a broccoli",
            52: "a carrot",
            53: "a hot dog",
            54: "a pizza",
            55: "a donut",
            56: "a cake",
            57: "a chair",
            58: "a couch",
            59: "a potted plant",
            60: "a bed",
            61: "a dining table",
            62: "a toilet",
            63: "a tv",
            64: "a laptop",
            65: "a mouse",
            66: "a remote",
            67: "a keyboard",
            68: "a cell phone",
            69: "a microwave",
            70: "a oven",
            71: "a toaster",
            72: "a sink",
            73: "a refrigerator",
            74: "a book",
            75: "a clock",
            76: "a vase",
            77: "a scissors",
            78: "a teddy bear",
            79: "a hair drier",
            80: "a toothbrush",
        }
        self.model = YOLO("yolov8x.pt")

    def predict(self, image, conf, ln):
        self.image = image
        self.conf = conf
        self.ln = ln
        self.results = self.model(self.image, conf=self.conf)
        self.txt = pb.Text(
            self.image,
            self.results[0].boxes.xyxy,
            self.results[0].boxes.cls.numpy(),
            self.labels,
            self.ln,
        )
        self.image = pb.PlotBoxes(self.image, self.results[0].boxes.boxes)
        return self.txt, self.image
