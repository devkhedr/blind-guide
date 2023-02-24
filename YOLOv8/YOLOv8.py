from ultralytics import YOLO
import funcToBoxes as pb
class YOLOv8():
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(YOLOv8, cls).__new__(cls)
        return cls.instance
    def __init__(self):
        self.labels = {0: u'__background__', 1: u'a person', 2: u'a bicycle',
          3: u'a car', 4: u'a motorcycle', 5: u'a airplane', 6: u'a bus',
          7: u'a train', 8: u'a truck', 9: u'a boat', 10: u'a traffic light',
          11: u'a fire hydrant', 12: u'a stop sign', 13: u'a parking meter',
          14: u'a bench', 15: u'a bird', 16: u'a cat', 17: u'a dog', 18: u'a horse',
          19: u'a sheep', 20: u'a cow', 21: u'a elephant', 22: u'a bear', 23: u'a zebra',
          24: u'a giraffe', 25: u'a backpack', 26: u'a umbrella', 27: u'a handbag',
          28: u'a tie', 29: u'a suitcase', 30: u'a frisbee', 31: u'a skis', 32: u'a snowboard',
          33: u'a sports ball', 34: u'a kite', 35: u'a baseball bat', 36: u'a baseball glove',
          37: u'a skateboard', 38: u'a surfboard', 39: u'a tennis racket', 40: u'a bottle',
          41: u'a wine glass', 42: u'a cup', 43: u'a fork', 44: u'a knife', 45: u'a spoon',
          46: u'a bowl', 47: u'a banana', 48: u'a apple', 49: u'a sandwich', 50: u'a orange',
          51: u'a broccoli', 52: u'a carrot', 53: u'a hot dog', 54: u'a pizza', 55: u'a donut',
          56: u'a cake', 57: u'a chair', 58: u'a couch', 59: u'a potted plant', 60: u'a bed',
          61: u'a dining table', 62: u'a toilet', 63: u'a tv', 64: u'a laptop', 65: u'a mouse',
          66: u'a remote', 67: u'a keyboard', 68: u'a cell phone', 69: u'a microwave',
          70: u'a oven', 71: u'a toaster', 72: u'a sink', 73: u'a refrigerator',
          74: u'a book', 75: u'a clock', 76: u'a vase', 77: u'a scissors',
          78: u'a teddy bear', 79: u'a hair drier', 80: u'a toothbrush'}
        self.model=YOLO('yolov8x.pt')
    def predict(self,image,conf,ln):
        self.image=image
        self.conf=conf
        self.ln=ln
        self.results= self.model(self.image,conf=self.conf)
        self.txt=pb.Text(self.image,
                      self.results[0].boxes.xyxy,
                      self.results[0].boxes.cls.numpy(),
                      self.labels,
                      self.ln)
        self.image=pb.PlotBoxes(self.image,
                                self.results[0].boxes.boxes)
        return self.txt,self.image