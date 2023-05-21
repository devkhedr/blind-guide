from ultralytics import YOLO
import FuncToBoxes as pb
class YOLOv8():
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(YOLOv8, cls).__new__(cls)
        return cls.instance
    def __init__(self):
        self.labels = {0:'__backgrond__',1:'person',2:'bicycle',
          3:'car',4:'motorcycle',5:'airplane',6:'bs',
          7:'train',8:'trck',9:'boat',10:'traffic light',
          11:'fire hydrant',12:'stop sign',13:'parking meter',
          14:'bench',15:'bird',16:'cat',17:'dog',18:'horse',
          19:'sheep',20:'cow',21:'elephant',22:'bear',23:'zebra',
          24:'giraffe',25:'backpack',26:'mbrella',27:'handbag',
          28:'tie',29:'sitcase',30:'frisbee',31:'skis',32:'snowboard',
          33:'sports ball',34:'kite',35:'baseball bat',36:'baseball glove',
          37:'skateboard',38:'srfboard',39:'tennis racket',40:'bottle',
          41:'wine glass',42:'cp',43:'fork',44:'knife',45:'spoon',
          46:'bowl',47:'banana',48:'apple',49:'sandwich',50:'orange',
          51:'broccoli',52:'carrot',53:'hot dog',54:'pizza',55:'dont',
          56:'cake',57:'chair',58:'coch',59:'potted plant',60:'bed',
          61:'dining table',62:'toilet',63:'tv',64:'laptop',65:'mose',
          66:'remote',67:'keyboard',68:'cell phone',69:'microwave',
          70:'oven',71:'toaster',72:'sink',73:'refrigerator',
          74:'book',75:'clock',76:'vase',77:'scissors',
          78:'teddy bear',79:'hair drier',80:'toothbrsh'}
        self.colors = [
          (89, 161, 197),(67, 161, 255),(19, 222, 24),(186, 55, 2),(167, 146, 11),(190, 76, 98),(130, 172, 179),
          (115, 209, 128),(204, 79, 135),(136, 126, 185),(209, 213, 45),(44, 52, 10),(101, 158, 121),(179, 124, 12),
          (25, 33, 189),(45, 115, 11),(73, 197, 184),(62, 225, 221),(32, 46, 52),(20, 165, 16),(54, 15, 57),
          (12, 150, 9),(10, 46, 99),(94, 89, 46),(48, 37, 106),(42, 10, 96),(7, 164, 128),(98, 213, 120),
          (40, 5, 219),(54, 25, 150),(251, 74, 172),(0, 236, 196),(21, 104, 190),(226, 74, 232),
          (120, 67, 25),(191, 106, 197),(8, 15, 134),(21, 2, 1),(142, 63, 109),(133, 148, 146),(187, 77, 253),
          (155, 22, 122),(218, 130, 77),(164, 102, 79),(43, 152, 125),(185, 124, 151),(95, 159, 238),
          (128, 89, 85),(228, 6, 60),(6, 41, 210),(11, 1, 133),(30, 96, 58),(230, 136, 109),(126, 45, 174),
          (164, 63, 165),(32, 111, 29),(232, 40, 70),(55, 31, 198),(148, 211, 129),(10, 186, 211),
          (181, 201, 94),(55, 35, 92),(129, 140, 233),(70, 250, 116),(61, 209, 152),(216, 21, 138),
          (100, 0, 176),(3, 42, 70),(151, 13, 44),(216, 102, 88),(125, 216, 93),(171, 236, 47),(253, 127, 103),
          (205, 137, 244),(193, 137, 224),(36, 152, 214),(17, 50, 238),(154, 165, 67),(114, 129, 60),
          (119, 24, 48),(73, 8, 110)]
        self.side=['on the top left','on the top','on the top right','on the left','on the middle','on the right','on the bottom left','on the bottom','on the bottom right']
        self.model=YOLO('yolov8x.pt')

    def predict(self,image,conf):
        self.image=image
        self.conf=conf
        self.results= self.model(self.image,conf=self.conf)
        self.image=pb.PlotBoxes(self.image,
                                self.results[0].boxes.boxes)
        self.results=self.results[0].cpu()
        self.text=pb.Text(self.image,
                      self.results.boxes.xyxy.numpy(),
                      self.results.boxes.cls.numpy(),
                      self.labels,
                      self.side)
        
        return self.text,self.image,self.results[0].boxes.boxes.numpy().tolist()
