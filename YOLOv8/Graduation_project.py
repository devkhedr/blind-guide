import cv2
import numpy as np
def DivideImageIntoBox(H,W):
    XAxis=np.linspace(0,W,4,dtype=np.int32)
    YAxis=np.linspace(0,H,4,dtype=np.int32)
    b1=np.array((XAxis[0],YAxis[0],XAxis[1],YAxis[1]))#TL
    b2=np.array((XAxis[1],YAxis[0],XAxis[2],YAxis[1]))#T
    b3=np.array((XAxis[2],YAxis[0],XAxis[3],YAxis[1]))#TR
    b4=np.array((XAxis[0],YAxis[1],XAxis[1],YAxis[2]))#L
    b5=np.array((XAxis[1],YAxis[1],XAxis[2],YAxis[2]))#C
    b6=np.array((XAxis[2],YAxis[1],XAxis[3],YAxis[2]))#R
    b7=np.array((XAxis[0],YAxis[2],XAxis[1],YAxis[3]))#BL
    b8=np.array((XAxis[1],YAxis[2],XAxis[2],YAxis[3]))#B
    b9=np.array((XAxis[2],YAxis[2],XAxis[3],YAxis[3]))#BR
    BB=np.array((b1,b2,b3,b4,b5,b6,b7,b8,b9))
    return BB
def showDivideImageIntoBox(image):
	img=image
	H,W=img.shape[:2]
	for i in DivideImageIntoBox(H,W):
    		p1,p2=(i[:2],i[2:])
    		cv2.rectangle(img, p1, p2, (0,0,0), 0, cv2.LINE_AA)
	cv2.imshow('DivideImageIntoBox',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
def resize(image,scale=0.5):
	img=image
	height=int(img.shape[0]*(scale))
	width=int(img.shape[1]*(scale))
	img1=cv2.resize(img,(width,height),interpolation=cv2.INTER_AREA)
	return img1

def iou(box1, box2):
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2
    xi1 = max(box1[0],box2[0])
    yi1 = max(box1[1],box2[1])
    xi2 = min(box1[2],box2[2])
    yi2 = min(box1[3],box2[3])
    inter_width =  max((xi2 - xi1),0)
    inter_height = max((yi2 - yi1),0)
    inter_area = inter_height*inter_width
    box1_area = (box1[3] - box1[1])*(box1[2] - box1[0])
    box2_area = (box2[3] - box2[1])*(box2[2] - box2[0])
    union_area = (box1_area + box2_area) - inter_area
    iou = inter_area / union_area
    
    return iou