from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
import cv2
import base64
import numpy as np
# from yolov8.yolov8 import YOLOv8  # Import the YOLOv8 model class
# Create your views here.


class DetectImage(APIView):

    # Load the YOLOv8 model once when the view is initialized
    # def __init__(self):
    #     self.yolo = YOLOv8()

    def post(self, request):
        data = request.data
        encoded_image = data["image64"]
        # with open("photo_2023-02-24_22-13-37 (2).jpg", "rb") as image_file:
        #     encoded_image = base64.b64encode(image_file.read())   # testing local image and succeeded
        decoded_image = base64.b64decode(encoded_image)
        with open("decoded_image.png", "wb") as fh:
            fh.write(decoded_image)
        img = cv2.imdecode(np.frombuffer(decoded_image, np.uint8), -1)
        print("encoded = ", encoded_image)
        # results = self.yolo.predict(img)
        return Response(
            {
                "result": "in progress",
            }, status=status.HTTP_200_OK
        )

    def get(self, request):
        return Response(
            {
                "about-this-api": "send a post request by your image, and I will respond to you by results"
            }, status=status.HTTP_200_OK
        )
