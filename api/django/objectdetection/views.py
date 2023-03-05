from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
import cv2
import base64
from .YOLOv8 import YOLOv8
from django.views import View
# Create your views here.


class DetectImage(APIView):
    def __init__(self):
        self.yolo = YOLOv8()  # self.yolo is a singleton and will be loaded
                              # once when using the API for the first time.

    def post(self, request):
        data = request.data
        encoded_image = data["image64"]
        decoded_image = base64.b64decode(encoded_image)
        with open("/content/drive/MyDrive/api/decoded_image.png", "wb") as fh:
            fh.write(decoded_image)
        image = cv2.imread("/content/drive/MyDrive/api/decoded_image.png")
        language = "en"
        conf = 0.7
        txt, image = self.yolo.predict(image, conf, language)
        retval, buffer = cv2.imencode(".jpg", image)
        encoded_processed_image = base64.b64encode(buffer)
        decoded_processed_image = base64.b64decode(encoded_processed_image)
        with open("/content/drive/MyDrive/api/decoded_processed_image.png", "wb") as fh:
            fh.write(decoded_processed_image)
        return Response(
            {"list": txt},
            status=status.HTTP_200_OK,
        )

    def get(self, request):
        return Response(
            {
                "about-this-api": "send a post request by your image, and I will respond to you by results"
            },
            status=status.HTTP_200_OK,
        )
