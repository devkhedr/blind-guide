from django.shortcuts import render
from django.views import View
from rest_framework.response import Response
from rest_framework.views import APIView
import requests
# Create your views here.


class DetectImage(APIView):
    def post(self, request, *args, **kwargs):
        data = request.data
        print(data["image"])
        image_data = data['image']
        # Run the image through the YOLOv8 model and get back the results 
        return Response(
            {
                "result": "in progress",
                "image": image_data
            }
        )

    def get(self, request, *args, **kwargs):
        return Response(
            {
                "about-this-api": "send a post request by your image, and I will respond to you by results"
            }
        )
