from django.shortcuts import render
from django.views import View
from rest_framework.response import Response
from rest_framework.views import APIView
# Create your views here.


class DetectImage(APIView):
    def post(self, request, *args, **kwargs):
        data = request.data
        print(data["image"])
        return Response(
            {
                'result': 'in progress',
                "image": data["image"]
            }
        )

    def get(self, request, *args, **kwargs):
        return Response(
            {
                'about-this-api': 'send a post request by your image, and I will respond to you by results',
            }
        )
