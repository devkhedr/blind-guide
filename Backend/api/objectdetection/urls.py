from django.urls import path
from . import views

urlpatterns = [
    path('detect-image/', views.DetectImage.as_view(), name='detect_image'),
]