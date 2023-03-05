from django.contrib import admin
from django.urls import path, include
from django.views.debug import default_urlconf

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", default_urlconf),
    path("detect-image/", include("objectdetection.urls")),
]
