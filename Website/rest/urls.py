from django.urls import path
from .views import UploadView

app_name = "api"
urlpatterns = [
    path('upload/', UploadView.as_view(), name="upload")
]
