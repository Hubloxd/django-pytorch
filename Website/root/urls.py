from django.urls import path
from .views import index

app_name = 'root'
urlpatterns = [
    path(r'', index, name='index'),
]
