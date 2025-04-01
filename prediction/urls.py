from django.urls import path
from .views import PredictBloomLevel

urlpatterns = [
    path('predict/', PredictBloomLevel.as_view(), name='predict'),
]