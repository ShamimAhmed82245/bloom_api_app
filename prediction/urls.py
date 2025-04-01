from django.urls import path
from .views import predict, predict_transformer

urlpatterns = [
    path('predict/', predict, name='predict'),
    path('predict_transformer/', predict_transformer, name='predict_transformer'),
]