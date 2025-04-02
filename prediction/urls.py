from django.urls import path
from django.views.generic import TemplateView
from . import views

urlpatterns = [
    path('test/', TemplateView.as_view(template_name='test.html'), name='test'),
    path('predict/', views.predict, name='predict'),
    path('predict_transformer/', views.predict_transformer, name='predict_transformer'),
]