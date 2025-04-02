"""
ASGI config for bloom_api project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/howto/deployment/asgi/
"""

import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'bloom_api.settings')
django.setup()

from channels.routing import ProtocolTypeRouter, URLRouter
from django.urls import path
from prediction.consumers import PredictionConsumer
from django.core.asgi import get_asgi_application

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": URLRouter([
        path('ws/predict/', PredictionConsumer.as_asgi()),
    ]),
})
