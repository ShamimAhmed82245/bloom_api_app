from channels.generic.websocket import AsyncWebsocketConsumer
import json
from asgiref.sync import sync_to_async
from .views import get_transformer_prediction, get_model
import torch
import gc
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Create a thread pool for running sync operations
thread_pool = ThreadPoolExecutor(max_workers=2)

class PredictionConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        data = json.loads(text_data)
        text = data.get('text', '')
        model_type = data.get('model_type', '')

        try:
            # Run prediction in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                thread_pool,
                self._predict,
                text,
                model_type
            )
            
            # Send result back to client
            await self.send(text_data=json.dumps(result))
        except Exception as e:
            await self.send(text_data=json.dumps({'error': str(e)}))

    def _predict(self, text, model_type):
        try:
            result = get_transformer_prediction(text, model_type)
            # Force cleanup after prediction
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return result
        except Exception as e:
            return {'error': str(e)}
