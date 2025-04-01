from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
import os
import gc
import time
from threading import Lock
from functools import lru_cache
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Model paths configuration
MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml_models')

# Lazy loading for ML models
def load_model(model_name):
    try:
        return joblib.load(os.path.join(MODEL_PATH, model_name))
    except FileNotFoundError:
        raise ImproperlyConfigured(f"Model file '{model_name}' not found in {MODEL_PATH}")

def get_model(model_type):
    if model_type not in models:
        model_files = {
            'knn': 'bt_tfidf_KNN_model.pkl',
            'multinomial_nb': 'bt_tfidf_MultinomialNB_model.pkl',
            'rf_ngram': 'random_forest_ngram_model.pkl',
            'xgb_ngram': 'xgboost_ngram_model.pkl',
            'svm_ngram': 'svm_ngram_model.pkl',
            'nb_trigram': 'naive_bayes_tri_gram_model.pkl',
            'nb_ngram': 'naive_bayes_ngram_model.pkl',
            'lr_trigram': 'logistic_regression_tri_gram_model.pkl',
            'lr_ngram': 'logistic_regression_ngram_model.pkl'
        }
        if model_type in model_files:
            models[model_type] = load_model(model_files[model_type])
    return models.get(model_type)

# Initialize empty dictionary for lazy loading
models = {}

# Transformer model configurations
TRANSFORMER_PATHS = {
    'bert': {
        'path': os.path.join(MODEL_PATH, 'bert_classifier'),
        'model_file': 'bloom_classifier_bert.pth',
        'config_path': os.path.join(MODEL_PATH, 'bert_classifier', 'config.json'),
        'vocab_path': os.path.join(MODEL_PATH, 'bert_classifier', 'vocab.txt')
    },
    'distilbert': {
        'path': os.path.join(MODEL_PATH, 'distilbert_classifier'),
        'model_file': 'distilbert_bloom_classifier.pth',
        'config_path': os.path.join(MODEL_PATH, 'distilbert_classifier', 'config.json'),
        'vocab_path': os.path.join(MODEL_PATH, 'distilbert_classifier', 'vocab.txt')
    },
    'roberta': {
        'path': os.path.join(MODEL_PATH, 'roberta_classifier'),
        'model_file': 'roberta_bloom_classifier.pth',
        'config_path': os.path.join(MODEL_PATH, 'roberta_classifier', 'config.json'),
        'vocab_path': os.path.join(MODEL_PATH, 'roberta_classifier', 'vocab.json')
    }
}

# Transformer model management
model_locks = {model: Lock() for model in TRANSFORMER_PATHS}
last_used = {}
model_cache = {}
CACHE_TIMEOUT = 300

@api_view(['POST'])
def predict(request):
    try:
        text = request.data.get('text', '')
        model_type = request.data.get('model_type', 'rf_ngram')

        # Handle traditional ML models
        if model_type in ['knn', 'multinomial_nb', 'rf_ngram', 'xgb_ngram', 'svm_ngram', 
                         'nb_trigram', 'nb_ngram', 'lr_trigram', 'lr_ngram']:
            
            model = get_model(model_type)
            if not model:
                return Response({'error': f'Model {model_type} not found'}, status=500)

            # Get prediction and probability
            prediction = model.predict([text])[0]
            probability = model.predict_proba([text])[0].max()

            return Response({
                'prediction': int(prediction),
                'probability': float(probability),
                'model_used': model_type
            })
            
        # Handle transformer models
        elif model_type in TRANSFORMER_PATHS:
            return predict_transformer(request)
            
        else:
            return Response({'error': 'Invalid model type'}, status=400)

    except Exception as e:
        return Response({'error': str(e)}, status=500)

# Transformer model helper functions
@lru_cache(maxsize=3)
def load_tokenizer(model_type):
    try:
        config = TRANSFORMER_PATHS[model_type]
        if not os.path.exists(config['path']):
            raise ImproperlyConfigured(f"Model directory not found: {config['path']}")
        if not os.path.exists(config['config_path']):
            raise ImproperlyConfigured(f"Model config not found: {config['config_path']}")
        if not os.path.exists(config['vocab_path']):
            raise ImproperlyConfigured(f"Model vocabulary not found: {config['vocab_path']}")
        
        return AutoTokenizer.from_pretrained(config['path'])
    except Exception as e:
        raise ImproperlyConfigured(f"Error loading tokenizer for {model_type}: {str(e)}")

def load_transformer_model(model_type):
    current_time = time.time()
    
    try:
        if model_type in model_cache and current_time - last_used.get(model_type, 0) < CACHE_TIMEOUT:
            last_used[model_type] = current_time
            return model_cache[model_type]
        
        config = TRANSFORMER_PATHS[model_type]
        model_path = os.path.join(config['path'], config['model_file'])
        
        if not os.path.exists(model_path):
            raise ImproperlyConfigured(f"Model file not found: {model_path}")
        
        if model_type in model_cache:
            del model_cache[model_type]
            torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
        
        model = AutoModelForSequenceClassification.from_pretrained(config['path'])
        state_dict = torch.load(model_path)
        
        # Handle BERT's mismatched layer names
        if model_type == 'bert':
            # Rename fc layers to classifier layers
            if 'fc.weight' in state_dict:
                state_dict['classifier.weight'] = state_dict.pop('fc.weight')
            if 'fc.bias' in state_dict:
                state_dict['classifier.bias'] = state_dict.pop('fc.bias')
        
        model.load_state_dict(state_dict)
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        model.eval()
        model_cache[model_type] = model
        last_used[model_type] = current_time
        
        return model
    except Exception as e:
        raise ImproperlyConfigured(f"Error loading model {model_type}: {str(e)}")

def clean_unused_models():
    current_time = time.time()
    for model_type in list(model_cache.keys()):
        if current_time - last_used.get(model_type, 0) > CACHE_TIMEOUT:
            with model_locks[model_type]:
                if model_type in model_cache:
                    del model_cache[model_type]
                    torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()

# New endpoint specifically for transformer models
@api_view(['POST'])
def predict_transformer(request):
    try:
        text = request.data.get('text', '')
        model_type = request.data.get('model_type', 'bert')
        
        if model_type not in TRANSFORMER_PATHS:
            return Response({'error': 'Invalid transformer model type'}, status=400)
        
        clean_unused_models()
        
        with model_locks[model_type]:
            tokenizer = load_tokenizer(model_type)
            model = load_transformer_model(model_type)
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            try:
                with torch.no_grad():
                    outputs = model(**inputs)
                    predictions = torch.softmax(outputs.logits, dim=1)
                    predicted_class = torch.argmax(predictions).item()
                    confidence = predictions[0][predicted_class].item()
                
                return Response({
                    'class': predicted_class,
                    'confidence': confidence,
                    'model_used': model_type
                })
            except Exception as e:
                if model_type in model_cache:
                    del model_cache[model_type]
                    torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
                raise e
    except Exception as e:
        return Response({'error': str(e)}, status=500)