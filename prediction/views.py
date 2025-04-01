import os
import joblib
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import QuestionSerializer

# Define model paths
MODEL_PATHS = {
    'bt_tfidf_KNN_model': os.path.join(settings.BASE_DIR, 'ml_models', 'bt_tfidf_KNN_model.pkl'),
    'bt_tfidf_MultinomialNB_model': os.path.join(settings.BASE_DIR, 'ml_models', 'bt_tfidf_MultinomialNB_model.pkl'),
    'random_forest_ngram_model': os.path.join(settings.BASE_DIR, 'ml_models', 'random_forest_ngram_model.pkl'),
    'xgboost_ngram_model': os.path.join(settings.BASE_DIR, 'ml_models', 'xgboost_ngram_model.pkl'),
    'svm_ngram_model': os.path.join(settings.BASE_DIR, 'ml_models', 'svm_ngram_model.pkl'),
    'naive_bayes_tri_gram_model': os.path.join(settings.BASE_DIR, 'ml_models', 'naive_bayes_tri_gram_model.pkl'),
    'naive_bayes_ngram_model': os.path.join(settings.BASE_DIR, 'ml_models', 'naive_bayes_ngram_model.pkl'),
    'logistic_regression_tri_gram_model': os.path.join(settings.BASE_DIR, 'ml_models', 'logistic_regression_tri_gram_model.pkl'),
    'logistic_regression_ngram_model': os.path.join(settings.BASE_DIR, 'ml_models', 'logistic_regression_ngram_model.pkl'),
}

class PredictBloomLevel(APIView):
    _models_loaded = False
    _models = {}

    @classmethod
    def _load_models(cls):
        if not cls._models_loaded:
            for model_name, model_path in MODEL_PATHS.items():
                try:
                    with open(model_path, 'rb') as model_file:
                        cls._models[model_name] = joblib.load(model_file)
                except FileNotFoundError:
                    raise RuntimeError(f"Model file not found: {model_path}")
                except Exception as e:
                    raise RuntimeError(f"Error loading model {model_name}: {e}")
            cls._models_loaded = True

    def post(self, request):
        self._load_models()
        serializer = QuestionSerializer(data=request.data)
        if serializer.is_valid():
            question = serializer.validated_data['question']
            predictions = {}
            for model_name, model in self._models.items():
                try:
                    predictions[model_name] = model.predict([question])[0]
                except Exception as e:
                    predictions[model_name] = f"Error: {e}"
            return Response(predictions)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)