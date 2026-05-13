import os
import torch
import joblib
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from typing import List, Dict

class ModelInference:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sentiment_model = None
        self.sentiment_tokenizer = None
        self.grooming_model = None
        self.grooming_tokenizer = None
        self.music_model = None
        self.music_scaler = None
        self.music_label_encoder = None
        self.music_features = None
        self.author_model = None
        self.author_tokenizer = None
        self.models_loaded = False

    def load_all_models(self):
        try:
            print("📂 Loading DistilBERT for sentiment...")
            self.sentiment_model = DistilBertForSequenceClassification.from_pretrained(
                "models/emotions_distilbert/best_model"
            ).to(self.device)
            self.sentiment_model.eval()
            self.sentiment_tokenizer = DistilBertTokenizer.from_pretrained(
                "models/emotions_distilbert/best_model"
            )
            print("✅ Sentiment model loaded")

            print("📂 Loading RoBERTa for grooming detection...")
            self.grooming_model = RobertaForSequenceClassification.from_pretrained(
                "models/predator_roberta/best_model"
            ).to(self.device)
            self.grooming_model.eval()
            self.grooming_tokenizer = RobertaTokenizer.from_pretrained(
                "models/predator_roberta/best_model"
            )
            print("✅ Grooming model loaded")

            self.models_loaded = True
            print(f"✅ Core models loaded successfully on {self.device}")
        except Exception as e:
            print(f"❌ Error loading core models: {e}")
            self.models_loaded = False

        # Author profiling model — loaded from raw checkpoint (bert-base-uncased, 2 labels)
        try:
            print("📂 Loading BERT for author profiling...")
            from transformers import BertConfig
            config = BertConfig(
                vocab_size=30522, hidden_size=768, num_hidden_layers=12,
                num_attention_heads=12, intermediate_size=3072,
                num_labels=2, id2label={0: "Adult", 1: "Minor"},
                label2id={"Adult": 0, "Minor": 1},
            )
            self.author_model = BertForSequenceClassification(config).to(self.device)
            checkpoint = torch.load(
                "models/author_bert/best_model.pt", map_location=self.device
            )
            self.author_model.load_state_dict(checkpoint["model_state_dict"])
            self.author_model.eval()
            # Tokenizer: try local first, fall back to cached bert-base-uncased
            tokenizer_path = (
                "models/author_bert"
                if os.path.exists("models/author_bert/vocab.txt")
                else "bert-base-uncased"
            )
            self.author_tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
            self.author_label_map = checkpoint.get("id_to_age", {0: "Adult", 1: "Minor"})
            print("✅ Author profiling model loaded")
        except Exception as e:
            print(f"⚠️ Author profiling model not loaded: {e}")
            self.author_model = None
            self.author_tokenizer = None

        # Random Forest music model — optional
        try:
            print("📂 Loading Random Forest music mood model...")
            model_path = "models/rf_music/random_forest_emotion_model.pkl"
            if os.path.exists(model_path):
                self.music_model = joblib.load(model_path)
                print("✅ Random Forest music model loaded")
                for attr, path in [
                    ("music_scaler", "models/rf_music/feature_scaler.pkl"),
                    ("music_label_encoder", "models/rf_music/label_encoder.pkl"),
                    ("music_features", "models/rf_music/feature_columns.pkl"),
                ]:
                    if os.path.exists(path):
                        setattr(self, attr, joblib.load(path))
                        print(f"✅ {attr} loaded")
                    else:
                        print(f"⚠️ {attr} not found at {path}")
            else:
                print(f"⚠️ Music model not found at {model_path}")
        except Exception as e:
            print(f"⚠️ Music model load error: {e}")
            self.music_model = None

    def analyze_sentiment(self, text: str) -> Dict:
        if not self.models_loaded:
            return {"error": "Models not loaded"}

        inputs = self.sentiment_tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]

        emotion_labels = self._get_emotion_labels()
        emotion_scores = {label: float(probs[i]) for i, label in enumerate(emotion_labels)}

        positive_emotions = ["joy", "love", "optimism", "gratitude", "admiration", "approval", "excitement", "relief", "caring"]
        negative_emotions = ["sadness", "anger", "fear", "disgust", "disappointment", "embarrassment", "grief", "remorse", "annoyance", "nervousness"]

        # Clamp sums to [0,1] — sigmoid outputs can sum above 1 for multi-label models
        pos_score = min(1.0, sum(emotion_scores.get(e, 0) for e in positive_emotions))
        neg_score = min(1.0, sum(emotion_scores.get(e, 0) for e in negative_emotions))

        sentiment = (pos_score + (1.0 - neg_score)) / 2.0
        primary_emotion = max(emotion_scores, key=emotion_scores.get)

        return {
            "sentiment_score": float(sentiment),
            "primary_emotion": primary_emotion,
            "emotion_scores": emotion_scores,
            "confidence": float(max(emotion_scores.values()))
        }

    def detect_grooming(self, messages: List[str]) -> Dict:
        if not self.models_loaded:
            return {"error": "Models not loaded"}

        conversation = " [SEP] ".join(messages)

        inputs = self.grooming_tokenizer(
            conversation,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.grooming_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        grooming_prob = float(probs[1])
        risk_level = "high_risk" if grooming_prob > 0.7 else "suspicious" if grooming_prob > 0.3 else "safe"

        return {
            "grooming_probability": grooming_prob,
            "risk_level": risk_level,
            "detected_stage": None,
            "explanation": self._generate_grooming_explanation(grooming_prob, messages)
        }

    def classify_music_mood(self, features: Dict[str, float]) -> Dict:
        if self.music_model is None:
            return {"error": "Music mood model not available"}

        try:
            feature_array = np.array([[
                float(features.get("valence", 0.5)),
                float(features.get("energy", 0.5)),
                float(features.get("danceability", 0.5)),
                float(features.get("acousticness", 0.5)),
                float(features.get("instrumentalness", 0)),
                float(features.get("tempo", 120)),
                float(features.get("loudness", -10)),
                float(features.get("speechiness", 0)),
                float(features.get("liveness", 0.1)),
                float(features.get("key", 5)),
                float(features.get("mode", 1))
            ]])

            prediction = self.music_model.predict(feature_array)[0]
            probabilities = self.music_model.predict_proba(feature_array)[0]

            prediction_int = int(prediction)
            confidence_float = float(max(probabilities))

            mood_map = {
                0: "calm", 1: "chill", 2: "energetic", 3: "focus",
                4: "happy", 5: "party", 6: "romantic", 7: "sad"
            }
            mood = mood_map.get(prediction_int, "unknown")

            return {"mood": mood, "confidence": confidence_float, "prediction": prediction_int}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"Music classification failed: {str(e)}"}

    def predict_author(self, text: str) -> Dict:
        if self.author_model is None:
            return {"error": "Author profiling model not loaded"}

        inputs = self.author_tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.author_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        predicted_class = int(np.argmax(probs))
        label_map = getattr(self, "author_label_map", {0: "Adult", 1: "Minor"})
        return {
            "predicted_class": predicted_class,
            "label": label_map.get(predicted_class, str(predicted_class)),
            "confidence": float(max(probs)),
            "probabilities": probs.tolist()
        }

    def _get_emotion_labels(self) -> list:
        return [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring",
            "confusion", "curiosity", "desire", "disappointment", "disapproval",
            "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
            "joy", "love", "nervousness", "optimism", "pride", "realization",
            "relief", "remorse", "sadness", "surprise", "neutral"
        ]

    def analyze_message_all_models(self, text: str) -> Dict:
        """
        Run DistilBERT + RoBERTa + BERT on a message.
        Saved sentiment_score = weighted mix of BERT (adult-prob) + RoBERTa (safe-prob).
        DistilBERT supplies the emotion_vector only — it is NOT included in the saved score.
        """
        if not self.models_loaded:
            return {"error": "Models not loaded"}

        # 1. DistilBERT → 28-emotion vector (display only)
        distilbert_result = self.analyze_sentiment(text)
        emotion_vector: dict = {}
        primary_emotion = "neutral"
        distilbert_sentiment: float = 0.5
        if "error" not in distilbert_result:
            emotion_vector = {k: float(v) for k, v in distilbert_result.get("emotion_scores", {}).items()}
            primary_emotion = distilbert_result.get("primary_emotion", "neutral")
            distilbert_sentiment = float(distilbert_result.get("sentiment_score", 0.5))

        # 2. RoBERTa → grooming probability
        roberta_result = self.detect_grooming([text])
        grooming_prob = 0.0
        stage_label = None
        human_reason = None
        if "error" not in roberta_result:
            grooming_prob = float(roberta_result.get("grooming_probability", 0.0))
            stage_label = roberta_result.get("detected_stage")
            human_reason = roberta_result.get("explanation")
        roberta_safe_score = 1.0 - grooming_prob  # higher = safer

        # 3. BERT → author profiling (Adult vs Minor)
        bert_result = self.predict_author(text)
        bert_adult_prob = 1.0
        bert_label = "Adult"
        if "error" not in bert_result:
            probs = bert_result.get("probabilities", [1.0, 0.0])
            bert_adult_prob = float(probs[0]) if probs else 1.0
            bert_label = bert_result.get("label", "Adult")

        # Mixed score saved to DB: 60% RoBERTa safety + 40% BERT adult-probability
        # Higher = safer conversation (not grooming, talking to an adult)
        mixed_sentiment_score = 0.6 * roberta_safe_score + 0.4 * bert_adult_prob

        return {
            "sentiment_score":     float(mixed_sentiment_score),   # BERT+RoBERTa mix
            "grooming_prob":       grooming_prob,
            "stage_label":         stage_label,
            "human_reason":        human_reason,
            "emotion_vector":      emotion_vector,
            "primary_emotion":     primary_emotion,
            "bert_author_label":   bert_label,
            "bert_adult_prob":     float(bert_adult_prob),
            "roberta_safe_score":  float(roberta_safe_score),
            "model_scores": {
                "distilbert_sentiment": distilbert_sentiment,
                "roberta_safe":         float(roberta_safe_score),
                "bert_adult":           float(bert_adult_prob),
            },
        }

    def _generate_grooming_explanation(self, probability: float, messages: List[str]) -> str:
        if probability > 0.7:
            return "High grooming risk detected. The conversation shows multiple predatory patterns."
        elif probability > 0.3:
            return "Suspicious patterns detected. Monitor this conversation closely."
        else:
            return "No grooming patterns detected."


inference_engine = ModelInference()
