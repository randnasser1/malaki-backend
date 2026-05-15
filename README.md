# Malaki Backend

FastAPI backend for Malaki AI Child Guardian. Provides grooming detection, sentiment analysis, music mood classification, and behavioral anomaly detection.

## Tech Stack

- FastAPI (Python 3.11)
- Firebase Admin SDK (Firestore)
- Transformers (Hugging Face)
- Spotipy (Spotify API)
- Jina Reader
- RapidAPI

## Setup

1. Clone repository

2. Install dependencies
pip install -r requirements.txt

3. Create .env file
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret

4. Place service-account.json in root directory (Firebase credentials)

5. Run backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /events/analyze | POST | Analyze messages for grooming patterns |
| /wellbeing/daily | POST | Process journal entries with DistilBERT |
| /music/process/{child_id} | POST | Classify music emotions using Random Forest |
| /analyze/tbats/{child_id} | GET | Detect behavioral anomalies with TBATS |
| /health | GET | Health check |

## Model Files

Place trained models in app/models/ directory:
- roberta_grooming/
- bert_author_profile/
- random_forest_music.pkl
- distilbert_emotion/

## Environment Variables

| Variable | Description |
|----------|-------------|
| SPOTIFY_CLIENT_ID | Spotify API client ID |
| SPOTIFY_CLIENT_SECRET | Spotify API client secret |
| GOOGLE_APPLICATION_CREDENTIALS | Path to Firebase service account |