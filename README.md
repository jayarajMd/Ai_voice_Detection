# 🎙️ AI Voice Detection API

**REST API for detecting AI-generated vs Human voices across 5 languages: Tamil, English, Hindi, Malayalam, Telugu**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.70+-green.svg)](https://fastapi.tiangolo.com/)

---

## 🌟 Overview

This API detects whether a voice sample is **AI-generated** or **Human** using 9 independent detection techniques including deep learning (Wav2Vec2) and advanced fingerprinting.

### Supported Languages
- Tamil
- English  
- Hindi
- Malayalam
- Telugu

---

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
# Create virtual environment (optional)
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the API Server

```powershell
python api.py
```

The server will start at `http://localhost:8000`

### 3. Test the API

```powershell
# In another terminal
python test_api.py
```

---

## 📡 API Endpoint

### POST `/api/voice-detection`

Analyzes a Base64-encoded MP3 audio file and determines if the voice is AI-generated or Human.

**URL:** `http://localhost:8000/api/voice-detection`

---

## 🔐 Authentication

All requests must include an API key in the header:

```
x-api-key: sk_test_123456789
```

Requests without a valid API key will be rejected with a 401 error.

---

## 📥 Request Format

### Headers
```
Content-Type: application/json
x-api-key: YOUR_API_KEY
```

### Body (JSON)
```json
{
    "language": "English",
    "audioFormat": "mp3",
    "audioBase64": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMAAAAAAA..."
}
```

### Request Fields

| Field | Type | Description |
|-------|------|-------------|
| `language` | string | One of: Tamil, English, Hindi, Malayalam, Telugu |
| `audioFormat` | string | Must be "mp3" |
| `audioBase64` | string | Base64-encoded MP3 audio data |

---

## 📤 Response Format

### Success Response
```json
{
    "status": "success",
    "language": "English",
    "classification": "AI_GENERATED",
    "confidenceScore": 0.85,
    "explanation": "Unnatural pitch consistency and robotic speech patterns detected"
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | "success" or "error" |
| `language` | string | Language of the audio |
| `classification` | string | "AI_GENERATED" or "HUMAN" |
| `confidenceScore` | float | Value between 0.0 and 1.0 |
| `explanation` | string | Short reason for the decision |

### Error Response
```json
{
    "status": "error",
    "message": "Invalid API key or malformed request"
}
```

---

## 🧪 cURL Example

```bash
curl -X POST http://localhost:8000/api/voice-detection \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk_test_123456789" \
  -d '{
    "language": "Tamil",
    "audioFormat": "mp3",
    "audioBase64": "YOUR_BASE64_ENCODED_MP3_DATA"
  }'
```

---

## 🐍 Python Example

```python
import requests
import base64

# Read and encode audio file
with open("audio.mp3", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode("utf-8")

# Make request
response = requests.post(
    "http://localhost:8000/api/voice-detection",
    headers={
        "Content-Type": "application/json",
        "x-api-key": "sk_test_123456789"
    },
    json={
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": audio_base64
    }
)

result = response.json()
print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidenceScore']}")
print(f"Explanation: {result['explanation']}")
```

---

## 📁 Project Structure

```
Voice_Detection/
├── api.py                    # REST API server (FastAPI)
├── ai_voice_detector.py      # Core detection engine
├── test_api.py               # API test script
├── requirements.txt          # Dependencies
├── ai_voices/                # Sample AI voice files
│   ├── ai_english_1.mp3
│   ├── ai_hindi_1.mp3
│   ├── ai_tamil_1.mp3
│   └── ...
└── README.md                 # This file
```

---

## 🔧 Configuration

### Change API Key

Set the `API_KEY` environment variable or modify in `api.py`:

```python
API_KEY = "your_secret_key_here"
```

Or set environment variable:
```powershell
$env:API_KEY = "your_secret_key_here"
python api.py
```

### Change Port

Modify the port in `api.py`:
```python
uvicorn.run("api:app", host="0.0.0.0", port=8080)
```

---

## 🧠 Detection Methods

The system uses 9 detection techniques:

### Traditional Tests (43%)
1. **Pitch Stability** (3%) - Analyzes voice naturalness
2. **Spectral Smoothness** (7%) - Detects frequency patterns
3. **Phase Artifacts** (8%) - Finds phase inconsistencies
4. **Noise Randomness** (22%) - Measures noise entropy
5. **Deep Learning** (3%) - Wav2Vec2 embeddings

### Fingerprinting Tests (57%)
6. **Micro-Artifacts** (12%) - Neural synthesis glitches
7. **Prosody Fingerprint** (45%) - Controlled rhythm/intonation

---

## ⚠️ Error Codes

| Status Code | Meaning |
|-------------|---------|
| 200 | Success |
| 400 | Bad Request (invalid input) |
| 401 | Unauthorized (invalid/missing API key) |
| 500 | Internal Server Error |

---

## 🚀 Deployment

### Local Development
```powershell
python api.py
```

### Production (with Uvicorn)
```powershell
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker (optional)
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 📝 License

MIT License

---

## 👤 Author

AI Voice Detection System - Built for GUVI Challenge
