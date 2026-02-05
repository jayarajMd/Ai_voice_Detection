# 🚀 AI Voice Detection API - Deployment Guide

## Quick Deployment Options

### Option 1: Railway (Recommended - Easiest) ⭐

**Railway** offers $5 free credits/month and easy deployment.

#### Steps:
1. **Create Account**: Go to [railway.app](https://railway.app) and sign up with GitHub
2. **Create New Project**: Click "New Project" → "Deploy from GitHub repo"
3. **Connect Repository**: 
   - Push your code to GitHub first
   - Select your repository
4. **Set Environment Variables**:
   - Go to your project → "Variables" tab
   - Add: `API_KEY` = `your_secret_api_key_here` (e.g., `sk_prod_abc123xyz789`)
   - Add: `PORT` = `8000`
5. **Deploy**: Railway will automatically build and deploy
6. **Get URL**: Once deployed, go to "Settings" → "Domains" → Generate domain

**Your endpoint will be**: `https://your-app.railway.app/api/voice-detection`

---

### Option 2: Render (Free Tier Available)

**Render** offers free hosting with some cold start delays.

#### Steps:
1. **Create Account**: Go to [render.com](https://render.com) and sign up
2. **New Web Service**: Click "New" → "Web Service"
3. **Connect Repository**: Connect your GitHub repo
4. **Configure**:
   - Name: `ai-voice-detection-api`
   - Environment: `Docker`
   - Plan: `Starter` (or Free)
5. **Environment Variables**:
   - Add: `API_KEY` = `your_secret_api_key_here`
6. **Deploy**: Click "Create Web Service"

**Your endpoint will be**: `https://ai-voice-detection-api.onrender.com/api/voice-detection`

---

### Option 3: Google Cloud Run (Production-Grade)

#### Prerequisites:
- Google Cloud account
- `gcloud` CLI installed

#### Steps:
```bash
# Login to Google Cloud
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Build and deploy
gcloud run deploy ai-voice-detection \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars API_KEY=your_secret_api_key_here \
  --memory 2Gi \
  --timeout 300
```

---

## 📋 What to Share with Judges

Once deployed, share these details:

### API Endpoint
```
POST https://your-deployed-url.com/api/voice-detection
```

### API Key
```
x-api-key: your_secret_api_key_here
```

### Example Request (cURL)
```bash
curl -X POST https://your-deployed-url.com/api/voice-detection \
  -H "Content-Type: application/json" \
  -H "x-api-key: your_secret_api_key_here" \
  -d '{
    "language": "English",
    "audioFormat": "mp3",
    "audioBase64": "BASE64_ENCODED_AUDIO_HERE"
  }'
```

### Example Response
```json
{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.85,
  "explanation": "Synthetic noise patterns and robotic speech rhythm detected"
}
```

---

## 🔧 Pre-Deployment Checklist

Before deploying, ensure:

- [ ] Code is pushed to GitHub
- [ ] `requirements.txt` is up to date
- [ ] `Dockerfile` is in the root directory
- [ ] No sensitive data in the code (API keys should be env variables)
- [ ] Test locally first: `python api.py`

---

## 📁 Required Files for Deployment

Your project should have these files:
```
Voice_Detection/
├── api.py                 # Main API file
├── ai_voice_detector.py   # Detection logic
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration
├── .dockerignore          # Files to ignore in Docker
├── Procfile               # For Heroku/Railway
├── railway.json           # Railway config
└── render.yaml            # Render config
```

---

## 🧪 Testing Your Deployed API

### Health Check
```bash
curl https://your-deployed-url.com/health
```

### API Documentation
```
https://your-deployed-url.com/docs
```

---

## ⚠️ Important Notes

1. **First Request Delay**: The first request may take 30-60 seconds as the model loads
2. **Memory Requirements**: The API needs at least 2GB RAM for the ML model
3. **API Key Security**: Never share your production API key publicly
4. **Rate Limiting**: Consider adding rate limiting for production use

---

## 🎯 Submission Format for Judges

```
AI Voice Detection API
=======================

Endpoint: POST https://your-app.railway.app/api/voice-detection

API Key: sk_prod_your_secret_key

Supported Languages: Tamil, English, Hindi, Malayalam, Telugu

Headers Required:
  - Content-Type: application/json
  - x-api-key: sk_prod_your_secret_key

Request Format:
{
  "language": "Tamil|English|Hindi|Malayalam|Telugu",
  "audioFormat": "mp3",
  "audioBase64": "<base64_encoded_mp3>"
}

Response Format:
{
  "status": "success",
  "language": "<language>",
  "classification": "AI_GENERATED|HUMAN",
  "confidenceScore": 0.0-1.0,
  "explanation": "<reason>"
}

API Documentation: https://your-app.railway.app/docs
```

---

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| Build fails | Check `requirements.txt` has all dependencies |
| Out of memory | Upgrade to a plan with more RAM (2GB minimum) |
| Timeout errors | Increase timeout to 300 seconds |
| Model not loading | Ensure `transformers` and `torch` are installed |

---

## 📞 Quick Deploy Commands

### Push to GitHub
```bash
git init
git add .
git commit -m "AI Voice Detection API"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ai-voice-detection.git
git push -u origin main
```

### Deploy to Railway (via CLI)
```bash
npm install -g @railway/cli
railway login
railway init
railway up
```
