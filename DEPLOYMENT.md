# 🚀 AI Voice Detection API - Deployment Guide

## Overview
FastAPI-based REST API for detecting AI-generated vs human voices using wav2vec2 transfer learning.

---

## 📋 Prerequisites

### Required Files
- ✅ `app.py` - FastAPI application
- ✅ `models/voice_detector_best.pth` - Trained model (download from Kaggle after training)
- ✅ `requirements.txt` - Python dependencies

### System Requirements
- **Python**: 3.11+
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB for model and dependencies
- **GPU**: Optional (CPU works fine for inference)

---

## 🔧 Local Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Add Trained Model
After training on Kaggle:
1. Download `voice_detector_best.pth` from Kaggle output
2. Place it in `models/` folder
```bash
mkdir -p models
# Copy voice_detector_best.pth to models/
```

### 3. Configure API Key
Edit `app.py` line 22:
```python
VALID_API_KEY = "your-secure-api-key-here"  # Change this!
```

### 4. Run Server
```bash
python app.py
```
or
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 5. Test API
```bash
python test_api.py
```

Access docs: http://localhost:8000/docs

---

## ☁️ Cloud Deployment

### Option 1: Render.com (Recommended - FREE)

#### Step 1: Create `render.yaml`
```yaml
services:
  - type: web
    name: voice-detection-api
    runtime: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
      - key: API_KEY
        value: your-api-key-here
        sync: false
```

#### Step 2: Upload Model
Since Render has file size limits, use environment variable or cloud storage:
```python
# In app.py, add model download from cloud storage
import os
if not os.path.exists(MODEL_PATH):
    # Download from your cloud storage (Google Drive, S3, etc.)
    pass
```

#### Step 3: Deploy
1. Push to GitHub
2. Connect Render to your repo
3. Deploy automatically

**URL**: `https://your-app-name.onrender.com`

---

### Option 2: Railway.app

#### Step 1: Create `railway.json`
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS",
    "buildCommand": "pip install -r requirements.txt"
  },
  "deploy": {
    "startCommand": "uvicorn app:app --host 0.0.0.0 --port $PORT",
    "restartPolicyType": "ON_FAILURE"
  }
}
```

#### Step 2: Deploy
```bash
# Install Railway CLI
npm i -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

---

### Option 3: Hugging Face Spaces

#### Create Spaces with Docker

**Dockerfile**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

Push to HF Spaces and it auto-deploys!

---

## 🔐 API Usage

### Authentication
All requests require API key in header:
```bash
Authorization: Bearer your-api-key-here
```

### Endpoints

#### 1. Health Check
```bash
GET /health
```

#### 2. Root Info
```bash
GET /
```

#### 3. Detect Voice
```bash
POST /detect
Content-Type: application/json
Authorization: Bearer your-api-key-here

{
  "audio": "base64_encoded_audio_file",
  "language": "en"  // optional: en, hi, ta, ml, te
}
```

**Response**:
```json
{
  "classification": "AI-generated",
  "confidence_score": 0.892,
  "explanation": "High confidence: Audio exhibits regularized pitch variations..."
}
```

---

## 📝 Example Usage

### Python
```python
import requests
import base64

# Encode audio
with open('voice.mp3', 'rb') as f:
    audio_b64 = base64.b64encode(f.read()).decode()

# Send request
response = requests.post(
    'https://your-api.onrender.com/detect',
    headers={'Authorization': 'Bearer your-key'},
    json={'audio': audio_b64, 'language': 'en'}
)

print(response.json())
```

### JavaScript
```javascript
const fs = require('fs');

const audioBase64 = fs.readFileSync('voice.mp3', 'base64');

fetch('https://your-api.onrender.com/detect', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer your-key',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    audio: audioBase64,
    language: 'en'
  })
})
.then(res => res.json())
.then(data => console.log(data));
```

### cURL
```bash
curl -X POST "https://your-api.onrender.com/detect" \
  -H "Authorization: Bearer your-key" \
  -H "Content-Type: application/json" \
  -d "{\"audio\":\"$(base64 -w 0 voice.mp3)\",\"language\":\"en\"}"
```

---

## 🎯 Model Performance

- **Training Accuracy**: ~75-85%
- **Validation Accuracy**: ~70-80%
- **Inference Speed**: ~1-2 seconds per audio clip
- **Supported Formats**: MP3, WAV, FLAC
- **Max Audio Length**: 10 seconds (auto-truncated)
- **Languages**: English, Tamil, Hindi, Malayalam, Telugu

---

## 🔧 Troubleshooting

### Issue: Model not found
**Solution**: Ensure `models/voice_detector_best.pth` exists
```bash
ls -lh models/voice_detector_best.pth
```

### Issue: Out of memory
**Solution**: Use CPU instead of GPU in production (smaller memory footprint)

### Issue: Slow inference
**Solution**: 
- Enable model quantization
- Use smaller batch sizes
- Deploy on GPU instance

---

## 📊 Monitoring

Add logging and monitoring:
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In detect endpoint
logger.info(f"Request received: classification={result['classification']}")
```

---

## 🛡️ Security Best Practices

1. **API Key Management**: Use environment variables
2. **Rate Limiting**: Implement with `slowapi`
3. **Input Validation**: Limit audio file size (< 10MB)
4. **HTTPS Only**: Enable SSL in production
5. **CORS**: Configure allowed origins

---

## 📚 Next Steps

1. ✅ Train model on Kaggle
2. ✅ Download trained model
3. ✅ Test locally with `python test_api.py`
4. ✅ Deploy to cloud (Render/Railway)
5. ✅ Test deployed API
6. ✅ Submit API endpoint for hackathon evaluation

---

## 📞 Support

For issues:
- Check logs: `tail -f app.log`
- Test health: `curl http://localhost:8000/health`
- Verify model: `python -c "import torch; print(torch.load('models/voice_detector_best.pth'))"`

---

**Built with**: FastAPI, PyTorch, Transformers (wav2vec2), Librosa
**License**: MIT
**Author**: Your Name
