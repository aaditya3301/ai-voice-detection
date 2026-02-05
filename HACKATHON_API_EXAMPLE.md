# 🎯 Hackathon API Testing Guide

## ✅ API Requirements Met

Your API now matches the **exact hackathon testing requirements**:

### Headers
- **x-api-key** ✅ (Required)

### Request Body
- **Language** ✅ (Required) - Language code (en, hi, ta, ml, te)
- **Audio Format** ✅ (Required) - File format (mp3, wav, flac, etc.)
- **Audio Base64 Format** ✅ (Required) - Base64-encoded audio

### Response
```json
{
  "classification": "AI-generated" or "human",
  "confidence_score": 0.892,
  "explanation": "Detailed explanation..."
}
```

---

## 📝 Example Request

### cURL
```bash
curl -X POST https://your-api.onrender.com/detect \
  -H "x-api-key: your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "Language": "en",
    "Audio Format": "wav",
    "Audio Base64 Format": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA..."
  }'
```

### Python
```python
import requests
import base64

# Read and encode audio file
with open('voice_sample.wav', 'rb') as f:
    audio_base64 = base64.b64encode(f.read()).decode('utf-8')

# API request
response = requests.post(
    'https://your-api.onrender.com/detect',
    headers={
        'x-api-key': 'your-api-key-here',
        'Content-Type': 'application/json'
    },
    json={
        'Language': 'en',
        'Audio Format': 'wav',
        'Audio Base64 Format': audio_base64
    }
)

result = response.json()
print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidence_score']:.1%}")
print(f"Explanation: {result['explanation']}")
```

### JavaScript
```javascript
const fs = require('fs');

// Read and encode audio file
const audioBuffer = fs.readFileSync('voice_sample.wav');
const audioBase64 = audioBuffer.toString('base64');

// API request
fetch('https://your-api.onrender.com/detect', {
  method: 'POST',
  headers: {
    'x-api-key': 'your-api-key-here',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    'Language': 'en',
    'Audio Format': 'wav',
    'Audio Base64 Format': audioBase64
  })
})
.then(res => res.json())
.then(data => {
  console.log('Classification:', data.classification);
  console.log('Confidence:', (data.confidence_score * 100) + '%');
  console.log('Explanation:', data.explanation);
});
```

---

## 🧪 Testing Locally

1. **Start the API**:
```bash
python app.py
```

2. **Run test suite**:
```bash
python test_api.py
```

3. **Quick test with sample file**:
```python
import requests
import base64

# Encode audio
with open('data/human/common_voice_en_12345.wav', 'rb') as f:
    audio_b64 = base64.b64encode(f.read()).decode('utf-8')

# Test
response = requests.post(
    'http://localhost:8000/detect',
    headers={'x-api-key': 'your-api-key-here'},
    json={
        'Language': 'en',
        'Audio Format': 'wav',
        'Audio Base64 Format': audio_b64
    }
)

print(response.json())
```

---

## 📊 Supported Languages

- **en** - English
- **hi** - Hindi
- **ta** - Tamil
- **ml** - Malayalam
- **te** - Telugu

---

## 🔑 API Key Setup

**For local testing:**
- API Key is set in `app.py`: `VALID_API_KEY = "your-api-key-here"`

**For deployment:**
- Set environment variable: `API_KEY=your-secret-key`
- Or update in deployment config (render.yaml, railway.json)

---

## ✅ Pre-Deployment Checklist

- [ ] Model file (`models/voice_detector_best.pth`) downloaded from Kaggle
- [ ] API tested locally with `python test_api.py`
- [ ] API key configured
- [ ] Deployment platform chosen (Render/Railway/HF Spaces)
- [ ] Sample requests working correctly

---

## 🚀 Quick Deploy to Render

1. **Push to GitHub**:
```bash
git init
git add .
git commit -m "AI Voice Detection API"
git push origin main
```

2. **Deploy on Render**:
- Go to https://render.com
- New → Web Service
- Connect your GitHub repo
- Set environment variable: `API_KEY=your-secret-key`
- Deploy!

3. **Test deployed endpoint**:
```bash
curl https://your-app.onrender.com/health
```

---

## 📞 Endpoint Summary

| Endpoint | Method | Auth | Purpose |
|----------|--------|------|---------|
| `/` | GET | No | API info |
| `/health` | GET | No | Health check |
| `/detect` | POST | Yes | Voice detection |

---

**🎓 Ready for hackathon submission!**

Your API now perfectly matches the testing requirements. Just deploy and submit your endpoint URL!
