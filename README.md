# 🎙️ AI Voice Detection API

Real-time detection of AI-generated vs human voices using wav2vec2 transfer learning. Supports 5 languages: English, Tamil, Hindi, Malayalam, Telugu.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Add Trained Model
Download `voice_detector_best.pth` from Kaggle and place in `models/` folder:
```bash
mkdir models
# Copy voice_detector_best.pth here
```

### 3. Run API
```bash
python app.py
```
API runs at: http://localhost:8000

### 4. Test
```bash
python test_api.py
```

## 📡 API Endpoints

### Detect Voice
```bash
POST /detect
Authorization: Bearer your-api-key-here

{
  "audio": "base64_encoded_audio",
  "language": "en"
}
```

**Response:**
```json
{
  "classification": "AI-generated",
  "confidence_score": 0.892,
  "explanation": "High confidence: Audio exhibits regularized pitch..."
}
```

## 🎯 Features

- ✅ Real-time voice classification
- ✅ Multi-language support (5 languages)
- ✅ High accuracy (~75-85%)
- ✅ Fast inference (~1-2 seconds)
- ✅ REST API with authentication
- ✅ Detailed confidence scores
- ✅ Human-readable explanations

## 📊 Model Details

- **Architecture**: wav2vec2-base + custom classifier
- **Training**: Transfer learning on 1,798 audio samples
- **Accuracy**: ~70-85% validation accuracy
- **Languages**: EN, HI, TA, ML, TE

## 🔧 Configuration

Edit `app.py`:
```python
VALID_API_KEY = "your-secure-key"  # Line 22
MODEL_PATH = "models/voice_detector_best.pth"  # Line 23
```

## 📁 Project Structure

```
Voice/
├── app.py                      # FastAPI application
├── train_model.py              # Training script
├── test_api.py                 # API testing
├── voice.ipynb                 # Kaggle training notebook
├── models/
│   └── voice_detector_best.pth # Trained model
├── data/
│   ├── human/                  # Human voice samples
│   └── ai_generated/           # AI voice samples
├── requirements.txt            # Dependencies
├── Dockerfile                  # Docker config
├── render.yaml                 # Render deployment
├── railway.json                # Railway deployment
└── DEPLOYMENT.md               # Full deployment guide
```

## 🌐 Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions for:
- Render.com (Free)
- Railway.app
- Hugging Face Spaces
- Docker

## 🧪 Testing

Interactive API docs: http://localhost:8000/docs

Test with sample audio:
```python
import requests
import base64

with open('sample.mp3', 'rb') as f:
    audio = base64.b64encode(f.read()).decode()

response = requests.post(
    'http://localhost:8000/detect',
    headers={'Authorization': 'Bearer your-api-key-here'},
    json={'audio': audio}
)

print(response.json())
```

## 📈 Performance

- **Inference Speed**: 1-2 seconds/clip
- **Memory Usage**: ~500MB
- **Max Audio Length**: 10 seconds
- **Supported Formats**: MP3, WAV, FLAC

## 🛡️ Security

- API key authentication
- Input validation
- Rate limiting ready
- HTTPS recommended for production

## 📝 License

MIT License - see LICENSE file

## 🙋 Support

- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- Test Suite: `python test_api.py`

---

Built with ❤️ using FastAPI, PyTorch, and Transformers
