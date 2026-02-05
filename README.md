---
title: AI Voice Detection
emoji: 🎤
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# 🎙️ AI Voice Detection API

A production-ready REST API for detecting AI-generated voices using deep learning. Built with FastAPI and Wav2Vec2, supporting multiple languages and audio formats.

## ✨ Features

- **Multi-language Support**: English, Hindi, Tamil, Malayalam, Telugu
- **Multiple Audio Formats**: WAV, MP3, FLAC, OGG
- **Deep Learning Model**: Wav2Vec2-based transfer learning
- **REST API**: Fast, scalable FastAPI backend
- **Docker Ready**: Containerized for easy deployment
- **Authentication**: API key-based security

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
docker build -t ai-voice-detector .
docker run -p 8000:8000 ai-voice-detector
```

### Local Setup

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Download Model**
Place `voice_detector_best.pth` in `models/` directory

3. **Run Server**
```bash
python app.py
```

API available at: `http://localhost:8000`

## 📡 API Reference

### Endpoint: `/detect`

**Method:** `POST`

**Headers:**
```
x-api-key: voice_api_5f4090eeb14e57977536228b1130da36b158fe8e
Content-Type: application/json
```

**Request Body:**
```json
{
  "language": "en",
  "audioFormat": "wav",
  "audioBase64": "base64_encoded_audio_data"
}
```

**Response:**
```json
{
  "classification": "human",
  "confidence_score": 0.92,
  "explanation": "Voice exhibits natural human characteristics"
}
```

### Supported Languages
- `en` - English
- `hi` - Hindi  
- `ta` - Tamil
- `ml` - Malayalam
- `te` - Telugu

### Supported Audio Formats
- WAV
- MP3
- FLAC
- OGG
## 🧪 Testing

### Interactive API Documentation
Visit: `http://localhost:8000/docs`

### Using Python
```python
import requests
import base64

# Read audio file
with open('sample.wav', 'rb') as f:
    audio_base64 = base64.b64encode(f.read()).decode()

# Make request
response = requests.post(
    'http://localhost:8000/detect',
    headers={'x-api-key': 'voice_api_5f4090eeb14e57977536228b1130da36b158fe8e'},
    json={
        'language': 'en',
        'audioFormat': 'wav',
        'audioBase64': audio_base64
    }
)

print(response.json())
```

### Using cURL
```bash
curl -X POST http://localhost:8000/detect \
  -H "x-api-key: voice_api_5f4090eeb14e57977536228b1130da36b158fe8e" \
  -H "Content-Type: application/json" \
  -d '{
    "language": "en",
    "audioFormat": "wav",
    "audioBase64": "your_base64_audio_here"
  }'
```

## 📊 Model Information

- **Architecture**: Wav2Vec2-base + Custom Classification Head
- **Training Data**: 1,798 audio samples (human + AI-generated)
- **Accuracy**: 70-85% validation accuracy
- **Languages**: English, Hindi, Tamil, Malayalam, Telugu
- **Inference Time**: ~1-2 seconds per request

## 🚀 Deployment

### Hugging Face Spaces
```bash
git push hf main
```

### Render / Railway
See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions

## 📁 Project Structure

```
.
├── app.py              # Main FastAPI application
├── train_model.py      # Model training script
├── test_api.py         # API testing utilities
├── models/             # Trained model files
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container configuration
├── .env.example        # Environment variables template
└── DEPLOYMENT.md       # Deployment guide
```

## 🔒 Security

- API key authentication required
- Environment variables for sensitive data
- Rate limiting recommended for production

## 📝 License

This project is for educational purposes.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ using FastAPI, PyTorch, and Transformers**with open('sample.mp3', 'rb') as f:
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
