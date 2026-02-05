from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field
import base64
import io
import librosa
import numpy as np
from typing import Optional
import uvicorn
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from pathlib import Path
import os
import gdown

# Initialize FastAPI app
app = FastAPI(
    title="AI Voice Detection API",
    description="Detects AI-generated vs human voices in multiple languages",
    version="1.0.0"
)

# Configuration
# NOTE: This is YOUR API key - set it to any secure random string (e.g., "sk_live_abc123xyz789")
# This is what clients will use to authenticate with YOUR API
VALID_API_KEY = "voice_api_5f4090eeb14e57977536228b1130da36b158fe8e"  # Change this to a secure random string
MODEL_PATH = "models/voice_detector_best.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAMPLE_RATE = 16000
MAX_DURATION = 10  # seconds

# Google Drive model download (REPLACE WITH YOUR FILE ID)
# Upload your model to Google Drive, get shareable link, extract the FILE_ID
# Example: https://drive.google.com/file/d/1abc123xyz/view -> FILE_ID = 1abc123xyz
GOOGLE_DRIVE_FILE_ID = "1J733egg7W_UwV3wfi_F0YvsBOAiojgjX"  # Extracted from your Google Drive link

# Model Architecture (same as training)
class VoiceClassifier(nn.Module):
    def __init__(self, pretrained_model_name='facebook/wav2vec2-base'):
        super(VoiceClassifier, self).__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model_name)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
    
    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state
        pooled = torch.mean(hidden_states, dim=1)
        logits = self.classifier(pooled)
        return logits

# Global variables for model and processor (loaded at startup)
processor = None
model = None

@app.on_event("startup")
async def load_model():
    """Load model when the API starts (fixes Windows multiprocessing issue)"""
    global processor, model
    
    # Download model from Google Drive if not present
    if not Path(MODEL_PATH).exists():
        print("📥 Model not found locally. Downloading from Google Drive...")
        os.makedirs("models", exist_ok=True)
        
        try:
            if GOOGLE_DRIVE_FILE_ID != "YOUR_GOOGLE_DRIVE_FILE_ID_HERE":
                url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
                gdown.download(url, MODEL_PATH, quiet=False)
                print("✅ Model downloaded successfully!")
            else:
                print("⚠️  Google Drive File ID not set. Please update GOOGLE_DRIVE_FILE_ID in app.py")
        except Exception as e:
            print(f"❌ Failed to download model: {e}")
            print("   API will run without model predictions")
    
    print("🚀 Loading AI Voice Detection Model...")
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
    model = VoiceClassifier().to(DEVICE)
    
    # Load trained weights
    if Path(MODEL_PATH).exists():
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
        print(f"   Validation Accuracy: {checkpoint['val_acc']:.2f}%")
    else:
        print(f"⚠️  Model file not found at {MODEL_PATH}")
        print("   API will run in demo mode with placeholder predictions")

# Request model (matching hackathon requirements)
class VoiceRequest(BaseModel):
    Language: str = Field(..., description="Language code (en, hi, ta, ml, te)")
    Audio_Format: str = Field(..., alias="Audio Format", description="Audio format (mp3, wav, etc.)")
    Audio_Base64_Format: str = Field(..., alias="Audio Base64 Format", description="Base64-encoded audio")
    
    class Config:
        populate_by_name = True  # Allow both field name and alias
    
# Response model
class VoiceResponse(BaseModel):
    classification: str  # "AI-generated" or "human"
    confidence_score: float  # 0.0 to 1.0
    explanation: str

# API Key authentication
async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify API key from x-api-key header"""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="x-api-key header missing")
    
    if x_api_key != VALID_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return x_api_key

# Audio processing functions
def decode_audio(base64_audio: str):
    """Decode Base64 audio to audio array"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_audio:
            base64_audio = base64_audio.split(',')[1]
        
        # Decode Base64
        audio_bytes = base64.b64decode(base64_audio)
        
        # Load audio using librosa
        audio_io = io.BytesIO(audio_bytes)
        audio, sr = librosa.load(audio_io, sr=SAMPLE_RATE)
        
        return audio, sr
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode audio: {str(e)}")

def preprocess_audio(audio, sr):
    """Preprocess audio for model input"""
    try:
        # Pad or truncate to max duration
        target_length = MAX_DURATION * sr
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        
        # Process with wav2vec2 processor
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
        
        return inputs.input_values
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio preprocessing failed: {str(e)}")

def classify_voice(audio_tensor):
    """
    Classify voice as AI-generated or human using trained model
    """
    try:
        with torch.no_grad():
            # Move to device
            audio_tensor = audio_tensor.to(DEVICE)
            
            # Get prediction
            outputs = model(audio_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get predicted class and confidence
            confidence, predicted = torch.max(probabilities, 1)
            confidence = float(confidence[0])
            predicted_class = int(predicted[0])
            
            # Class 0 = Human, Class 1 = AI
            if predicted_class == 1:
                classification = "AI-generated"
                explanation = generate_explanation("ai", confidence)
            else:
                classification = "human"
                explanation = generate_explanation("human", confidence)
            
            return classification, confidence, explanation
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

def generate_explanation(classification_type, confidence):
    """Generate human-readable explanation based on classification"""
    if classification_type == "ai":
        if confidence > 0.9:
            return "Very high confidence: Detected consistent spectral patterns and unnatural prosody characteristic of synthetic voice generation."
        elif confidence > 0.75:
            return "High confidence: Audio exhibits regularized pitch variations and uniform energy distribution typical of AI-generated speech."
        else:
            return "Moderate confidence: Some synthetic patterns detected, but with natural-like variations present."
    else:  # human
        if confidence > 0.9:
            return "Very high confidence: Natural breathing patterns, irregular micro-pauses, and authentic vocal tract resonance detected."
        elif confidence > 0.75:
            return "High confidence: Organic voice characteristics including natural pitch variations and authentic speech dynamics observed."
        else:
            return "Moderate confidence: Predominantly human voice features with some regularities that may indicate post-processing."

# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "message": "AI Voice Detection API is operational",
        "version": "1.0.0"
    }

@app.post("/detect", response_model=VoiceResponse, dependencies=[Depends(verify_api_key)])
async def detect_voice(request: VoiceRequest):
    """
    Detect if a voice sample is AI-generated or human
    
    - **Language**: Language code (en, hi, ta, ml, te)
    - **Audio_Format**: Audio format (mp3, wav, flac, etc.)
    - **Audio_Base64_Format**: Base64-encoded audio file
    """
    try:
        # Step 1: Decode audio
        audio, sr = decode_audio(request.Audio_Base64_Format)
        
        # Validate audio length (at least 1 second)
        if len(audio) < sr:
            raise HTTPException(status_code=400, detail="Audio too short (minimum 1 second required)")
        
        # Step 2: Preprocess audio
        audio_tensor = preprocess_audio(audio, sr)
        
        # Step 3: Classify
        classification, confidence, explanation = classify_voice(audio_tensor)
        
        # Step 4: Return response
        return VoiceResponse(
            classification=classification,
            confidence_score=round(confidence, 3),
            explanation=explanation
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check for monitoring"""
    return {"status": "healthy"}

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
