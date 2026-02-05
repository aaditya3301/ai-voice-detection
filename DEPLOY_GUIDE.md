# 🚀 Quick Deployment Guide

## ✅ Step 1: Upload Model to Google Drive

1. Go to https://drive.google.com
2. Click "New" → "File upload"
3. Upload `models/voice_detector_best.pth` (1GB file)
4. Right-click the uploaded file → "Share"
5. Set to "Anyone with the link can view"
6. Click "Copy link"
7. Extract FILE_ID from the URL:
   - URL looks like: `https://drive.google.com/file/d/1abc123xyz456def/view`
   - FILE_ID is: `1abc123xyz456def`

## ✅ Step 2: Update app.py with FILE_ID

Open `app.py` and find line ~30:

```python
GOOGLE_DRIVE_FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID_HERE"
```

Replace with your actual FILE_ID:

```python
GOOGLE_DRIVE_FILE_ID = "1abc123xyz456def"
```

## ✅ Step 3: Push to GitHub

```bash
# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "AI Voice Detection API"

# Create repo on GitHub (go to github.com):
# - Click "New repository"
# - Name: voice-detection-api
# - Public/Private: Your choice
# - Don't initialize with README

# Link and push (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/voice-detection-api.git
git branch -M main
git push -u origin main
```

## ✅ Step 4: Deploy on Render

1. Go to https://render.com
2. Sign up/Login with GitHub
3. Click "New +" → "Web Service"
4. Connect your GitHub repository: `voice-detection-api`
5. Configure settings:
   - **Name**: `voice-detection-api`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
   - **Instance Type**: `Free` (or paid for faster performance)

6. Click "Advanced" → "Add Environment Variable":
   - **Key**: `API_KEY`
   - **Value**: `voice_api_5f4090eeb14e57977536228b1130da36b158fe8e`

7. Click "Create Web Service"

## ✅ Step 5: Wait for Deployment

Render will:
1. Install dependencies (~3-5 minutes)
2. Download model from Google Drive (~2-3 minutes for 1GB file)
3. Start your API

Watch the logs for:
```
✅ Model downloaded successfully!
🚀 Loading AI Voice Detection Model...
✅ Model loaded successfully from models/voice_detector_best.pth
   Validation Accuracy: 100.00%
```

## ✅ Step 6: Test Your Deployed API

Your API will be at: `https://voice-detection-api.onrender.com`

Test it:
```bash
# Health check
curl https://voice-detection-api.onrender.com/health

# Root endpoint
curl https://voice-detection-api.onrender.com/
```

## 📝 Submit to Hackathon

Submit this URL to the hackathon tester:
```
https://voice-detection-api.onrender.com/detect
```

API Key:
```
    voice_api_5f4090eeb14e57977536228b1130da36b158fe8e
```

## 🎉 You're Done!

Your AI Voice Detection API is now live and ready for the hackathon! 🏆
