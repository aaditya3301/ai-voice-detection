# Model File Setup for Deployment

Your trained model (`voice_detector_best.pth`) is ~1GB and too large for GitHub.

## Option 1: Google Drive (Recommended)

1. Upload `models/voice_detector_best.pth` to Google Drive
2. Right-click → Share → Get link
3. Make sure "Anyone with the link can view"
4. Copy the FILE_ID from the URL: `https://drive.google.com/file/d/FILE_ID_HERE/view`

## Option 2: Hugging Face Hub

1. Create account at https://huggingface.co
2. Create a new model repository
3. Upload `voice_detector_best.pth`
4. Get the download URL

## For Render Deployment

Add this to your `app.py` startup to download the model automatically:

```python
import os
import requests

# Download model if not present
if not Path(MODEL_PATH).exists():
    print("📥 Downloading model from cloud...")
    os.makedirs("models", exist_ok=True)
    
    # Google Drive direct download
    file_id = "YOUR_FILE_ID_HERE"
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    
    response = requests.get(url)
    with open(MODEL_PATH, 'wb') as f:
        f.write(response.content)
    print("✅ Model downloaded successfully")
```
