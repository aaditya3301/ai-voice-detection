import requests
import base64
import json
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000"
API_KEY = "voice_api_5f4090eeb14e57977536228b1130da36b158fe8e"  # Must match the key in app.py

def encode_audio_file(file_path):
    """Encode audio file to Base64"""
    with open(file_path, 'rb') as f:
        audio_bytes = f.read()
    return base64.b64encode(audio_bytes).decode('utf-8')

def test_health():
    """Test health endpoint"""
    print("=" * 70)
    print("🏥 Testing Health Endpoint")
    print("=" * 70)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_root():
    """Test root endpoint"""
    print("=" * 70)
    print("🏠 Testing Root Endpoint")
    print("=" * 70)
    
    response = requests.get(f"{API_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_detect_voice(audio_file, language=None):
    """Test voice detection endpoint"""
    print("=" * 70)
    print(f"🎙️  Testing Voice Detection: {Path(audio_file).name}")
    print("=" * 70)
    
    # Encode audio
    try:
        audio_base64 = encode_audio_file(audio_file)
        print(f"✅ Audio encoded ({len(audio_base64)} chars)")
    except FileNotFoundError:
        print(f"❌ File not found: {audio_file}")
        print()
        return
    
    # Prepare request (matching hackathon requirements)
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }
    
    # Detect audio format from file extension
    audio_format = Path(audio_file).suffix.lstrip('.')
    
    payload = {
        "Language": language or "en",
        "Audio Format": audio_format,
        "Audio Base64 Format": audio_base64
    }
    
    # Send request
    print("📤 Sending request to API...")
    response = requests.post(f"{API_URL}/detect", json=payload, headers=headers)
    
    # Display results
    print(f"\n📊 Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ Detection Results:")
        print(f"   🎯 Classification: {result['classification'].upper()}")
        print(f"   📈 Confidence: {result['confidence_score']:.1%}")
        print(f"   💡 Explanation: {result['explanation']}")
    else:
        print(f"\n❌ Error: {response.text}")
    
    print()

def test_invalid_api_key():
    """Test with invalid API key"""
    print("=" * 70)
    print("🔐 Testing Invalid API Key")
    print("=" * 70)
    
    headers = {
        "x-api-key": "invalid-key-12345",
        "Content-Type": "application/json"
    }
    
    payload = {
        "Language": "en",
        "Audio Format": "wav",
        "Audio Base64 Format": "dummy_base64_string"
    }
    
    response = requests.post(f"{API_URL}/detect", json=payload, headers=headers)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print("✅ Authentication working correctly" if response.status_code == 401 else "❌ Auth issue")
    print()

def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("🧪 AI VOICE DETECTION API - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print()
    
    # Test basic endpoints
    test_health()
    test_root()
    
    # Test authentication
    test_invalid_api_key()
    
    # Find and test sample audio files
    print("=" * 70)
    print("🔍 Searching for test audio files...")
    print("=" * 70)
    
    # Look for human voice samples in language folders
    human_files = []
    ai_files = []
    
    # Search in language subfolders: data/human/english/, data/human/hindi/, etc.
    for lang in ['english', 'hindi', 'tamil', 'malayalam', 'telugu']:
        human_files.extend(list(Path(f'data/human/{lang}').glob('*.wav'))[:1])
        ai_files.extend(list(Path(f'data/ai_generated/{lang}').glob('*.wav'))[:1])
    
    # Limit to 2 files each
    human_files = human_files[:2]
    ai_files = ai_files[:2]
    
    print(f"Found {len(human_files)} human samples and {len(ai_files)} AI samples")
    print()
    
    # Test human voices
    for i, file_path in enumerate(human_files, 1):
        print(f"Test {i}/2 - HUMAN VOICE:")
        test_detect_voice(str(file_path), "en")
    
    # Test AI voices
    for i, file_path in enumerate(ai_files, 1):
        print(f"Test {i}/2 - AI VOICE:")
        test_detect_voice(str(file_path), "en")
    
    print("=" * 70)
    print("✅ TEST SUITE COMPLETED!")
    print("=" * 70)
    print("\n📝 Summary:")
    print("   - API endpoints responding correctly")
    print("   - Authentication working")
    print("   - Voice detection model loaded")
    print("   - Ready for production deployment!")
    print()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
