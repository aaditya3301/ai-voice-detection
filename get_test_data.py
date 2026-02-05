"""
Quick script to generate test data for API testing
"""
import base64
import json
from pathlib import Path

print("\n" + "="*70)
print("🔑 AI VOICE DETECTION API - TEST DATA GENERATOR")
print("="*70)

# 1. Show the API key
print("\n📌 X-API-KEY:")
print("   x-api-key: your-api-key-here")
print("   (This is set in app.py - use this exact value for testing)")

# 2. Find a sample audio file
print("\n🎵 Finding sample audio file...")

# Try to find a WAV file in language subfolders
audio_file = None
for lang in ['english', 'hindi', 'tamil', 'malayalam', 'telugu']:
    for folder in [f'data/human/{lang}', f'data/ai_generated/{lang}']:
        wav_files = list(Path(folder).glob('*.wav'))
        if wav_files:
            audio_file = wav_files[0]
            break
    if audio_file:
        break

if not audio_file:
    print("   ❌ No WAV files found in data/ folder")
    print("   Please make sure you have audio files in data/human or data/ai_generated")
    exit(1)

print(f"   ✅ Found: {audio_file}")

# 3. Encode to Base64
print(f"\n🔄 Encoding to Base64...")
with open(audio_file, 'rb') as f:
    audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

print(f"   ✅ Encoded successfully ({len(audio_base64)} characters)")

# 4. Create sample request
sample_request = {
    "Language": "en",
    "Audio Format": "wav",
    "Audio Base64 Format": audio_base64
}

# Save to file
output_file = "test_request.json"
with open(output_file, 'w') as f:
    json.dump(sample_request, f, indent=2)

print(f"\n💾 Saved sample request to: {output_file}")

# 5. Show cURL command
print("\n" + "="*70)
print("📋 READY TO TEST!")
print("="*70)

print("\n✅ Option 1 - Using Python:")
print("""
import requests
import json

with open('test_request.json') as f:
    payload = json.load(f)

response = requests.post(
    'http://localhost:8000/detect',
    headers={
        'x-api-key': 'your-api-key-here',
        'Content-Type': 'application/json'
    },
    json=payload
)

print(response.json())
""")

print("\n✅ Option 2 - Using cURL:")
print(f"""
curl -X POST http://localhost:8000/detect \\
  -H "x-api-key: your-api-key-here" \\
  -H "Content-Type: application/json" \\
  -d @test_request.json
""")

print("\n✅ Option 3 - Run the test suite:")
print("""
python test_api.py
""")

# 6. Show first 100 chars of base64 for reference
print("\n📝 Base64 Preview (first 100 chars):")
print(f"   {audio_base64[:100]}...")

print("\n" + "="*70)
print("🎯 Quick Summary:")
print("="*70)
print(f"   x-api-key: your-api-key-here")
print(f"   Language: en")
print(f"   Audio Format: wav")
print(f"   Base64 length: {len(audio_base64)} chars")
print(f"   Full request saved in: {output_file}")
print("\n✅ Ready to test! Start your API with: python app.py")
print("="*70 + "\n")
