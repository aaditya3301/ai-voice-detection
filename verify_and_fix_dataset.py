import os
from pathlib import Path
import asyncio
import edge_tts
import random

# Configuration
LANGUAGES = ['english', 'hindi', 'tamil', 'malayalam', 'telugu']
TARGET_COUNT = 200

# Voice configurations
VOICE_CONFIG = {
    'english': ['en-US-AriaNeural', 'en-US-GuyNeural'],
    'hindi': ['hi-IN-SwaraNeural', 'hi-IN-MadhurNeural'],
    'tamil': ['ta-IN-PallaviNeural', 'ta-IN-ValluvarNeural'],
    'malayalam': ['ml-IN-SobhanaNeural', 'ml-IN-MidhunNeural'],
    'telugu': ['te-IN-ShrutiNeural', 'te-IN-MohanNeural']
}

# Sample texts
SAMPLE_TEXTS = {
    'english': ["Hello, how are you?", "This is a test.", "Good morning.", "Thank you very much."],
    'hindi': ["नमस्ते", "यह एक परीक्षण है", "शुभ प्रभात", "धन्यवाद"],
    'tamil': ["வணக்கம்", "இது ஒரு சோதனை", "காலை வணக்கம்", "நன்றி"],
    'malayalam': ["ഹലോ", "ഇതൊരു പരീക്ഷണമാണ്", "സുപ്രഭാതം", "നന്ദി"],
    'telugu': ["హలో, మీరు ఎలా ఉన్నారు?",
        "ఈరోజు వాతావరణం చాలా అందంగా ఉంది.",
        "నేను ప్రతిరోజూ కొత్త విషయాలు నేర్చుకోవడం ఇష్టపడతాను.",
        "సాంకేతికత వేగంగా అభివృద్ధి చెందుతోంది.",
        "దయచేసి తిరిగి కాల్ చేయండి.",
        "సమావేశం ఏ సమయంలో షెడ్యూల్ చేయబడింది?",
        "నేను ఐదు నిమిషాల్లో అక్కడికి చేరుకుంటాను.",
        "మీ సహాయానికి ధన్యవాదాలు.",
        "కృత్రిమ మేధస్సు ప్రపంచాన్ని మారుస్తోంది.",
        "మీరు నన్ను స్పష్టంగా వినగలరా?",
        "నేను ఈ ప్రాజెక్ట్‌ను ఈరోజు పూర్తి చేయాలి.",
        "శుభోదయం, మంచి రోజు కలగాలని కోరుకుంటున్నాను.",
        "వచ్చే వారం ఒక సమావేశం షెడ్యూల్ చేద్దాం.",
        "ఇది చాలా ముఖ్యమైనది.",
        "నేను మీకు సహాయం చేయడానికి ఇక్కడ ఉన్నాను."]
}

def check_and_cleanup_folder(folder_path, target_count):
    """Check folder and clean up to exactly target_count files"""
    
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)
        return 0, target_count
    
    # Get all audio files
    audio_files = sorted(list(folder_path.glob('*.mp3')) + list(folder_path.glob('*.wav')))
    current_count = len(audio_files)
    
    if current_count > target_count:
        # Delete extras
        files_to_delete = audio_files[target_count:]
        for file in files_to_delete:
            file.unlink()
        return target_count, 0
    elif current_count < target_count:
        # Need to generate more
        return current_count, target_count - current_count
    else:
        return current_count, 0

async def generate_ai_voice(text, voice, output_path):
    """Generate single AI voice"""
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(output_path))
        return True
    except:
        return False

async def generate_missing_ai_voices(lang_name, needed_count, start_index):
    """Generate missing AI voices for a language"""
    
    voices = VOICE_CONFIG[lang_name]
    texts = SAMPLE_TEXTS[lang_name]
    output_dir = Path(f'data/ai_generated/{lang_name}')
    
    print(f"   🎙️  Generating {needed_count} AI voices...")
    
    generated = 0
    for i in range(needed_count):
        idx = start_index + i
        voice = voices[i % len(voices)]
        text = texts[i % len(texts)]
        
        output_path = output_dir / f"ai_clip_{idx:05d}.mp3"
        
        success = await generate_ai_voice(text, voice, output_path)
        if success:
            generated += 1
            if (generated % 50 == 0):
                print(f"      ✓ Generated {generated}/{needed_count}...")
    
    return generated

async def main():
    print("🔍 Checking all dataset folders...")
    print("=" * 60)
    
    total_human = 0
    total_ai = 0
    
    for lang in LANGUAGES:
        print(f"\n📂 {lang.upper()}")
        
        # Check human voices
        human_path = Path(f'data/human/{lang}')
        human_count, human_needed = check_and_cleanup_folder(human_path, TARGET_COUNT)
        print(f"   Human voices: {human_count}/{TARGET_COUNT}", end="")
        if human_needed > 0:
            print(f" ⚠️  Missing {human_needed} files!")
        else:
            print(" ✅")
        total_human += human_count
        
        # Check AI voices
        ai_path = Path(f'data/ai_generated/{lang}')
        ai_count, ai_needed = check_and_cleanup_folder(ai_path, TARGET_COUNT)
        print(f"   AI voices:    {ai_count}/{TARGET_COUNT}", end="")
        
        if ai_needed > 0:
            print(f" 🎙️  Generating {ai_needed}...")
            generated = await generate_missing_ai_voices(lang, ai_needed, ai_count)
            total_ai += ai_count + generated
            print(f"   ✅ AI voices complete: {ai_count + generated}/{TARGET_COUNT}")
        else:
            print(" ✅")
            total_ai += ai_count
    
    print("\n" + "=" * 60)
    print("📊 Final Dataset Summary:")
    print(f"   Human voices:  {total_human} clips")
    print(f"   AI voices:     {total_ai} clips")
    print(f"   TOTAL:         {total_human + total_ai} clips")
    
    if total_human == 1000 and total_ai == 1000:
        print("\n✅ Dataset is complete and ready for training!")
    else:
        print(f"\n⚠️  Note: You may need to add more human voice files manually")

if __name__ == "__main__":
    asyncio.run(main())
