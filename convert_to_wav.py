import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

def convert_mp3_to_wav(mp3_path, wav_path):
    """Convert MP3 to WAV using librosa"""
    try:
        # Load MP3
        audio, sr = librosa.load(str(mp3_path), sr=None)
        # Save as WAV
        sf.write(str(wav_path), audio, sr)
        return True
    except Exception as e:
        print(f"Error converting {mp3_path}: {e}")
        return False

def convert_all_mp3s():
    """Convert all MP3 files to WAV in the dataset"""
    
    data_folders = [
        'data/human',
        'data/ai_generated'
    ]
    
    total_converted = 0
    total_failed = 0
    
    for folder in data_folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            continue
        
        # Find all MP3 files
        mp3_files = list(folder_path.rglob('*.mp3'))
        
        print(f"\n📁 {folder}: Found {len(mp3_files)} MP3 files")
        
        for mp3_file in tqdm(mp3_files, desc=f"Converting {folder}"):
            wav_file = mp3_file.with_suffix('.wav')
            
            if wav_file.exists():
                mp3_file.unlink()  # Delete MP3 if WAV exists
                continue
            
            if convert_mp3_to_wav(mp3_file, wav_file):
                mp3_file.unlink()  # Delete MP3 after successful conversion
                total_converted += 1
            else:
                total_failed += 1
    
    print(f"\n✅ Conversion complete!")
    print(f"   Converted: {total_converted} files")
    print(f"   Failed: {total_failed} files")

if __name__ == "__main__":
    print("🎵 Converting MP3 files to WAV format...")
    print("This fixes the 'aifc' module error in Python 3.13")
    print("=" * 60)
    
    convert_all_mp3s()
