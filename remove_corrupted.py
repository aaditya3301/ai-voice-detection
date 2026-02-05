import os
from pathlib import Path
import librosa
from tqdm import tqdm

def test_audio_file(file_path):
    """Test if an audio file can be loaded"""
    try:
        audio, sr = librosa.load(str(file_path), sr=None, duration=0.1)
        return True
    except:
        return False

def remove_corrupted_files(base_dir):
    """Remove corrupted audio files"""
    base_path = Path(base_dir)
    
    corrupted_files = []
    total_files = 0
    
    print("🔍 Scanning for corrupted audio files...\n")
    
    # Scan all subdirectories
    for audio_file in tqdm(list(base_path.rglob("*.mp3")), desc="Testing files"):
        total_files += 1
        if not test_audio_file(audio_file):
            corrupted_files.append(audio_file)
    
    print(f"\n📊 Found {len(corrupted_files)} corrupted files out of {total_files} total files")
    
    if corrupted_files:
        print(f"\n🗑️  Removing corrupted files...")
        for file_path in tqdm(corrupted_files, desc="Deleting"):
            try:
                file_path.unlink()
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        
        print(f"\n✅ Removed {len(corrupted_files)} corrupted files")
        print(f"✅ {total_files - len(corrupted_files)} valid files remaining")
    else:
        print("\n✅ No corrupted files found!")

if __name__ == "__main__":
    print("🧹 Corrupted Audio File Cleanup")
    print("=" * 60)
    
    # Clean human voices
    if os.path.exists("data/human"):
        print("\n📁 Cleaning data/human...")
        remove_corrupted_files("data/human")
    
    # Clean AI generated voices
    if os.path.exists("data/ai_generated"):
        print("\n📁 Cleaning data/ai_generated...")
        remove_corrupted_files("data/ai_generated")
    
    print("\n✨ Cleanup complete!")
