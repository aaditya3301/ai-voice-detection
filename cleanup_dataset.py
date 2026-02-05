import os
from pathlib import Path
import shutil

# Configuration: language -> number of clips to keep
TARGET_COUNTS = {
    'telugu': 200
}

def cleanup_language_folder(lang_name, target_count):
    """Keep only target_count clips in a language folder, delete the rest"""
    
    folder_path = Path(f'data/human/{lang_name}')
    
    if not folder_path.exists():
        print(f"❌ Folder not found: {folder_path}")
        return
    
    # Get all audio files (mp3, wav, etc.)
    audio_extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
    all_files = []
    
    for ext in audio_extensions:
        all_files.extend(list(folder_path.glob(f'*{ext}')))
    
    # Sort files by name for consistency
    all_files.sort()
    
    current_count = len(all_files)
    
    print(f"\n📂 {lang_name.upper()}")
    print(f"   Current: {current_count} clips")
    print(f"   Target:  {target_count} clips")
    
    if current_count <= target_count:
        print(f"   ✅ No cleanup needed (already at or below target)")
        return
    
    # Keep first target_count files, delete the rest
    files_to_keep = all_files[:target_count]
    files_to_delete = all_files[target_count:]
    
    print(f"   🗑️  Deleting {len(files_to_delete)} extra clips...")
    
    deleted_count = 0
    for file_path in files_to_delete:
        try:
            file_path.unlink()  # Delete the file
            deleted_count += 1
        except Exception as e:
            print(f"      ⚠️  Error deleting {file_path.name}: {e}")
    
    print(f"   ✅ Cleanup complete! Deleted {deleted_count} files")
    print(f"   📊 Final count: {len(files_to_keep)} clips")

def main():
    print("🧹 Dataset Cleanup Tool")
    print("=" * 50)
    
    # Check if data folder exists
    if not Path('data/human').exists():
        print("❌ Error: data/human folder not found!")
        print("   Make sure you're running this from the Voice directory")
        return
    
    # Process each language
    for lang_name, target_count in TARGET_COUNTS.items():
        cleanup_language_folder(lang_name, target_count)
    
    print("\n" + "=" * 50)
    print("🎉 Dataset cleanup complete!")
    print("\n📊 Final Dataset Summary:")
    
    total_clips = 0
    for lang_name in TARGET_COUNTS.keys():
        folder_path = Path(f'data/human/{lang_name}')
        if folder_path.exists():
            audio_files = list(folder_path.glob('*.mp3')) + list(folder_path.glob('*.wav'))
            count = len(audio_files)
            total_clips += count
            print(f"   {lang_name.capitalize()}: {count} clips")
    
    print(f"\n   TOTAL: {total_clips} human voice clips")

if __name__ == "__main__":
    main()
