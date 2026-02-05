import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import librosa
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import json
from tqdm import tqdm

# Configuration
SAMPLE_RATE = 16000
MAX_DURATION = 10  # seconds
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"🖥️  Using device: {DEVICE}")

# Dataset class
class VoiceDataset(Dataset):
    def __init__(self, file_paths, labels, processor, max_duration=MAX_DURATION):
        self.file_paths = file_paths
        self.labels = labels
        self.processor = processor
        self.max_duration = max_duration
        self.sample_rate = SAMPLE_RATE
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load audio
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            
            # Trim or pad to max_duration
            max_length = self.max_duration * self.sample_rate
            if len(audio) > max_length:
                audio = audio[:max_length]
            else:
                audio = np.pad(audio, (0, max_length - len(audio)), mode='constant')
            
            # Process with wav2vec2 processor
            inputs = self.processor(audio, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
            
            return {
                'input_values': inputs.input_values.squeeze(0),
                'label': torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return zeros if file fails to load
            return {
                'input_values': torch.zeros(self.max_duration * self.sample_rate),
                'label': torch.tensor(label, dtype=torch.long)
            }

# Model class
class VoiceClassifier(nn.Module):
    def __init__(self, pretrained_model_name='facebook/wav2vec2-base'):
        super(VoiceClassifier, self).__init__()
        
        # Load pre-trained wav2vec2
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model_name)
        
        # Freeze wav2vec2 layers initially (we'll fine-tune later)
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),  # wav2vec2-base outputs 768 dimensions
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # Binary classification: AI vs Human
        )
    
    def forward(self, input_values):
        # Extract features from wav2vec2
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state
        
        # Global average pooling
        pooled_output = torch.mean(hidden_states, dim=1)
        
        # Classify
        logits = self.classifier(pooled_output)
        return logits
    
    def unfreeze_wav2vec2(self):
        """Unfreeze wav2vec2 for fine-tuning"""
        for param in self.wav2vec2.parameters():
            param.requires_grad = True

# Data loading function
def load_dataset():
    """Load all audio files and create labels"""
    
    print("📂 Loading dataset...")
    
    file_paths = []
    labels = []
    languages = []
    
    data_root = Path('data')
    
    # Load human voices (label = 0)
    human_path = data_root / 'human'
    for lang_folder in human_path.iterdir():
        if lang_folder.is_dir():
            for audio_file in lang_folder.glob('*.mp3'):
                file_paths.append(str(audio_file))
                labels.append(0)  # Human
                languages.append(lang_folder.name)
            for audio_file in lang_folder.glob('*.wav'):
                file_paths.append(str(audio_file))
                labels.append(0)
                languages.append(lang_folder.name)
    
    # Load AI voices (label = 1)
    ai_path = data_root / 'ai_generated'
    for lang_folder in ai_path.iterdir():
        if lang_folder.is_dir():
            for audio_file in lang_folder.glob('*.mp3'):
                file_paths.append(str(audio_file))
                labels.append(1)  # AI
                languages.append(lang_folder.name)
            for audio_file in lang_folder.glob('*.wav'):
                file_paths.append(str(audio_file))
                labels.append(1)
                languages.append(lang_folder.name)
    
    print(f"✅ Loaded {len(file_paths)} audio files")
    print(f"   Human: {labels.count(0)} | AI: {labels.count(1)}")
    
    return file_paths, labels, languages

# Training function
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        input_values = batch['input_values'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_values)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_values = batch['input_values'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_values)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy, all_preds, all_labels

# Main training script
def main():
    print("🚀 AI Voice Detection - Transfer Learning with Wav2Vec2")
    print("=" * 70)
    
    # Load dataset
    file_paths, labels, languages = load_dataset()
    
    # Train-val-test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        file_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\n📊 Dataset Split:")
    print(f"   Training:   {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")
    print(f"   Test:       {len(X_test)} samples")
    
    # Load processor
    print("\n📥 Loading Wav2Vec2 processor...")
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
    
    # Create datasets
    print("🔧 Creating datasets...")
    train_dataset = VoiceDataset(X_train, y_train, processor)
    val_dataset = VoiceDataset(X_val, y_val, processor)
    test_dataset = VoiceDataset(X_test, y_test, processor)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Initialize model
    print("\n🤖 Initializing model...")
    model = VoiceClassifier().to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("\n🎓 Starting training...")
    print("=" * 70)
    
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/voice_detector_best.pth')
            print(f"✅ Saved best model! (Val Acc: {val_acc:.2f}%)")
        
        # Unfreeze wav2vec2 after 3 epochs for fine-tuning
        if epoch == 2:
            print("\n🔓 Unfreezing Wav2Vec2 for fine-tuning...")
            model.unfreeze_wav2vec2()
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE/10)
    
    # Final evaluation on test set
    print("\n" + "=" * 70)
    print("📊 Final Evaluation on Test Set")
    print("=" * 70)
    
    model.load_state_dict(torch.load('models/voice_detector_best.pth'))
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, DEVICE)
    
    print(f"\n✅ Test Accuracy: {test_acc:.2f}%")
    print(f"   Test Loss: {test_loss:.4f}")
    
    # Classification report
    print("\n📈 Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=['Human', 'AI-Generated']))
    
    # Confusion matrix
    print("\n🔢 Confusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds)
    print(cm)
    
    # Save training history
    with open('models/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n✅ Training complete!")
    print(f"📁 Model saved to: models/voice_detector_best.pth")
    print(f"📊 History saved to: models/training_history.json")

if __name__ == "__main__":
    # Create models directory
    Path('models').mkdir(exist_ok=True)
    
    main()
