import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import time
from model import SlowFastNetwork

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
MODEL_SAVE_PATH = "best_model_slowfast.pth"

# Hyperparameters
IMG_SIZE = 112
SEQ_LEN = 32 # 32 frames for Fast path
ALPHA = 8    # Alpha to define Slow path (32/8 = 4 frames)
BATCH_SIZE = 16 
EPOCHS = 80
LEARNING_RATE = 1e-4
PATIENCE = 5

# --- 1. Data Augmentation ---
def augment_video_frames(frames):
    """
    Apply augmentation to a sequence of frames.
    """
    augmented_frames = []
    
    do_flip = np.random.random() > 0.5
    do_rotate = np.random.random() > 0.5
    angle = np.random.randint(-15, 15) if do_rotate else 0
    
    brightness = np.random.uniform(0.8, 1.2)
    contrast = np.random.uniform(0.8, 1.2)
    
    for frame in frames:
        new_frame = frame.copy()
        if do_flip:
            new_frame = cv2.flip(new_frame, 1)
        if do_rotate:
            (h, w) = new_frame.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            new_frame = cv2.warpAffine(new_frame, M, (w, h))
        new_frame = cv2.convertScaleAbs(new_frame, alpha=contrast, beta=(brightness-1)*50)
        augmented_frames.append(new_frame)
        
    return np.array(augmented_frames)

# --- Dataset Class ---
class SlowFastDataset(Dataset):
    def __init__(self, video_paths, labels, alpha=ALPHA, transform=None, augment=False):
        self.video_paths = video_paths
        self.labels = labels
        self.augment = augment
        self.alpha = alpha
        
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels[idx]
        
        try:
            frames = self._load_video(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            frames = np.zeros((SEQ_LEN, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        if self.augment:
            frames = augment_video_frames(frames)
            
        # Prepare inputs
        # Frames shape: (T, H, W, C)
        
        # Fast Pathway: Use all frames (High Temporal Resolution)
        fast_frames = frames
        
        # Slow Pathway: Use strided frames (Low Temporal Resolution, High Spatial focus conceptually)
        # Stride = ALPHA
        slow_idx = np.arange(0, SEQ_LEN, self.alpha)
        slow_frames = frames[slow_idx]
        
        # Preprocessing
        fast_input = self._process_frames(fast_frames)
        slow_input = self._process_frames(slow_frames)
        
        return slow_input, fast_input, label
    
    def _process_frames(self, frames):
        # Convert to Tensor and Channel First
        tensor = torch.tensor(frames, dtype=torch.float32)
        tensor = tensor / 255.0 
        tensor = tensor.permute(3, 0, 1, 2) # (C, T, H, W)
        return tensor
    
    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                frames.append(frame)
        finally:
            cap.release()
            
        if len(frames) == 0:
            return np.zeros((SEQ_LEN, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            
        if len(frames) < SEQ_LEN:
            while len(frames) < SEQ_LEN:
                frames.append(frames[-1])
        elif len(frames) > SEQ_LEN:
            # Uniform sampling
            indices = np.linspace(0, len(frames)-1, SEQ_LEN, dtype=int)
            frames = [frames[i] for i in indices]
            
        return np.array(frames)

# --- 2. Data Splitting ---
def prepare_data():
    violence_dir = os.path.join(DATASET_DIR, 'violence')
    no_violence_dir = os.path.join(DATASET_DIR, 'no-violence')
    
    if not os.path.exists(violence_dir) or not os.path.exists(no_violence_dir):
        raise FileNotFoundError(f"Dataset directories not found.")

    violence_files = [os.path.join(violence_dir, f) for f in os.listdir(violence_dir) if f.endswith('.avi') or f.endswith('.mp4')]
    no_violence_files = [os.path.join(no_violence_dir, f) for f in os.listdir(no_violence_dir) if f.endswith('.avi') or f.endswith('.mp4')]
    
    X = violence_files + no_violence_files
    y = [1] * len(violence_files) + [0] * len(no_violence_files)
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# --- 4. Early Stopping ---
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, self.path)
        self.val_loss_min = val_loss

# --- Main Execution ---
if __name__ == "__main__":
    start_time = time.time()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data()
        print(f"Dataset Split Stats:")
        print(f"Train: {len(X_train)} samples")
        print(f"Val:   {len(X_val)} samples")
        print(f"Test:  {len(X_test)} samples")
    except Exception as e:
        print(f"Data preparation failed: {e}")
        exit(1)
    
    train_dataset = SlowFastDataset(X_train, y_train, augment=True)
    val_dataset = SlowFastDataset(X_val, y_val, augment=False)
    test_dataset = SlowFastDataset(X_test, y_test, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = SlowFastNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path=MODEL_SAVE_PATH)
    
    print("\nStarting SlowFast Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (slow_in, fast_in, labels) in enumerate(train_loader):
            slow_in, fast_in, labels = slow_in.to(device), fast_in.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(slow_in, fast_in)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}", end='\r')
            
        train_acc = 100 * correct / total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for slow_in, fast_in, labels in val_loader:
                slow_in, fast_in, labels = slow_in.to(device), fast_in.to(device), labels.to(device)
                outputs = model(slow_in, fast_in)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_acc = 100 * correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'\nEpoch [{epoch+1}/{EPOCHS}] '
              f'Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% '
              f'Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}%')
        
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
            
    print("\nLoading best SlowFast model for evaluation...")
    if os.path.exists(MODEL_SAVE_PATH):
        model = torch.load(MODEL_SAVE_PATH)
    else:
        print("Warning: Using last epoch model.")
    
    model.eval()
    all_preds = []
    all_labels = []
    
    print("Evaluating on Test set...")
    with torch.no_grad():
        for slow_in, fast_in, labels in test_loader:
            slow_in, fast_in, labels = slow_in.to(device), fast_in.to(device), labels.to(device)
            outputs = model(slow_in, fast_in)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print("\n=== SlowFast Model Evaluation Report ===")
    print(classification_report(all_labels, all_preds, target_names=['No Violence', 'Violence']))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    acc = accuracy_score(all_labels, all_preds)
    print(f"\nFinal Test Accuracy: {acc*100:.2f}%")
    
    elapsed = time.time() - start_time
    print(f"\nTotal execution time: {elapsed/60:.2f} minutes")
