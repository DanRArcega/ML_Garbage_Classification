import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler, LabelEncoder

from utils import get_metrics, load_train_test_imgs, convert_to_grayscale, sample_per_label, new_load_train_test_imgs
from preprocessing.dataset import build_dataloaders
from preprocessing.config import DataConfig, DATA_CONFIG, Classes, CLASSES
from CNN import GarbageClassificationCNN

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cnn = GarbageClassificationCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Khởi tạo Optimizer và Scheduler
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
    
    # ReduceLROnPlateau:
    # mode='min': Reduce the learning rate when the quantity monitored has stopped decreasing (i.e., when validation loss stops improving).
    # factor=0.5: Reduce the learning rate by a factor of 0.5 (i.e., new_lr = old_lr * 0.5) when triggered.
    # patience=3: Wait for 3 epochs with no improvement before reducing the learning rate. 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Normalize pixel values to [0, 1]
    def imgs_to_4d_tensor(imgs, device):
        arrays = np.stack([np.array(img, dtype=np.float32) / 255.0 for img in imgs])
        return torch.tensor(arrays[:, np.newaxis, :, :], dtype=torch.float32).to(device)

    print("Building dataloaders...")
    train_loader, test_loader, validation_loader = build_dataloaders(DATA_CONFIG, CLASSES)
    print("Dataloaders ready.")

    num_epochs = 50

    for epoch in range(num_epochs):
        # --- TRAINING PHASE ---
        cnn.train()
        total_train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = cnn(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- VALIDATION PHASE ---
        cnn.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for val_images, val_labels in validation_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = cnn(val_images)
                v_loss = criterion(val_outputs, val_labels)
                total_val_loss += v_loss.item()
                
        avg_val_loss = total_val_loss / len(validation_loader)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr}")

        scheduler.step(avg_val_loss)

    # --- TESTING PHASE ---
    cnn.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = cnn(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cnn_metrics = get_metrics(y_true=all_labels, y_pred=all_preds)
    print("\nCNN Metrics:")
    for metric, value in cnn_metrics.items():
        print(f"{metric.capitalize()}: {value}")

if __name__ == "__main__":
    main()