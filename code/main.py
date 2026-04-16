import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler, LabelEncoder

import utils
from utils import get_metrics, load_train_test_imgs, convert_to_grayscale, sample_per_label, new_load_train_test_imgs
from preprocessing.dataset import build_dataloaders
from preprocessing.config import DataConfig, DATA_CONFIG, Classes, CLASSES
from CNN import GarbageClassificationCNN
import matplotlib.pyplot as plt
import argparse

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--knn", action = 'store_true', help = "Use KNN model" )
    # parser.add_argument("--lr", action = 'store_true', help = "Use logistic regression model" )
    # args = parser.parse_args()
    # output_dir = "../data/graphs"
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cnn = GarbageClassificationCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    print("Building dataloaders...")
    train_loader, test_loader, validation_loader = build_dataloaders(DATA_CONFIG, CLASSES)
    print("Dataloaders ready.")

    # # If KNN or Logistic Regression are to be used, load data into memory from dataloaders
    # scaled_train_pixels = []
    # encoded_train_labels = []
    # scaled_test_pixels = []
    # encoded_test_labels = []
    # if args.knn or args.lr:
    #     scaled_train_pixels, encoded_train_labels = utils.extract_dataloader_data(train_loader)
    #     scaled_test_pixels, encoded_test_labels = utils.extract_dataloader_data(test_loader)

    # if args.knn:

    #     # Train KNN
    #     knn_model = KNeighborsClassifier(n_neighbors=5)
    #     knn_model.fit(scaled_train_pixels, encoded_train_labels)

    #     # Predict
    #     knn_predicted_labels = knn_model.predict(scaled_test_pixels)

    #     # Get metrics for KNN
    #     knn_metrics = get_metrics(y_true=encoded_test_labels, y_pred=knn_predicted_labels)

    #     print("KNN Metrics:")
    #     for metric, value in knn_metrics.items():
    #         print(f"{metric.capitalize()}: {value}")

    #     # Create and save confusion matrix
    #     cnn_matrix = metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(y_true=encoded_test_labels,
    #                                                                          y_pred=knn_predicted_labels),
    #                                                                          display_labels=CLASSES.names)
    #     cnn_matrix.plot()
    #     plt.savefig(output_dir + "/knn_confusion_matrix.png")
    #     print("Saving confusion matrix...")

    # if args.lr:
    #     # Train Logistic Regression
    #     lr_model = LogisticRegression(max_iter=1000)
    #     lr_model.fit(scaled_train_pixels, encoded_train_labels)

    #     # Predict with Logistic Regression
    #     lr_predicted_labels = lr_model.predict(scaled_test_pixels)

    #     # Get metrics for Logistic Regression
    #     lr_metrics = get_metrics(y_true=encoded_test_labels, y_pred=lr_predicted_labels)
    #     print("\nLogistic Regression Metrics:")
    #     for metric, value in lr_metrics.items():
    #         print(f"{metric.capitalize()}: {value}")

    #     # Create and save confusion matrix
    #     cnn_matrix = metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(y_true=encoded_test_labels,
    #                                                                          y_pred=lr_predicted_labels),
    #                                                                          display_labels=CLASSES.names)
    #     cnn_matrix.plot()
    #     plt.savefig(output_dir + "/lr_confusion_matrix.png")
    #     print("Saving confusion matrix...")

    # Training CNN
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

    # Create and save confusion matrix
    # cnn_matrix = metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(y_true = all_labels, y_pred = all_preds), display_labels=CLASSES.names)
    # cnn_matrix.plot()
    # plt.savefig(output_dir + "/cnn_confusion_matrix.png")
    # print("Saving confusion matrix...")
    # plt.close()

if __name__ == "__main__":
    main()
