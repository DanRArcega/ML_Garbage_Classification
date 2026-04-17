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
    parser = argparse.ArgumentParser()
    parser.add_argument("--knn", action = 'store_true', help = "Use KNN model" )
    parser.add_argument("--lr", action = 'store_true', help = "Use logistic regression model" )
    args = parser.parse_args()
    output_dir = "../data/graphs"
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cnn = GarbageClassificationCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

    # Normalize pixel values to [0, 1]
    def imgs_to_4d_tensor(imgs, device):
        arrays = np.stack([np.array(img, dtype=np.float32) / 255.0 for img in imgs])
        return torch.tensor(arrays[:, np.newaxis, :, :], dtype=torch.float32).to(device)


    print("Building dataloaders...")
    train_loader, test_loader, validation_loader = build_dataloaders(DATA_CONFIG, CLASSES)
    print("Dataloaders ready.")
    #X_train_tensor = imgs_to_4d_tensor(train_imgs, device)
    #X_test_tensor  = imgs_to_4d_tensor(test_imgs, device)
    #y_train_tensor = torch.tensor(encoded_train_labels, dtype=torch.long).to(device)
    #y_test_tensor  = torch.tensor(encoded_test_labels,  dtype=torch.long).to(device)

    all_metrics = dict()
    # If KNN or Logistic Regression are to be used, load data into memory from dataloaders
    scaled_train_pixels = []
    encoded_train_labels = []
    scaled_test_pixels = []
    encoded_test_labels = []
    if args.knn or args.lr:
        scaled_train_pixels, encoded_train_labels = utils.extract_dataloader_data(train_loader)
        scaled_test_pixels, encoded_test_labels = utils.extract_dataloader_data(test_loader)

    if args.knn:

        # Train KNN
        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(scaled_train_pixels, encoded_train_labels)

        # Predict
        knn_predicted_labels = knn_model.predict(scaled_test_pixels)

        # Get metrics for KNN
        knn_metrics = get_metrics(y_true=encoded_test_labels, y_pred=knn_predicted_labels)
        all_metrics["knn"] = knn_metrics
        print("KNN Metrics:")
        for metric, value in knn_metrics.items():
            print(f"{metric.capitalize()}: {value}")

        # Create and save confusion matrix
        cnn_matrix = metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(y_true=encoded_test_labels,
                                                                             y_pred=knn_predicted_labels),
                                                                             display_labels=CLASSES.names)
        cnn_matrix.plot()
        plt.savefig(output_dir + "/knn_confusion_matrix.png")
        print("Saving confusion matrix...")

    if args.lr:
        # Train Logistic Regression
        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(scaled_train_pixels, encoded_train_labels)

        # Predict with Logistic Regression
        lr_predicted_labels = lr_model.predict(scaled_test_pixels)

        # Get metrics for Logistic Regression
        lr_metrics = get_metrics(y_true=encoded_test_labels, y_pred=lr_predicted_labels)
        all_metrics["lr"] = lr_metrics
        print("\nLogistic Regression Metrics:")
        for metric, value in lr_metrics.items():
            print(f"{metric.capitalize()}: {value}")

        # Create and save confusion matrix
        cnn_matrix = metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(y_true=encoded_test_labels,
                                                                             y_pred=lr_predicted_labels),
                                                                             display_labels=CLASSES.names)
        cnn_matrix.plot()
        plt.savefig(output_dir + "/lr_confusion_matrix.png")
        print("Saving confusion matrix...")

    # Training CNN
    num_epochs = 20
    #batch_size = 32

    for epoch in range(num_epochs):
        cnn.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = cnn(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

    cnn.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
    #    for i in range(0, len(X_test_tensor), batch_size):
    #        X_batch = X_test_tensor[i:i + batch_size]
            images = images.to(device)
            outputs = cnn(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    #predicted_labels = torch.cat(all_preds).numpy()

    #y_test_cpu = y_test_tensor.cpu().numpy()

    cnn_metrics = get_metrics(y_true = all_labels, y_pred = all_preds)
    all_metrics["cnn"] = cnn_metrics
    print("\nCNN Metrics:")
    for metric, value in cnn_metrics.items():
        print(f"{metric.capitalize()}: {value}")

    # Create and save confusion matrix
    cnn_matrix = metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(y_true = all_labels, y_pred = all_preds), display_labels=CLASSES.names)
    cnn_matrix.plot()
    plt.savefig(output_dir + "/cnn_confusion_matrix.png")
    print("Saving confusion matrix...")

    #Create comparison graphs

    plt.close()
if __name__ == "__main__":
    main()
