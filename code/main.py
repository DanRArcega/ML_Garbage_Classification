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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    print("Building dataloaders...")
    train_loader, test_loader, validation_loader = build_dataloaders(DATA_CONFIG, CLASSES)
    print("Dataloaders ready.")

    # If KNN or Logistic Regression are to be used, load data into memory from dataloaders
    scaled_train_pixels = []
    encoded_train_labels = []
    scaled_test_pixels = []
    encoded_test_labels = []
    if args.knn or args.lr:
        scaled_train_pixels, encoded_train_labels = utils.extract_dataloader_data(train_loader)
        scaled_test_pixels, encoded_test_labels = utils.extract_dataloader_data(test_loader)

    all_metrics = dict()

    if args.knn:

        # Load data
        train_pixels, train_labels = utils.extract_dataloader_data(train_loader)
        test_pixels, test_labels = utils.extract_dataloader_data(test_loader)
        evaluation_pixels, evaluation_labels = utils.extract_dataloader_data(validation_loader)

        # Scale pixel values
        scaler = StandardScaler()
        scaled_train_pixels = scaler.fit_transform(train_pixels)
        scaled_test_pixels = scaler.transform(test_pixels)
        scaled_evaluation_pixels = scaler.transform(evaluation_pixels)

        # Encode labels
        label_encoder = LabelEncoder()
        encoded_train_labels = label_encoder.fit_transform(train_labels)
        encoded_test_labels = label_encoder.transform(test_labels)
        encoded_evaluation_labels = label_encoder.transform(evaluation_labels)

        # Find 'optimal' k for KNN
        k_values = range(1, 50, 2)
        knn_accuracies = []
        for k in k_values:
            knn_model = KNeighborsClassifier(n_neighbors=k)
            knn_model.fit(scaled_train_pixels, encoded_train_labels)
            knn_predicted_labels = knn_model.predict(scaled_evaluation_pixels)
            knn_metrics = get_metrics(y_true=encoded_evaluation_labels, y_pred=knn_predicted_labels)
            knn_accuracies.append(knn_metrics['f1_score'])

        best_k = k_values[np.argmax(knn_accuracies)]
        print(f"Best k for KNN: {best_k} with F1 Score: {max(knn_accuracies)}")

        final_knn = KNeighborsClassifier(n_neighbors=best_k)
        final_knn.fit(scaled_train_pixels, encoded_train_labels)
        final_test_predictions = final_knn.predict(scaled_test_pixels)
        final_test_metrics = get_metrics(y_true=encoded_test_labels, y_pred=final_test_predictions)

        print("KNN Metrics:")
        for metric, value in final_test_metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")

        all_metrics["KNN"] = final_test_metrics
        # Create and save confusion matrix
        cnn_matrix = metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(y_true=encoded_test_labels,
                                                                             y_pred=final_test_predictions),
                                                                             display_labels=CLASSES.names)
        cnn_matrix.plot()
        plt.savefig(output_dir + "/knn_confusion_matrix.png")
        print("Saving confusion matrix...")
        plt.clf()


    if args.lr:
        # Train Logistic Regression
        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(scaled_train_pixels, encoded_train_labels)

        # Predict with Logistic Regression
        lr_predicted_labels = lr_model.predict(scaled_test_pixels)

        # Get metrics for Logistic Regression
        lr_metrics = get_metrics(y_true=encoded_test_labels, y_pred=lr_predicted_labels)
        print("\nLogistic Regression Metrics:")
        for metric, value in lr_metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        all_metrics["LR"] = lr_metrics

        # Create and save confusion matrix
        cnn_matrix = metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(y_true=encoded_test_labels,
                                                                             y_pred=lr_predicted_labels),
                                                                             display_labels=CLASSES.names)
        cnn_matrix.plot()
        plt.savefig(output_dir + "/lr_confusion_matrix.png")
        print("Saving confusion matrix...")
        plt.clf()





    # Training CNN
    num_epochs = 50

    # Set up early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 5

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

        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(cnn.state_dict(), "best_cnn_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print("Early stopping triggered. No improvement in validation loss for 8 epochs.")
                break

    # --- TESTING PHASE ---
    # Load the best model and evaluate on the test set
    cnn.load_state_dict(torch.load("best_cnn_model.pth", map_location=device))
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
        print(f"{metric.capitalize()}: {value:.4f}")

    # Create and save confusion matrix
    cnn_matrix = metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(y_true = all_labels, y_pred = all_preds), display_labels=CLASSES.names)
    cnn_matrix.plot()
    plt.savefig(output_dir + "/cnn_confusion_matrix.png")
    print("Saving confusion matrix...")
    plt.clf()

    # Create accuracy and f1_score comparison graphs
    plt.close()

if __name__ == "__main__":
    main()
