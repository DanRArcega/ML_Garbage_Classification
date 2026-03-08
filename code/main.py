import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import numpy as np

from utils import get_metrics, load_train_test_imgs, convert_to_grayscale, sample_per_label

def main():
    train_imgs, test_imgs, train_labels, test_labels = load_train_test_imgs()

    # Get 3 samples per label for training and testing
    train_imgs, train_labels = sample_per_label(train_imgs, train_labels, n_per_label=10)
    test_imgs, test_labels = sample_per_label(test_imgs, test_labels, n_per_label=10)

    train_imgs = convert_to_grayscale(train_imgs)
    test_imgs = convert_to_grayscale(test_imgs)

    # Flatten pixel values
    train_pixels = [list(img.get_flattened_data()) for img in train_imgs]
    test_pixels = [list(img.get_flattened_data()) for img in test_imgs]

    # Scale pixels
    scaler = StandardScaler()
    scaled_train_pixels = scaler.fit_transform(train_pixels)
    scaled_test_pixels = scaler.transform(test_pixels)

    scaled_train_pixels = np.array(scaled_train_pixels)
    scaled_test_pixels = np.array(scaled_test_pixels)

    # Apply PCA
    pca = PCA(n_components=50)
    pca_train_pixels = pca.fit_transform(scaled_train_pixels)
    pca_test_pixels = pca.transform(scaled_test_pixels)

    scaled_train_pixels = pca_train_pixels
    scaled_test_pixels = pca_test_pixels

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_train_labels = label_encoder.fit_transform(train_labels)
    encoded_test_labels = label_encoder.transform(test_labels)

    # Train KNN
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(scaled_train_pixels, encoded_train_labels)

    # Predict
    predicted_labels = knn_model.predict(scaled_test_pixels)

    # Get metrics for KNN
    knn_metrics = get_metrics(y_true=encoded_test_labels, y_pred=predicted_labels)

    print("KNN Metrics:")
    for metric, value in knn_metrics.items():
        print(f"{metric.capitalize()}: {value}")

    # Train Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(scaled_train_pixels, encoded_train_labels)

    # Predict with Logistic Regression
    lr_predicted_labels = lr_model.predict(scaled_test_pixels)

    # Get metrics for Logistic Regression
    lr_metrics = get_metrics(y_true=encoded_test_labels, y_pred=lr_predicted_labels)
    print("\nLogistic Regression Metrics:")
    for metric, value in lr_metrics.items():
        print(f"{metric.capitalize()}: {value}")


if __name__ == "__main__":
    main()