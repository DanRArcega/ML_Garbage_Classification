import pandas as pd
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import numpy as np

from utils import load_train_test_imgs, convert_to_grayscale, sample_per_label

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

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_train_labels = label_encoder.fit_transform(train_labels)
    encoded_test_labels = label_encoder.transform(test_labels)

    # Train KNN
    nn_model = NearestNeighbors(n_neighbors=5, algorithm='auto')
    nn_model.fit(scaled_train_pixels)

    # Predict
    distances, indices = nn_model.kneighbors(scaled_test_pixels)

    predicted_labels = []
    for idx in indices:
        neighbor_labels = encoded_train_labels[idx]
        values, counts = np.unique(neighbor_labels, return_counts=True)
        predicted_labels.append(values[np.argmax(counts)])

    # Print accuracy, recall, precision, and F1-score
    cm = confusion_matrix(encoded_test_labels, predicted_labels)
    print("Confusion Matrix:")
    print(cm)

    # Accuracy
    accuracy = np.mean(predicted_labels == encoded_test_labels)

    # Precision, Recall, F1-score (per class)
    precision = []
    recall = []
    f1 = []

    for i in range(len(cm)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        p = tp / (tp + fp) if (tp + fp) != 0 else 0
        r = tp / (tp + fn) if (tp + fn) != 0 else 0
        f = 2 * p * r / (p + r) if (p + r) != 0 else 0

        precision.append(p)
        recall.append(r)
        f1.append(f)

    # Average metrics
    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)
    avg_f1 = np.mean(f1)

    print("Accuracy:", accuracy)
    print("Precision:", avg_precision)
    print("Recall:", avg_recall)
    print("F1-score:", avg_f1)


if __name__ == "__main__":
    main()