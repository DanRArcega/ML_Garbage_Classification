import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler, LabelEncoder

from utils import get_metrics, load_train_test_imgs, convert_to_grayscale, sample_per_label
from CNN import GarbageClassificationCNN

def main():
    train_imgs, test_imgs, train_labels, test_labels = load_train_test_imgs()

    train_imgs, train_labels = sample_per_label(train_imgs, train_labels, n_per_label=50)
    test_imgs, test_labels   = sample_per_label(test_imgs,  test_labels,  n_per_label=50)

    train_imgs = convert_to_grayscale(train_imgs)
    test_imgs  = convert_to_grayscale(test_imgs)

    # Flatten pixel values
    train_pixels = [list(img.getdata()) for img in train_imgs]
    test_pixels = [list(img.getdata()) for img in test_imgs]

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

    # # Train KNN
    # knn_model = KNeighborsClassifier(n_neighbors=5)
    # knn_model.fit(scaled_train_pixels, encoded_train_labels)

    # # Predict
    # predicted_labels = knn_model.predict(scaled_test_pixels)

    # # Get metrics for KNN
    # knn_metrics = get_metrics(y_true=encoded_test_labels, y_pred=predicted_labels)

    # print("KNN Metrics:")
    # for metric, value in knn_metrics.items():
    #     print(f"{metric.capitalize()}: {value}")

    # # Train Logistic Regression
    # lr_model = LogisticRegression(max_iter=1000)
    # lr_model.fit(scaled_train_pixels, encoded_train_labels)

    # # Predict with Logistic Regression
    # lr_predicted_labels = lr_model.predict(scaled_test_pixels)

    # # Get metrics for Logistic Regression
    # lr_metrics = get_metrics(y_true=encoded_test_labels, y_pred=lr_predicted_labels)
    # print("\nLogistic Regression Metrics:")
    # for metric, value in lr_metrics.items():
    #     print(f"{metric.capitalize()}: {value}")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cnn = GarbageClassificationCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

    # Normalize pixel values to [0, 1]
    def imgs_to_4d_tensor(imgs, device):
        arrays = np.stack([np.array(img, dtype=np.float32) / 255.0 for img in imgs])
        return torch.tensor(arrays[:, np.newaxis, :, :], dtype=torch.float32).to(device)
    
    X_train_tensor = imgs_to_4d_tensor(train_imgs, device)
    X_test_tensor  = imgs_to_4d_tensor(test_imgs, device)
    y_train_tensor = torch.tensor(encoded_train_labels, dtype=torch.long).to(device)
    y_test_tensor  = torch.tensor(encoded_test_labels,  dtype=torch.long).to(device)

    num_epochs = 20
    batch_size = 32

    for epoch in range(num_epochs):
        cnn.train()
        for i in range(0, len(X_train_tensor), batch_size):
            X_batch = X_train_tensor[i:i + batch_size]
            y_batch = y_train_tensor[i:i + batch_size]

            outputs = cnn(X_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    cnn.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_test_tensor), batch_size):
            X_batch = X_test_tensor[i:i + batch_size]
            outputs = cnn(X_batch)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu())

    predicted_labels = torch.cat(all_preds).numpy()
    y_test_cpu = y_test_tensor.cpu().numpy()

    cnn_metrics = get_metrics(y_true=y_test_cpu, y_pred=predicted_labels)
    print("\nCNN Metrics:")
    for metric, value in cnn_metrics.items():
        print(f"{metric.capitalize()}: {value}")

if __name__ == "__main__":
    main()
