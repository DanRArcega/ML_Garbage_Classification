"""
file: svm_classifier.py
description: An implementation of a svm classifier for garbage image classification. Runs all combinations of feature
and kernel modes. Includes PCA dimensionality reduction and hyperparameter tuning.
language: python3
author: Sam Whitney, shw9601@rit.edu
"""


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

from features.feature_extractor import extract_features
from preprocessing.config import DATA_CONFIG



PARAMETER_GRIDS = {
    "rbf": {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.001, 0.01]
    },
    "poly": {
        "C": [0.1, 1, 10],
        "degree": [2, 3, 4],
        "gamma": ["scale", "auto"]
    },
    "linear": {
        "C": [0.01, 0.1, 1, 10]
    }
}


PCA_VARIANCE = 0.95
CV_FOLDS = 5
MODES = ("hog", "color", "spatial_2x2", "spatial_3x3", "both")
KERNELS = ("rbf", "poly", "linear")



def apply_pca(
        X_train: np.ndarray,
        X_eval: np.ndarray,
        variance: float = PCA_VARIANCE
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """
    Fits a PCA on training data and transforms training and testing features.
    :param X_train: The training feature set.
    :type X_train: np.ndarray
    :param X_eval: The evaluation feature set.
    :type X_eval: np.ndarray
    :param variance: The variance proportion to retain (defaults to 95%).
    :type variance: float
    :return: The transformed training and testing features and the PCA.
    :rtype: tuple[np.ndarray, np.ndarray, PCA]
    """
    pca = PCA(n_components = variance, svd_solver = "full")
    X_train_reduced = pca.fit_transform(X_train)
    X_eval_reduced = pca.transform(X_eval)
    print("PCA:")
    print(f"    Training: {X_train.shape} -> {X_train_reduced.shape}")
    print(f"    Testing: {X_eval.shape} -> {X_eval_reduced.shape}")
    return X_train_reduced, X_eval_reduced, pca



def train_and_evaluate(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_eval: np.ndarray,
        y_eval: np.ndarray,
        kernel: str
) -> dict:
    """
    Trains an SVM with GridSearchCV and evaluates on the evaluation set (validation or test, depending on phase).
    :param X_train: The training feature set.
    :type X_train: np.ndarray
    :param y_train: The training label set.
    :type y_train: np.ndarray
    :param X_eval: The evaluation feature set.
    :type X_eval: np.ndarray
    :param y_eval: The evaluation label set.
    :type y_eval: np.ndarray
    :param kernel: The SVM kernel type (rbf, linear or poly).
    :type kernel: str
    :return: A dictionary of metrics and the best hyperparameters.
    :rtype: dict
    """
    print("\n" + ("= " * 50))
    print("Running GridSearchCV with parameters:")
    print(f"    Kernel: {kernel}")
    print(f"    Folds: {CV_FOLDS}")
    svm = SVC(kernel = kernel)
    grid_search = GridSearchCV(
        svm,
        PARAMETER_GRIDS[kernel],
        cv = CV_FOLDS,
        scoring = "accuracy",
        n_jobs = -2,
        verbose = 1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print(f"    Best Parameters: {grid_search.best_params_}")
    print(f"    Best CV Accuracy: {grid_search.best_score_:.4f}")

    y_pred = best_model.predict(X_eval)
    return {
        "accuracy": accuracy_score(y_eval, y_pred),
        "precision": precision_score(y_eval, y_pred, average = "macro", zero_division = 0),
        "recall": recall_score(y_eval, y_pred, average = "macro", zero_division = 0),
        "f1": f1_score(y_eval, y_pred, average = "macro", zero_division = 0),
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "best_estimator": best_model
    }



def evaluate_final(
        manifest_path: Path,
        mode: str,
        kernel: str,
        best_params: dict,
        hog_scaler: StandardScaler,
        color_scaler: StandardScaler,
        label_encoder: LabelEncoder,
        pca: PCA
) -> dict:
    """
    Evaluates the most successful model and its parameters against the testing set and reports the final metrics.
    :param manifest_path: The path to the manifest csv.
    :type manifest_path: Path
    :param mode: The best feature mode from training and validation.
    :type mode: str
    :param kernel: The best kernel from training and validation.
    :type kernel: str
    :param best_params: The best hyperparameters from GridSearchCV.
    :type best_params: dict
    :param hog_scaler: Fitted HOG scaler from training and validation.
    :type hog_scaler: StandardScaler
    :param color_scaler: Fitted color scaler from training and validation.
    :type color_scaler: StandardScaler
    :param label_encoder: Fitted label encoder from training and validation.
    :type label_encoder: LabelEncoder
    :param pca: The fitted PCA from training and validation.
    :type pca: PCA
    :return: A dictionary of metrics from the final evaluation of the test set.
    :rtype: dict
    """
    X_train, y_train, _, _, _ = extract_features(
        manifest_path,
        split = "train",
        mode = mode,
        hog_scaler = hog_scaler,
        color_scaler = color_scaler,
        label_encoder = label_encoder
    )
    X_test, y_test, _, _, _ = extract_features(
        manifest_path,
        split = "test",
        mode = mode,
        hog_scaler = hog_scaler,
        color_scaler = color_scaler,
        label_encoder = label_encoder
    )

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    svm = SVC(kernel = kernel, **best_params)
    svm.fit(X_train_pca, y_train)
    y_pred = svm.predict(X_test_pca)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average = "macro", zero_division = 0),
        "recall": recall_score(y_test, y_pred, average = "macro", zero_division = 0),
        "f1": f1_score(y_test, y_pred, average = "macro", zero_division = 0),
    }

    print("\nFinal Test Metrics:")
    for metric, value in metrics.items():
        print(f"    {metric.capitalize():<12} : {value:.4f}")
    return metrics



def print_results(results: list[dict]) -> None:
    """
    Prints out a table of all validation results.
    :param results: The list of results from running each model combination.
    :type results: list[dict]
    :return: None.
    :rtype: None
    """
    print("\n" + ("=" * 80))
    print("VALIDATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Mode':<10} {'Kernel':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 80)
    for result in results:
        print(f"{result["mode"]:<10} {result["kernel"]:<10} "
        f"{result["accuracy"]:<12.4f} {result["precision"]:<12.4f} "
              f"{result["recall"]:<12.4f} {result["f1"]:<12.4f}")
        print("=" * 80)

    best = max(results, key = lambda x: x["f1"])
    print(f"\nBest combination: mode = {best["mode"]}, kernel = {best["kernel"]}",
          f"F1 = {best["f1"]:.4f}, Accuracy = {best["accuracy"]:.4f}")
    print(f"Best Parameters:    {best['best_params']}")




def main():
    manifest_path = DATA_CONFIG.processed_data_path / "manifest.csv"
    results = []

    training_start = time.time()
    for mode in tqdm(MODES, desc = "Feature Extraction Modes", unit = "mode"):
        print(f"\n{"=" * 60}")
        print(f"Extracting features - Mode: {mode}")
        print("=" * 60)

        X_train, y_train, scalers, label_encoder = extract_features(
            manifest_path,
            split = "train",
            mode = mode
        )
        X_val, y_val, _, _ = extract_features(
            manifest_path,
            split = "validation",
            mode = mode,
            scalers = scalers,
            label_encoder = label_encoder
        )

        X_train_pca, X_val_pca, pca = apply_pca(X_train, X_val)

        for kernel in tqdm(KERNELS, desc = f"{mode} kernels", unit = "kernel", leave = False):
            print(f"\n--- Mode: {mode} | Kernel: {kernel} ---")
            metrics = train_and_evaluate(
                X_train = X_train_pca,
                y_train = y_train,
                X_eval = X_val_pca,
                y_eval = y_val,
                kernel = kernel
            )
            results.append({
                "mode": mode,
                "kernel": kernel,
                "scalers": scalers,
                "label_encoder": label_encoder,
                "pca": pca,
                **{k: v for k, v in metrics.items() if k != "best_estimator"}
            })

    print_results(results)
    results_path = DATA_CONFIG.processed_data_path / "results.csv"
    csv_results = [{k: v for k, v in result.items()
                    if k not in ("scalers", "label_encoder", "pca", "best_estimator")}
                   for result in results]
    pd.DataFrame(csv_results).to_csv(results_path, index = False)
    print(f"\nResults saved to {results_path}")
    training_total = time.time() - training_start
    print(f"\nTotal training time: {str(datetime.timedelta(seconds=int(round(training_total))))}")
    testing_start = time.time()
    best = max(results, key = lambda x: x["f1"])
    final_metrics = evaluate_final(
        manifest_path = manifest_path,
        mode = best["mode"],
        kernel = best["kernel"],
        best_params = best["best_params"],
        scalers = best["scalers"],
        label_encoder = best["label_encoder"],
        pca = best["pca"],
    )
    testing_end = time.time() - testing_start
    print(f"\nTesting time: {str(datetime.timedelta(seconds=int(round(testing_end))))}")




if __name__ == "__main__":
    main()