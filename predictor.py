import tensorflow as tf
import os
import numpy as np
import cv2
import imghdr
from modeler import get_samples, classify

def get_test_sample(img_name, size):
    img_formats = ["jpeg", "png", "jpg"]
    img = []

    for file in os.listdir("model/tests"):
        if img_name == file:
            file_path = os.path.join("model/tests", file)
            print(file_path)
            if imghdr.what(file_path):
                if imghdr.what(file_path).lower() in img_formats:
                    img.append(cv2.resize(cv2.imread(file_path), (size, size)))

    return np.array(img)

def _compute_threshold_metrics(y_true, y_prob, threshold=0.5):
    y_true = y_true.flatten()
    y_prob = y_prob.flatten()
    y_pred = (y_prob >= threshold).astype(int)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    balanced_accuracy = (recall + specificity) / 2
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp + 1e-8) * (tp + fn + 1e-8) * (tn + fp + 1e-8) * (tn + fn + 1e-8))
    brier = np.mean((y_prob - y_true) ** 2)

    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
        "brier": brier,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }

def get_model(num=0):
    model = tf.keras.models.load_model(f"model/models/test_model_{num}.keras")

    # Load samples for evaluation
    samples = get_samples()
    read_images, properties = classify(samples, 50)  # Adjust size to match training size
    train_len = len(read_images) - 400
    valid_data = read_images[train_len:]
    valid_prop = properties[train_len:]

    # Evaluate the model on validation data using built-in metrics
    loss, acc, precision, recall, auc = model.evaluate(valid_data, valid_prop, verbose=2)

    # Compute richer post-hoc metrics at threshold = 0.5
    probas = model.predict(valid_data, verbose=0).flatten()
    threshold_metrics = _compute_threshold_metrics(valid_prop, probas, threshold=0.5)

    metrics = {
        "accuracy": acc,
        "loss": loss,
        "precision": precision,
        "recall": recall,
        "auc": auc,
        "specificity": threshold_metrics["specificity"],
        "f1": threshold_metrics["f1"],
        "balanced_accuracy": threshold_metrics["balanced_accuracy"],
        "mcc": threshold_metrics["mcc"],
        "brier": threshold_metrics["brier"],
        "tp": threshold_metrics["tp"],
        "tn": threshold_metrics["tn"],
        "fp": threshold_metrics["fp"],
        "fn": threshold_metrics["fn"],
        "n_eval": len(valid_prop),
    }

    return model, metrics

if __name__ == "__main__":
    size = 50  # Ensure this matches the size used in training
    model, metrics = get_model()
    samples = get_samples()
    read_images, properties = classify(samples, size)
    train_len = len(read_images) - 400
    train_img = read_images[:train_len]
    train_lbl = properties[:train_len]
    ev = model.evaluate(train_img, train_lbl, verbose=2)

    print(model.summary())
    print(f"Model restored; {ev}")
    print("Validation metrics:", metrics)

    while True:
        img_name = input("Enter the test image filename: ")
        sample = get_test_sample(img_name, size)
        if sample.size == 0:
            print("Invalid image or file not found. Please try again.")
            continue
        predictions = model.predict(sample)
        pred_class = np.argmax(predictions, axis=-1)
        print("Predictions shape:", predictions.shape)
        print("Prediction class:", pred_class)