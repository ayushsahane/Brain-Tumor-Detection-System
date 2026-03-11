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


def get_model(num=0):
    model = tf.keras.models.load_model(f"model/models/test_model_{num}.keras")
    
    # Load samples for evaluation
    samples = get_samples()
    read_images, properties = classify(samples, 50)  # Adjust size to match training size
    train_len = len(read_images) - 400
    valid_data = read_images[train_len:]
    valid_prop = properties[train_len:]

    # Evaluate the model on validation data
    loss, acc, precision, recall, auc = model.evaluate(valid_data, valid_prop, verbose=2)

    metrics = {
        'accuracy': acc,
        'loss': loss,
        'precision': precision,
        'recall': recall,
        'auc': auc
    }

    return model, metrics


if __name__ == "__main__":
    size = 50  # Ensure this matches the size used in training
    model, _ = get_model()
    samples = get_samples()
    read_images, properties = classify(samples, size)
    train_len = len(read_images) - 400
    train_img = read_images[:train_len]
    train_lbl = properties[:train_len]
    ev = model.evaluate(train_img, train_lbl, verbose=2)

    print(model.summary())
    print(f"Model restored; {ev}")

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
