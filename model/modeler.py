import numpy as np
import os
import cv2
import random
import json
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
    ModelCheckpoint,
)
import tensorflow as tf
import datetime
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.metrics import AUC, Precision, Recall


# =========================
# Utility Functions
# =========================

def is_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()
        return True
    except Exception:
        return False


def clean_ds_store(path):
    """Remove macOS hidden files safely"""
    for root, dirs, files in os.walk(path):
        for file in files:
            if file == ".DS_Store":
                os.remove(os.path.join(root, file))


# =========================
# Data Loading
# =========================

def get_samples():
    data_dir = os.path.join(os.getcwd(), "model", "cropped")
    paths = []
    img_formats = ["jpeg", "png", "jpg"]

    if not os.path.exists(data_dir):
        print("Data directory not found:", data_dir)
        return paths

    for directory in os.listdir(data_dir):
        dir_path = os.path.join(data_dir, directory)

        # 🔥 Skip non-directories (.DS_Store fix)
        if not os.path.isdir(dir_path):
            continue

        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)

            if (
                os.path.isfile(file_path)
                and file.split(".")[-1].lower() in img_formats
                and is_image(file_path)
            ):
                paths.append(file_path)

    random.shuffle(paths)
    return paths


def get_test_samples(size):
    data_dir = os.path.join(os.getcwd(), "tests")
    paths = []
    sample_imgs = []
    img_formats = ["jpeg", "png", "jpg"]

    for directory in os.listdir(data_dir):
        dir_path = os.path.join(data_dir, directory)

        if not os.path.isdir(dir_path):
            continue

        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)

            if (
                os.path.isfile(file_path)
                and file.split(".")[-1].lower() in img_formats
                and is_image(file_path)
            ):
                paths.append(file_path)

    random.shuffle(paths)

    for image_path in paths:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (size, size))
        sample_imgs.append(image)

    return np.array(sample_imgs)


def get_test_sample(img_name, size):
    img_formats = ["jpeg", "png", "jpg"]
    img = []
    test_dir = os.path.join(os.getcwd(), "tests")

    for file in os.listdir(test_dir):
        if img_name == file:
            file_path = os.path.join(test_dir, file)

            if (
                os.path.isfile(file_path)
                and file.split(".")[-1].lower() in img_formats
                and is_image(file_path)
            ):
                image = cv2.imread(file_path)
                image = cv2.resize(image, (size, size))
                img.append(image)

    return np.array(img)


# =========================
# Classification
# =========================

def classify(img_paths, size):
    read_images = []
    properties = []

    for image_path in img_paths:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (size, size))

        read_images.append(image)

        properties.append(
            1 if "yes" in os.path.normpath(image_path).split(os.path.sep) else 0
        )

    mn_list = list(zip(read_images, properties))
    random.shuffle(mn_list)

    read_images, properties = zip(*mn_list)

    return np.array(read_images), np.array(properties)


# =========================
# Training
# =========================

def train(read_images, properties, size=50):

    classes = 1
    train_len = len(read_images) - 400

    valid_data = read_images[train_len:]
    valid_prop = properties[train_len:]

    train_data = read_images[:train_len]
    train_prop = properties[:train_len]

    print(
        f"Training with {len(train_data)} images and validating with {len(valid_data)} images"
    )

    train_data = train_data.astype("float32") / 255.0
    valid_data = valid_data.astype("float32") / 255.0

    model = keras.Sequential(
        [
            keras.Input(shape=(size, size, 3)),
            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.5),
            layers.Flatten(),
            layers.Dense(
                128,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(0.001),
            ),
            layers.Dense(classes, activation="sigmoid"),
        ]
    )

    optimizer = AdamW(learning_rate=0.001, weight_decay=1e-5)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            "accuracy",
            AUC(name="auc"),
            Precision(name="precision"),
            Recall(name="recall"),
        ],
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
    )

    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("history", exist_ok=True)

    log_dir = os.path.join(
        "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    tensorboard_callback = TensorBoard(log_dir=log_dir)

    checkpoint = ModelCheckpoint(
        "models/best_model.keras",
        monitor="val_precision",
        save_best_only=True,
        mode="max",
    )

    history = model.fit(
        train_data,
        train_prop,
        epochs=100,
        validation_data=(valid_data, valid_prop),
        callbacks=[early_stopping, reduce_lr, tensorboard_callback, checkpoint],
    )

    with open("history/history_latest.json", "w") as f:
        json.dump(history.history, f)

    model.save("models/final_model.keras")

    return model


# =========================
# Main
# =========================

if __name__ == "__main__":
    clean_ds_store(os.getcwd())
    size = 50
    samples = get_samples()
    read_images, properties = classify(samples, size)
    train(read_images, properties, size)