import numpy as np
import os
import cv2
import random
import json
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard,ModelCheckpoint
import tensorflow as tf
import datetime
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.metrics import AUC, Precision, Recall

def is_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()
        return True
    except (IOError, SyntaxError):
        return False

def get_samples():
    data_dir = os.path.join(os.getcwd(), "model","cropped")
    paths = []
    img_formats = ["jpeg", "png", "jpg"]

    for directory in os.listdir(data_dir):
        for file in os.listdir(os.path.join(data_dir, directory)):
            file_path = os.path.join(data_dir, directory, file)
            if is_image(file_path) and file_path.split(".")[-1].lower() in img_formats:
                paths.append(file_path)
            else:
                print(file_path, "not recognized")

    random.shuffle(paths)
    return paths

def get_test_samples(size):
    data_dir = os.path.join(os.getcwd(), "tests")
    paths = []
    sample_imgs = []
    img_formats = ["jpeg", "png", "jpg"]

    for directory in os.listdir(data_dir):
        for file in os.listdir(os.path.join(data_dir, directory, file)):
            file_path = os.path.join(data_dir, directory, file)
            if is_image(file_path) and file_path.split(".")[-1].lower() in img_formats:
                paths.append(file_path)
            else:
                print(file_path, "not recognized")

    random.shuffle(paths)

    for image_path in paths:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (size, size))
        sample_imgs.append(image)

    return np.array(sample_imgs)

def get_test_sample(img_name):
    img_formats = ["jpeg", "png", "jpg"]
    img = []

    for file in os.listdir("tests"):
        if img_name == file:
            file_path = os.path.join("tests", file)
            if is_image(file_path) and file_path.split(".")[-1].lower() in img_formats:
                img.append(cv2.resize(cv2.imread(file_path), (size, size)))

    return np.array(img)

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
    read_images = np.array(read_images)
    properties = np.array(properties)

    return read_images, properties

def train(read_images, properties):
    classes = 1
    train_len = len(read_images) - 400
    valid_data = read_images[train_len:]
    valid_prop = properties[train_len:]
    train_data = read_images[:train_len]
    train_prop = properties[:train_len]

    print(
        f"Training with {len(train_data)} images and validating with {len(valid_data)} images"
    )
    print("Shape:", train_data.shape, train_prop.shape)

    train_data = train_data.astype("float32") / 255.0
    valid_data = valid_data.astype("float32") / 255.0

    model = keras.Sequential(
        [
            keras.Input(shape=(size, size, 3)),
            layers.Conv2D(32, (3, 3), padding="SAME", activation="relu"),
            layers.MaxPooling2D((2, 2), strides=2),
            layers.Conv2D(64, (3, 3), padding="SAME", activation="relu"),
            layers.MaxPooling2D((2, 2), strides=2),
            layers.Dropout(0.6),
            layers.Flatten(),
            layers.Dense(
                128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)
            ),
            layers.Dense(classes, activation="sigmoid"),
        ]
    )

    initial_learning_rate = 0.001
    optimizer = AdamW(learning_rate=initial_learning_rate, weight_decay=1e-5)

    loss_function = tf.keras.losses.BinaryCrossentropy()

    metrics = [
        'accuracy',
        AUC(name='auc'),
        Precision(name='precision'),
        Recall(name='recall')
    ]

    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=metrics,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=30, restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001
    )

    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    checkpoint = ModelCheckpoint(
        "best_model_p.keras", 
        monitor='val_precision', 
        verbose=1, 
        save_best_only=True, 
        mode='max'
    )

    history = model.fit(
        train_data,
        train_prop,
        epochs=500,
        validation_data=(valid_data, valid_prop),
        callbacks=[early_stopping, reduce_lr, tensorboard_callback, checkpoint],
    )
    history_dic = history.history
    os.makedirs(os.path.join("history"), exist_ok=True)
    with open(
        os.path.join(
            "history",
            f'history_{len(os.listdir(os.path.join("history")))}.json',
        ),
        "w",
    ) as f:
        json.dump(history_dic, f)

    model.save(
        os.path.join(
            "models", f"test_model_{len(os.listdir(os.path.join('models')))}.keras"
        )
    )

def rm_r_ds_store(path):
    os.system(f"find . -name '.DS_Store' -type f -delete")

if __name__ == "__main__":
    rm_r_ds_store(0)
    size = 50
    samples = get_samples()
    read_images, properties = classify(samples, size)
    train(read_images, properties)
