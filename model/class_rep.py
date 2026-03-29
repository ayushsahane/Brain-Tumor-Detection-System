from sklearn import model_selection
from tensorflow.keras import Sequential
from sklearn.metrics import classification_report
from modeler import classify, get_samples
from predictor import get_model
import numpy as np

# Load and preprocess the samples
samples = get_samples()
read_images, properties = classify(samples, 50)  # Ensure this matches the input size in your model

# Split the data
train_len = len(read_images) - 400
valid_data = read_images[train_len:]
valid_prop = properties[train_len:]

# Normalize the data
valid_data = np.array(valid_data).astype("float32") / 255.0

# Load the model
model, _ = get_model(0)

# Make predictions
predicted_probabilities = model.predict(valid_data)
predicted_classes = (predicted_probabilities > 0.5).astype("int32")  # For binary classification

# Generate classification report
report = classification_report(valid_prop, predicted_classes)
print(report)
