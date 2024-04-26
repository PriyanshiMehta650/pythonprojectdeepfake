

Original file is located at
    https://colab.research.google.com/drive/1HOrGcT9zRDCR-Yy-KcjZCX1rWoV7W_bP
"""

# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES
# TO THE CORRECT LOCATION (/kaggle/input) IN YOUR NOTEBOOK,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil

CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'deepfake-and-real-images:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F1909705%2F3134515%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240423%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240423T150333Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D9d7974db86130309a95a1a80e03b408f0b5a05fb559e3342d493867e42fbf3d38cad808ad2bdba108d4c78d6ffd930e24dce953182ad1dd45467f4c8f77ecb65aebe899d1b66a284031e0888b0a90de951566906b2b4ddf47f81cc4ec696d93f1bff9b0dcf3a4c45e078c91a825f4e1852724d779577a833ec0639b32615f1fec1fe9f74e83b3ed5480f2408732a87ac85ee5b82bc67490fbc70df31bc34b3ac956a228d09e0bf2b1b896f5247ae60e8945125349de34ead2bd8bbe55a592a0e35137ad68a56c6f7cdafb894d6c3b507c98bc693be640a045c4a22952aa03e29bbda1add49b94225f7b0d1b36ba232a804e47e0596cdcc8de333e523b9e5f872'

KAGGLE_INPUT_PATH='/kaggle/input'
KAGGLE_WORKING_PATH='/kaggle/working'
KAGGLE_SYMLINK='kaggle'

!umount /kaggle/input/ 2> /dev/null
shutil.rmtree('/kaggle/input', ignore_errors=True)
os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

try:
  os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
except FileExistsError:
  pass
try:
  os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
except FileExistsError:
  pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = fileres.headers['content-length']
            print(f'Downloading {directory}, {total_length} bytes compressed')
            dl = 0
            data = fileres.read(CHUNK_SIZE)
            while len(data) > 0:
                dl += len(data)
                tfile.write(data)
                done = int(50 * dl / int(total_length))
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                sys.stdout.flush()
                data = fileres.read(CHUNK_SIZE)
            if filename.endswith('.zip'):
              with ZipFile(tfile) as zfile:
                zfile.extractall(destination_path)
            else:
              with tarfile.open(tfile.name) as tarfile:
                tarfile.extractall(destination_path)
            print(f'\nDownloaded and uncompressed: {directory}')
    except HTTPError as e:
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
        continue
    except OSError as e:
        print(f'Failed to load {download_url} to path {destination_path}')
        continue

print('Data source import complete.')

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames[:10]:
        print(os.path.join(dirname, filename))
import os

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import numpy as np
import random
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.optimizers import Adam

train_folder = '/kaggle/input/deepfake-and-real-images/Dataset/Train'
valid_folder = '/kaggle/input/deepfake-and-real-images/Dataset/Validation'
test_folder = '/kaggle/input/deepfake-and-real-images/Dataset/Test'

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
])

# Modify the data augmentation
data_augmentation.layers[1].rotation_range = 0.3  # Change the rotation range

# Print the modified data augmentation
print(data_augmentation)

# Function to load and preprocess images
def load_and_preprocess_images(folder, num_images):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        for filename in random.sample(os.listdir(label_folder), num_images):
            img_path = os.path.join(label_folder, filename)
            try:
                image = cv2.imread(img_path)  # Load image using OpenCV
                if image is not None:
                    image = cv2.resize(image, (224, 224))  # Resize image
                    images.append(image)
                    labels.append(label)
            except Exception as e:
                print(f"Error processing image {img_path}: {str(e)}")
    return np.array(images), np.array(labels)

num_images_per_folder = 100

# Load and preprocess a sample of images for training, testing, and validation
train_images, train_labels = load_and_preprocess_images(train_folder, num_images_per_folder)
test_images, test_labels = load_and_preprocess_images(test_folder, num_images_per_folder)
validation_images, validation_labels = load_and_preprocess_images(valid_folder, num_images_per_folder)

# Create a new model with the pre-trained ResNet50 base and custom classification head
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)

# Freeze the pre-trained ResNet50 layers
base_model.trainable = False

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Create label encoder
label_encoder = LabelEncoder()

# Fit label encoder and transform labels to integers
train_labels_encoded = label_encoder.fit_transform(train_labels)
validation_labels_encoded = label_encoder.transform(validation_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Train the model with encoded labels
history_fine = model.fit(train_images, train_labels_encoded, epochs=10, batch_size=32, validation_data=(validation_images, validation_labels_encoded))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels_encoded)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Calculate precision, recall, and F1 score
predictions = model.predict(test_images)
predictions_binary = np.where(predictions > 0.5, 1, 0)

precision = precision_score(test_labels_encoded, predictions_binary)
recall = recall_score(test_labels_encoded, predictions_binary)
f1 = f1_score(test_labels_encoded, predictions_binary)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Plot training and validation loss
plt.plot(history_fine.history['loss'], label='loss')
plt.plot(history_fine.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 2])
plt.legend(loc='upper right')
plt.show()

model.save('my_model.keras')

import os

file_path = 'my_model.keras'  # Replace with the actual file path
if os.path.exists(file_path):
    print("The model has been saved successfully.")
else:
    print("The model has not been saved.")

import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('my_model.keras')

# Optionally, unfreeze some layers for fine-tuning
# Example: Unfreeze the last few layers
for layer in model.layers[:-5]:
    layer.trainable = False

# Compile the model with a smaller learning rate for fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Continue training the model on additional data or the same data with a smaller learning rate
history_fine_tuned = model.fit(train_images, train_labels_encoded, epochs=8, batch_size=12, validation_data=(validation_images, validation_labels_encoded))

# Plot training history
plt.plot(history_fine_tuned.history['accuracy'], label='accuracy')
plt.plot(history_fine_tuned.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels_encoded)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

from google.colab import files
import tensorflow as tf
import numpy as np
import cv2

# Load the saved model
model = tf.keras.models.load_model('my_model.keras')

# Function to load and preprocess an image
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)  # Load image using OpenCV
    if image is not None:
        image = cv2.resize(image, (224, 224))  # Resize image
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = image / 255.0  # Normalize the image
        return image
    else:
        return None

# Function to make a prediction
def predict_image_real_or_fake(image_path):
    image = load_and_preprocess_image(image_path)
    if image is not None:
        prediction = model.predict(image)
        if prediction[0][0] > 0.5:
            return "Fake"
        else:
            return "Real"
    else:
        return "Error: Unable to process the image"

# Upload an image file
uploaded = files.upload()

# Get the file path of the uploaded image
file_path = next(iter(uploaded))

# Make a prediction based on the uploaded image
prediction = predict_image_real_or_fake(file_path)
print("Prediction:", prediction)

!pip install fastapi uvicorn

!pip install python-multipart

!pip install nest_asyncio
!pip install pyngrok

from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
import cv2
import asyncio
from starlette.responses import JSONResponse
from pyngrok import ngrok

# Load the saved model
model = tf.keras.models.load_model('my_model.keras')

# Function to load and preprocess an image
async def load_and_preprocess_image(image_data):
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)  # Load image from bytes
    if image is not None:
        image = cv2.resize(image, (224, 224))  # Resize image
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = image / 255.0  # Normalize the image
        return image
    else:
        return None

# Function to make a prediction
async def predict_image_real_or_fake(image_data):
    image = await load_and_preprocess_image(image_data)
    if image is not None:
        prediction = model.predict(image)
        if prediction[0][0] > 0.5:
            return "Real"
        else:
            return "Fake"
    else:
        return "Error: Unable to process the image"

# Initialize the FastAPI app
app = FastAPI()

# Endpoint for image prediction
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image_data = await file.read()
    prediction = await predict_image_real_or_fake(image_data)
    return JSONResponse(content={"prediction": prediction})

if __name__ == "__main__":
    # Set your ngrok authentication token
    ngrok.set_auth_token("2fbAAz6AEbsNhHRBLZ19EFaHbwv_NREAXPV5njvyznBDZR2T")

    # Start ngrok and get the public URL
    public_url = ngrok.connect(8000).public_url
    print(f"FastAPI app available at {public_url}/predict")

    # Run the FastAPI application
    async def main():
        await uvicorn.run(app, host="0.0.0.0", port=8000)

    asyncio.run(main())
