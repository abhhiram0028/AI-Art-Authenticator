import tensorflow as tf
import keras
import numpy as np
import os
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import time
import io
model = keras.models.load_model(r'C:\Users\abhir\OneDrive\Desktop\project\SC1.h5')
def test(image_ip, true_label):

  
  progress = 0
  predicted_labels = []

  for image_file in range(1):
    image_data = image_ip.read()
    
    new_image = Image.open(io.BytesIO(image_data))
    new_image = new_image.resize((224, 224))  # Resize to match the input size of the model
    new_image_array = np.array(new_image)
    new_image_array = np.expand_dims(new_image_array, axis=0)
    new_image_array = new_image_array / 255.0  # Normalize pixel values

    # Perform classification
    prediction = model.predict(new_image_array, verbose=0)
    print(prediction)
    if true_label == "AI": # if true label is AI
      if prediction[0][0] < 0.5:
        predicted_labels.append(1) # correct prediction
      else:
        predicted_labels.append(0) # wrong prediction

    if true_label == "nAI": # if actual label in not AI "nAI"

      if prediction[0][0] >= 0.5:
        predicted_labels.append(0) # correct prediction
      else:
        predicted_labels.append(1) # wrong prediction

    progress += 1
    #bar.update(progress)  # Update the progress bar

  return predicted_labels
