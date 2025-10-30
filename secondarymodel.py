import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
import numpy as np  
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
   
 
IMG_SIZE = (150,50)  
BATCH_SIZE = 50
EPOCHS = 25


dataset_path = "path_to_dataset"  
train_dir = os.path.join(dataset_path, "train")
validation_dir = os.path.join(dataset_path, "validation")


train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.3,
    horizontal_flip=True
)


validation_datagen = ImageDataGenerator(rescale=1.0/255)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'  # Use 'categorical' if more than 2 classes
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode_
