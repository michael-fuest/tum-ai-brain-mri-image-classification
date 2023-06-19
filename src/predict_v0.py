import cv2
# import imutils
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import shutil
# import sklearn

import os
from pathlib import Path

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping
from tqdm import tqdm
import keras.utils
import keras.optimizers

#Defining project directories
root_dir = Path.cwd().parent
data_dir = root_dir / 'tum-ai-brain-mri-image-classification' / 'data'
train_dir = data_dir / 'train'
test_dir = data_dir / 'test'
validation_dir = data_dir / 'validate'


# Load training dataset
train_ds = keras.utils.image_dataset_from_directory(
    directory=train_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(244,244)
    )

# Load validation dataset
val_ds = keras.utils.image_dataset_from_directory(
    directory=validation_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(244,244)
    )

# Build Model
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=(244,244,3), num_classes=2)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_ds,
    epochs=25,
    validation_data=val_ds,
)
