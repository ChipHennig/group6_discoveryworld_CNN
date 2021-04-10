from PreliminaryCaching import has_cached_emotions, cache_emotions
from ImageSequence import ImageSequence

import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint
import os
from pathlib import Path



# Number of emotions we are predicting.
EMOTION_COUNT = 11

# Number of images to feed into the CNN for each batch
BATCH_SIZE = 64

# Number of epochs to have the model train for
EPOCHS = 10


def read_cached_emotions(partition):
    """
    Reads in cached emotion data as a Map<Int, Int> that maps an image ID to an emotion.

    @partition Partition of the dataset. Ex: Train, Validation
    """
    return pd.read_csv(f"{Path(__file__).parent}/data/AffectNet/{partition}.csv", header=None, index_col=0, squeeze=True).to_dict()


# TODO Replace with pretrained model
# TODO Maybe make work with color?
def create_model():
    """
    Creates a CNN with 224x224 inputs and `EMOTION_COUNT` outputs.
    """

    # Initialising the CNN
    model = Sequential()

    # 1 - Convolution
    model.add(Conv2D(64,(3,3), padding='same', input_shape=(224, 224, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 2nd Convolution layer
    model.add(Conv2D(128,(5,5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 3rd Convolution layer
    model.add(Conv2D(512,(3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 4th Convolution layer
    model.add(Conv2D(512,(3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Flattening
    model.add(Flatten())

    # Fully connected layer 1st layer
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # Fully connected layer 2nd layer
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(EMOTION_COUNT, activation='softmax'))

    opt = Adam(lr=0.0001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# TODO add support for continuing training from a prior state.
def train_model(model, training_data, validation_data):
    """
    Trains a model with the given training and validation data.

    @param model CNN model to train.
    @param training_data Map<Int, Int> that maps image ids to emotions.
    @param validation_data Map<Int, Int> that maps image ids to emotions.
    """

    # Initialize generators that only load small portion of dataset to train on at a time.
    train_generator = ImageSequence(training_data, "/data/datasets/affectNet/train_set/images", BATCH_SIZE)
    validation_generator = ImageSequence(validation_data, "/data/datasets/affectNet/val_set/images", BATCH_SIZE)

    checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    # TODO I think we can store history? Could be helpful for displaying graphs and stuff later.
    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_generator),
        epochs=EPOCHS,
        validation_data = validation_generator,
        validation_steps = len(validation_generator),
        callbacks = [checkpoint]
    )

    # Write model structure to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)


######
# Main
######

# Cache dataset info for more efficient training if this hasn't already been done.
if not has_cached_emotions():
    cache_emotions()

train_model(
    create_model(),
    read_cached_emotions("train"),
    read_cached_emotions("validation")
)
