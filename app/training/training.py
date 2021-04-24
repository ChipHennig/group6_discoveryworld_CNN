from .PreliminaryCaching import has_cached_emotions, cache_emotions
from .ImageSequences import BasicImageSequence, TrainingSequence

import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys
from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications.resnet import ResNet50
from pathlib import Path


# Number of emotions we are predicting.
EMOTION_COUNT = 11


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

    base_model = ResNet50(include_top = True, weights = "imagenet")

    new_output = Dense(EMOTION_COUNT)(base_model.layers[-1].output)
    new_model = Model(inputs=base_model.input, outputs=new_output)

    # Leave last 10 layers trainable.
    for layer in new_model.layers[:-10]:
        layer.trainable = False

    opt = Adam(lr=0.00001)
    new_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return new_model


# TODO add support for continuing training from a prior state.
def train_model(model, training_data, validation_data, epochs, batch_size, save_directory):
    """
    Trains a model with the given training and validation data.

    @param model CNN model to train.
    @param training_data Map<Int, Int> that maps image ids to emotions.
    @param validation_data Map<Int, Int> that maps image ids to emotions.
    @param epochs Number of epochs to train for
    @param batch_size Batch size to use
    @param save_directory Directory to save model checkpoints to.
    """

    # Initialize generators that only load small portion of dataset to train on at a time.
    train_generator = TrainingSequence(training_data, "/data/datasets/affectNet/train_set/images", batch_size)
    validation_generator = BasicImageSequence(validation_data, "/data/datasets/affectNet/val_set/images")

    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = save_directory + "/checkpoints/cp-{epoch:04d}.ckpt"

    # Saves the model's weights every 100 batches in case something goes wrong.
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True, save_freq=100)
    model.save_weights(checkpoint_path.format(epoch=0))

    # Decrease learning rate by a factor of 5 if no improvement is seen after 3 epochs.
    learning_rate_callback = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=3, verbose=0,
        mode='auto', min_delta=0.0001, cooldown=0, min_lr=0
    )

    # TODO I think we can store history? Could be helpful for displaying graphs and stuff later.
    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data = validation_generator,
        validation_steps = len(validation_generator),
        callbacks = [checkpoint_callback, learning_rate_callback]
    )

    return model, history


def get_commandline_args():
    """
    Parses command line information from the user that will be used in training.

    @return Object that contains the following values:
        epochs - Number of epochs to train for. Is always greater than 1.
        save_directory - Directory to save model information to. This includes the model checkpoints.
        batch_size - Batch size to use. Default is 32. Is always greater than 1.
        model_file - Specifies location of model to load and start training again. If `None` a new model should be created.
    """

    # See https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser(description="Trains CNN for emotion recognition on the AffectNet dataset.")
    parser.add_argument("epochs", type=int, help="Number of epochs to train the model for.")
    parser.add_argument("save_directory", type=str, help="Directory to save the trained model and its checkpoints to.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size to train model with.")
    parser.add_argument("--model_file", type=str, default=None, help="Optional. Continue training model at given location. The model is then saved to 'save_location'.")
    args = parser.parse_args()

    if args.epochs < 1:
        sys.exit("Epochs must be greater than 1.")

    if args.batch_size < 1:
        sys.exit("Batch size must be greater than 1.")

    return args 


######
# Main
######

args = get_commandline_args()

# Cache dataset info for more efficient training if this hasn't already been done.
if not has_cached_emotions():
    cache_emotions()

model = create_model() if args.model_file is None else load_model(args.model_file)

trained_model, history = train_model(
    model,
    read_cached_emotions("train"),
    read_cached_emotions("validation"),
    args.epochs,
    args.batch_size,
    args.save_directory
)

trained_model.save(args.save_directory + "/TrainedModel")
