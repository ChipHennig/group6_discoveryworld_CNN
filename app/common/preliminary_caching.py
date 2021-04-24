# TODO Eventually we will also want to run the race detector on all images and then cache those results in a csv file.
# This make take several hours but will allow us to stratify by race efficiently.
import csv
import os
from pathlib import Path
import numpy as np
import pandas as pd


base_path = Path(__file__).parent


def load_affectnet_examples(partition_folder):
    
    """
    Loads a map of image IDs to emotions from a certain parition of AffectNet.

    The affectnet images and their labels are stored in two separate directories. Each image name is of the form <id>.jpeg.
    Each of these images have a corresponding .npy which stores the emotion of an image. A map is created by iterating over
    all images, and extracting the ID from the file name, and then reading the appropriate emotion file.

    @partition_folder Folder to read affect net examples from. Ex: Train or Validation
    @returns Map<Int, Int> that maps an image ID to an integer representing an emotion.
    """

    emotion_map = dict()
    
    # Iterate through all images and get their identifiers, and then load that images emotion.
    for image in os.listdir(f"{partition_folder}/images"):

        # Give one update on how many images have been loaded since it can take a while.
        if len(emotion_map) % 1_000 == 0:
            print(f"\rLoaded {len(emotion_map)} examples from {partition_folder}", end="")
        
        image_id = image.split(".")[0]
        emotion = np.load(f"{partition_folder}/annotations/{image_id}_exp.npy")
        
        emotion_map[image_id] = emotion
        
    print(f"\rLoaded {len(emotion_map)} examples from {partition_folder}")
    
    return emotion_map


def write_affectnet_info(emotion_map, partition):

    """
    Writes the mapping from image IDs to emotions into a specified CSV.

    @param emotion_map Map<Int, Int> that maps image ids to emotions.
    @param partition Partition to save. This is used as the CSV name. Ex: Train, Validation
    """

    with open(f"{partition}.csv", 'w+', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar="\"", quoting=csv.QUOTE_MINIMAL)
    
        for entry in emotion_map:
            writer.writerow([entry, emotion_map[entry]])


def cache_emotions():

    """
    Caches the emotions corresponding to an image ID in a single csv file.

    Since all image emotions are stored as individual files, it can take a long time to read each emotion individually.
    The emotions for each image are read in and then cached in the train.csv and validation.csv files under data/AffectNet
    """

    train_emotion_map = load_affectnet_examples("/data/datasets/affectNet/train_set")
    validation_emotion_map = load_affectnet_examples("/data/datasets/affectNet/val_set")

    write_affectnet_info(train_emotion_map, "train")
    write_affectnet_info(validation_emotion_map, "validation")


def has_cached_emotions():

    """
    Checks if the appropriate csv files containing cached emotion information about the AffectNet dataset exist.

    @returns boolean that describes if the appropriate files exist. 
    """
    
    return Path(f"{base_path}/training/data/AffectNet/train.csv").exists() and Path(f"{base_path}/training/data/AffectNet/validation.csv").exists()


def read_cached_emotions(partition):
    """
    Reads in cached emotion data as a Map<Int, Int> that maps an image ID to an emotion.

    @partition Partition of the dataset. Ex: Train, Validation
    """
    return pd.read_csv(f"{base_path}/training/data/AffectNet/{partition}.csv", header=None, index_col=0, squeeze=True).to_dict()