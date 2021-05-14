import csv
import os
from pathlib import Path
import numpy as np
import pandas as pd
import face_extractor
import fairface
import matplotlib.image as mpimg
import re

emotion_to_id = {
    "neutral": 0,
    "happy": 1,
    "happiness": 1,
    "sadness": 2,
    "sad": 2,
    "surprise": 3,
    "fear": 4, 
    "disgust": 5,
    "angry": 6, 
    "contempt": 7,  
    "none": 8, 
    "uncertain": 9, 
    "noface": 10
}

base_path = Path(__file__).parent.parent

def load_affectnet_examples(partition_folder, include_fairface_7=False):
    
    """
    Loads a map of image IDs to (emotions, min_x, min_y, max_x, max_y) from a certain parition of AffectNet.

    The affectnet images and their labels are stored in two separate directories. Each image name is of the form <id>.jpeg.
    Each of these images have a corresponding .npy which stores the emotion of an image. A map is created by iterating over
    all images, and extracting the ID from the file name, and then reading the appropriate emotion file.

    @partition_folder Folder to read affect net examples from. Ex: Train or Validation
    @returns Map<Int, Int> (last 3 depend on include_fairface7)
        that maps an image ID to a tuple of the emotion followed by the boundaries
    """

    emotion_map = dict()
    
    # Iterate through all images and get their identifiers, and then load that images emotion.
    for image in os.listdir(f"{partition_folder}/images"):
        # Give one update on how many images have been loaded since it can take a while.
        if len(emotion_map) % 1_000 == 0:
            print(f"\rLoaded {len(emotion_map)} examples from {partition_folder}", end="")

        image_id = image.split(".")[0]
        image_path = f"{partition_folder}/images/{image_id}.jpg"
        emotion = np.load(f"{partition_folder}/annotations/{image_id}_exp.npy")
        
        emotion_map[image_path] = int(emotion)
        
    print(f"\rLoaded {len(emotion_map)} examples from {partition_folder}")
    return emotion_map


def write_dataset_info(emotion_map, partition):

    """
    Writes the mapping from image IDs to emotions into a specified CSV.

    @param emotion_map Map<Int, (Int,Int,Int,Int,Int)> that maps image ids to emotions, min_x, min_y, max_x, max_y.
    @param partition Partition to save. This is used as the CSV name. Ex: Train, Validation
    """

    with open(f"{base_path}/data/{partition}.csv", 'w+', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar="\"", quoting=csv.QUOTE_MINIMAL)
        for entry in emotion_map:
            data = emotion_map[entry]
            row = [entry] + list(data)
            writer.writerow(row)


def load_cafe_dataset(starting_child_id, ending_child_id):
    """
    Loads a map of paths to emotion IDs from the CAFE dataset from the images between two given image IDs.

    @param starting_child_id The ID of the image to start loading from.
    @param ending_child_id The ID of the iamge to end loading from.
    """

    dataset_path = "/data/datasets/ChildAffectiveFacialExpression/databrary30-LoBue-Thrasher-The_Child_Affective_Facial/sessions"
    emotion_map = dict()

    for child_id in range(starting_child_id, ending_child_id):
        for image in os.listdir(f"{dataset_path}/{child_id}"):

            match = re.search(".*(angry|disgust|fearful|happy|neutral|sad|surprise).*", image)
            emotion = match.group(1)

            if emotion == "fearful": 
                continue

            emotion_id = emotion_to_id[emotion]
            emotion_map[f"{dataset_path}/{child_id}/{image}"] = emotion_id
        
        # Give one update on how many images have been loaded since it can take a while.
        print(f"\rLoaded {child_id - 6280 + 1} examples from {dataset_path}", end="")

    print(f"\rLoaded {len(emotion_map)} examples from {dataset_path}")
    
    return emotion_map


def calculate_extra_data(emotion_mappings, fair_face_function):
    """
    Calculates extra information about an image given its path, such as the face bounds, race, age, and gender.

    @param emotion_mappings A map of image paths to emotion IDs
    @param fair_face_function A fair face function that returns a list of additional information. 
                              See 'predict_race4' and 'predict_race7' in 'fairface.py'

    @return A map of image paths to maps of various image properties. The map for each image has the following keys:
            emotion => Emotion ID
            bounds => Tuple of ints corresponding to face bounds
            race => Race of individual in image. Possibilities depend on 'fair_face_function'
            age => Integer representing age range ONLY if using 'predict_race7'
            gender => Int representing gender ONLY if using 'predict_race7'
    """

    image_data = {}

    for image_path in emotion_mappings:

        # Give one update on how many images have had extra data calculated since it can take a while.
        print(f"\rExtracted additional data from {len(image_data)} images", end="")

        actual_image = mpimg.imread(image_path)
        bounding_box = face_extractor.get_face_bounding_boxes(actual_image, 0.0,1)

        if len(bounding_box) == 0: continue
           
        face = face_extractor.extract_faces(actual_image, bounding_box)[0]

        bounding_box = bounding_box[0]
        image_data[image_path] = [emotion_mappings[image_path], bounding_box[1], bounding_box[2], bounding_box[3], bounding_box[4]]
        image_data[image_path].extend(fair_face_function(face))

    return image_data

def cache_emotions():

    """
    Caches the emotions corresponding to an image ID in a single csv file.

    Since all image emotions are stored as individual files, it can take a long time to read each emotion individually.
    The emotions for each image are read in and then cached in the train.csv and validation.csv files under data/AffectNet
    """

    train_emotion_map = load_affectnet_examples("/data/datasets/affectNet/train_set")
    validation_emotion_map = load_affectnet_examples("/data/datasets/affectNet/val_set")

    train_child_emotion_map = load_cafe_dataset(6280, 6400)
    validation_child_emotion_map = load_cafe_dataset(6400, 6437)

    validation_combined_emotion_map = dict()
    validation_combined_emotion_map.update(validation_emotion_map)
    validation_combined_emotion_map.update(validation_child_emotion_map)

    train_combined_emotion_map = dict()
    train_combined_emotion_map.update(train_emotion_map)
    train_combined_emotion_map.update(train_child_emotion_map)

    validation_combined_data_map = calculate_extra_data(validation_combined_emotion_map, fairface.predict_race7)
    train_combined_data_map = calculate_extra_data(train_combined_emotion_map, fairface.predict_race4)

    write_dataset_info(train_combined_data_map, "train")
    write_dataset_info(validation_combined_data_map, "validation")


def has_cached_emotions():

    """
    Checks if the appropriate csv files containing cached emotion information about the AffectNet dataset exist.

    @returns boolean that describes if the appropriate files exist. 
    """
    
    return Path(f"{base_path}/data/train.csv").exists() and Path(f"{base_path}/data/validation.csv").exists()


def read_cached_data(partition):

    """
    Reads in cached emotion data as a Map<Int, (Int,Int,Int,Int,Int)> that maps an image ID to an emotion and boundary coordinates

    @param partition Partition of the dataset. Ex: train, validation
    """
    
    image_data_map = {}
    df = pd.read_csv(f"{base_path}/data/{partition}.csv", header=None, index_col=0, squeeze=True)
    
    for index, row in df.iterrows():

        image_data_map[index] = {
            "emotion": row.iloc[0],
            "bounds": tuple(row.iloc[1:5]),
            "race": row.iloc[5]
        }

        if partition == "validation":
            image_data_map[index].update({
                "age": row.iloc[6],
                "gender": row.iloc[7]
            })
        
    return image_data_map


def filter_emotion_data(emotion_data_map, excluded_classes):
    """
    Filters out the examples in a dictionary that contain unwanted emotions.

    @param emotion_data_map Maps image paths to map of data corresponding to the image.
    @param excluded_classes List of integers that correspond to the IDs of unwanted emotions.

    @returns New dictionary containing only images with the wanted emotions.
    """

    filtered_data = dict()

    for (key, value) in emotion_data_map.items():
    
        if value["emotion"] not in excluded_classes:
            filtered_data[key] = value

    return filtered_data
