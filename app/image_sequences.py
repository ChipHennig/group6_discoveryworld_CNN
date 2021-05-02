from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.image import random_flip_left_right
from keras.utils import to_categorical
from PIL import Image
import numpy as np
import math
import random
from face_extractor import get_face_bounding_boxes, extract_faces
import matplotlib.image as mpimg


class BasicImageSequence(Sequence):

    """
    Implementation of a sequence that reads images from the AffectNet dataset and passes them
    to a model for validation or testing. This allows for only a small portion of a dataset to be loaded
    in memory at a time. Unlike TrainingSequence, images with this sequence are read in one by
    one and no augmentation is done to any of the images. TrainingSequence stratifies examples
    by emotion, which is something this sequence does not do.

    See: tensorflow.org/api_docs/python/tf/keras/utils/Sequence for more info.

    @param emotion_map Map<Int, Int> that maps image ids to an emotion.
    @param partition_folder Folder that contains the images to read.
    """
    
    def __init__(self, emotion_map, partition_folder):
        
        self.emotion_map = emotion_map
        self.ids = list(emotion_map.keys())
        self.partition_folder = partition_folder


    def __len__(self):
        """
        Returns number of images in an epoch.
        """
        return len(self.ids)
    

    def __getitem__(self, index):
        """
        Gets a tuple of an image and an emotion label.
        """

        X = load_image(self.ids[index], self.partition_folder)
        y = np.array(to_categorical(self.emotion_map[self.ids[index]], 11))

        # Must pad some extra dimension or tensorflow will complain when hooking up to model
        X = np.expand_dims(X, axis=0) 
        y = np.expand_dims(y, axis=0)
        
        return X, y
    

    def on_epoch_end(self):
        """
        Shuffles the data on the end of every epoch. 
        """
        random.shuffle(self.ids)


class TrainingSequence(Sequence):

    """
    Implementation of a sequence that reads images from the AffectNet dataset and passes them
    to a model for training. Basically allows for only a small portion of a dataset to be loaded
    in memory at a time. Unlike BasicTrainingSequence, this sequence stratifies images by emotion
    so that each epoch will have the same number of images in each class. This sequence also augments
    the images it recieves in an attempt the model it is used with more generalizable.

    See: tensorflow.org/api_docs/python/tf/keras/utils/Sequence for more info.

    @param emotion_map Map<Int, Int> that maps image ids to an emotion.
    @param partition_folder Folder that contains the images to read.
    @param batch_size Batch size to use. 
    """

    def __init__(self, emotion_map, partition_folder, batch_size):
        
        self.emotion_map = emotion_map
        self.partition_folder = partition_folder
        self.batch_size = batch_size
        self.emotion_lists = self.__separate_classes__(emotion_map)
        self.min_class_size = self.__calculate_min_class_size__(self.emotion_lists)
        self.current_ids = self.__calculate_current_ids__()


    def __len__(self):
        """
        Returns number of batches in an epoch (rounded down).
        """
        return math.floor(len(self.current_ids) / self.batch_size)
    

    def __getitem__(self, index):
        """
        Gets a batch given a batch index.
        """
        
        batch_ids = self.current_ids[index * self.batch_size: (index + 1) * self.batch_size]
        return self.__data_generation__(batch_ids)


    def __data_generation__(self, ids):
        """
        Generates a tuple containing the image arrays and emotion labels for the given IDs

        @param ids Array of image IDs to load data for.
        """
        
        X = np.array(list(map(self.__load_augmented_image__, ids)))
        y = np.array(to_categorical(list(map(lambda id: self.emotion_map[id], ids)), 11))

        # Since we are using Resnet50, we preprocess the images in the way it expects.
        return preprocess_input(X), y
    

    def __load_augmented_image__(self, image_id):
        """
        Loads a single image as a 2d numpy array given an id.

        @param image_id Id of image to load.
        """
        
        image = load_image(image_id, self.partition_folder)
        return random_flip_left_right(image)


    def __calculate_min_class_size__(self, emotion_lists):
        """
        Returns the number of images the class with the least amount of examples has.

        @param emotions_lists Map<Int, List<Int>> that maps emotions to a list of images that show that emotion.
        """

        return min(list(map(lambda lst: len(lst), emotion_lists.values())))


    def __separate_classes__(self, emotion_map):
        """
        Takes of Map<Int, Int> where the first value represents the image ID and the second value
        represents an emotion label, and transforms it into a Map<Int, List<Int>> where the first value
        is an emotion label, and the list contain the IDs of all the images that have that emotion.

        @param emotion_map Map<Int, Int> that maps image IDs to emotions.
        """

        emotion_lists = dict()
        for id in self.emotion_map:

            if emotion_map[id] not in emotion_lists:
                emotion_lists[emotion_map[id]] = []

            emotion_lists[emotion_map[id]].append(id)

        return emotion_lists


    def __calculate_current_ids__(self):
        """
        Randomly selects an equal number of images from each emotion class. The number of selected images
        is determined by __calculate_min_class_size__. This ensures that each emotion will be equally
        represented for each epoch, and the model is not biased to a particular one. This function
        should be called before each epoch runs.
        """

        current_ids = []

        for emotion in self.emotion_lists:

            random.shuffle(self.emotion_lists[emotion])
            current_ids.extend(self.emotion_lists[emotion][:self.min_class_size])

        random.shuffle(current_ids)

        return current_ids
    

    def on_epoch_end(self):
        """
        Randomizes the images to train on. 
        """
        self.current_ids = self.__calculate_current_ids__()


# TODO Run image through face extractor and return distored image.
def load_image(image_id, partition_folder):
    """
    Loads a single image as a 2d numpy array given an id.

    @image_id Id of image to load.
    @partition_folder Folder containing the image.
    """

    image = mpimg.imread(f"{partition_folder}/{image_id}.jpg")
    
    # Get most likely bounds for face.
    bounds = get_face_bounding_boxes(image, 0.0, 1)

    # If no face was found we will just use the entire image.
    if len(bounds) > 0:
        image = extract_faces(image, bounds)[0]
    else:
        print(f"Could not locate face in image with id: {image_id}")

    return image 
