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

    @param image_data_map Map<Int, Int> that maps image ids to an emotion.
    """
    
    def __init__(self, image_data_map):
        
        self.image_data_map = image_data_map
        self.image_paths = list(image_data_map.keys())


    def __len__(self):
        """
        Returns number of images in an epoch.
        """
        return len(self.image_paths)
    

    def __getitem__(self, index):
        """
        Gets a tuple of an image and an emotion label.
        """

        image_path = self.image_paths[index]
        X = load_image(image_path, self.image_data_map)
        y = np.array(to_categorical(self.image_data_map[image_path]["emotion"], 11))

        # Must pad some extra dimension or tensorflow will complain when hooking up to model
        X = np.expand_dims(X, axis=0) 
        y = np.expand_dims(y, axis=0)
        
        return preprocess_input(X), y
    

    def on_epoch_end(self):
        """
        Shuffles the data on the end of every epoch. 
        """
        random.shuffle(self.image_paths)


class TrainingSequence(Sequence):

    """
    Implementation of a sequence that reads images from the AffectNet dataset and passes them
    to a model for training. Basically allows for only a small portion of a dataset to be loaded
    in memory at a time. Unlike BasicTrainingSequence, this sequence stratifies images by emotion
    so that each epoch will have the same number of images in each class. This sequence also augments
    the images it recieves in an attempt the model it is used with more generalizable.

    See: tensorflow.org/api_docs/python/tf/keras/utils/Sequence for more info.

    @param image_data_map Map<Int, Int> that maps image ids to an emotion.
    @param batch_size Batch size to use.
    @param strat_classes String or list of classes to stratify by ([] for no strat) 
    """

    def __init__(self, image_data_map, batch_size, strat_classes):
        self.image_data_map = image_data_map
        self.batch_size = batch_size
        self.strat_classes = [strat_classes] if isinstance(strat_classes, str) else strat_classes
        self.class_lists = self.__separate_classes__(image_data_map)
        self.min_class_size = self.__calculate_min_class_size__(self.class_lists)
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


    def __data_generation__(self, image_paths):
        """
        Generates a tuple containing the image arrays and emotion labels for the given IDs

        @param ids Array of image IDs to load data for.
        """
        
        X = np.array(list(map(self.__load_augmented_image__, image_paths)))
        y = np.array(to_categorical(list(map(lambda path: self.image_data_map[path]["emotion"], image_paths)), 11))

        # Since we are using Resnet50, we preprocess the images in the way it expects.
        return preprocess_input(X), y
    

    def __load_augmented_image__(self, image_path):
        """
        Loads a single image as a 2d numpy array given an id.

        @param image_id Id of image to load.
        """
        
        image = load_image(image_path, self.image_data_map)
        return random_flip_left_right(image)


    def __calculate_min_class_size__(self, class_lists):
        """
        Returns the number of images the class with the least amount of examples has.

        @param emotions_lists Map<Int, List<Int>> that maps emotions to a list of images that show that emotion.
        """

        return min(list(map(lambda lst: len(lst), class_lists.values())))


    def __separate_classes__(self, image_data_map):
        """
        Takes of Map<Int, Int> where the first value represents the image ID and the second value
        represents an emotion label, and transforms it into a Map<Int, List<Int>> where the first value
        is an emotion label, and the list contain the IDs of all the images that have that emotion.

        @param image_data_map Map<Int, Int> that maps image IDs to emotions.
        """

        class_lists = dict()
        for path in self.image_data_map:
            class_name = map(lambda cat: str(image_data_map[path][cat]), self.strat_classes)
            class_name = "-".join(class_name)
            if class_name not in class_lists:
                class_lists[class_name] = []
            class_lists[class_name].append(path)

        return class_lists


    def __calculate_current_ids__(self):
        """
        Randomly selects an equal number of images from each emotion class. The number of selected images
        is determined by __calculate_min_class_size__. This ensures that each emotion will be equally
        represented for each epoch, and the model is not biased to a particular one. This function
        should be called before each epoch runs.
        """

        current_ids = []

        for category in self.class_lists:

            random.shuffle(self.class_lists[category])
            current_ids.extend(self.class_lists[category][:self.min_class_size])

        random.shuffle(current_ids)

        return current_ids
    

    def on_epoch_end(self):
        """
        Randomizes the images to train on. 
        """
        self.current_ids = self.__calculate_current_ids__()


# TODO Run image through face extractor and return distored image.
def load_image(image_path, image_data_map):
    """
    Loads a single image as a 2d numpy array given an id.

    @image_path Path to the image to load
    @image_data_map Map containing information about a training image.
    """

    image = mpimg.imread(image_path)
    
    # If no face was found we will just use the entire image.
    # Use a confidence of zero as a placeholder.
    return extract_faces(image, [(0, *image_data_map[image_path]["bounds"])])[0]