# TODO Have the sequence take into account race data and stratify based on it.
# TODO Have the sequence balance "epochs" by emotion class.
# TODO Right now we use this for both validation and training. 
# Maybe in future just use this for validation and have a different one for training?

from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img
from keras.utils import to_categorical
from PIL import Image
import numpy as np
import math
import random


class ImageSequence(Sequence):

    """
    Implementation of a sequence that reads images from the AffectNet dataset and passes them
    to a model for training. Basically allows for only a small portion of a dataset to be loaded
    in memory at a time, which is pretty nessasary with a 120 GB dataset...

    See: tensorflow.org/api_docs/python/tf/keras/utils/Sequence for more info.

    @param emotion_map Map<Int, Int> that maps image ids to an emotion.
    @param partition_folder Folder that contains the images to read.
    @param batch_size Batch size to use. 
    """
    
    def __init__(self, emotion_map, partition_folder, batch_size):
        
        self.emotion_map = emotion_map
        self.ids = list(emotion_map.keys())
        self.partition_folder = partition_folder
        self.batch_size = batch_size


    def __len__(self):
        """
        Returns number of batches in an epoch (rounded down).
        """

        return math.floor(len(self.ids) / self.batch_size)
    

    def __getitem__(self, index):
        """
        Gets a batch given a batch index.
        """
        
        batch_ids = self.ids[index * self.batch_size: (index + 1) * self.batch_size]
        return self.__data_generation__(batch_ids)


    def __data_generation__(self, ids):
        
        X = np.array(list(map(self.__load_image__, ids)))
        X = np.expand_dims(X, axis=3) # Must pad extra dimension or tensorflow will complain when hooking up to model
        y = np.array(to_categorical(list(map(lambda id: self.emotion_map[id], ids)), 11))
        
        return X, y
    

    # TODO Add support for color images.
    def __load_image__(self, image_id):
        """
        Loads a single image as a 2d numpy array given an id.

        @image_id Id of image to load.
        """
        
        image = load_img(
            f"{self.partition_folder}/{image_id}.jpg",
            grayscale=False,
            color_mode='grayscale', # Might need to change back to 'rgb'
            target_size=None,
            interpolation='nearest'
        )
        
        # TODO Do some augmentation
        # TODO Run image through face extractor and return distored image.
        return np.asarray(image)

    
    def on_epoch_end(self):
        """
        Shuffles the data on the end of every epoch. 
        """
        # TODO Do some fancy stuff for stratification here.
        random.shuffle(self.ids)