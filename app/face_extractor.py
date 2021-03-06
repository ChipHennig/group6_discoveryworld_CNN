from imutils.video import VideoStream
import numpy as np
import argparse
from pathlib import Path
import imutils
import time
from tensorflow.keras.models import load_model
from cv2.dnn import blobFromImage, readNetFromCaffe
from cv2 import cvtColor, resize, COLOR_RGB2GRAY

# pip install opencv-python for this to work
import cv2

base_folder = str(Path(__file__).parent.absolute())

proto_text = base_folder + "/deploy.prototxt.txt"
face_extraction_weights = base_folder + "/opencv_face_detector.caffemodel"
net = readNetFromCaffe(proto_text, face_extraction_weights)


def get_face_bounding_boxes(image, threshold=0.7, max_faces=400, min_face_size=(25, 25)):
    """
    Gets a list of tuples that contains information on the bounding boxes of faces found in an image.
    The tuples are of the form (confidence, min x, min y, max x, max y). If no faces are found, an
    empty list is returned. 

    @param image RGB image to extract faces from.
    @param threshold The minimum confidence required to extract a face.
    @param max_faces The maximum number of faces one wants to extract. Can not be more than 400.
    @param min_face_size Minimum size of extracted face. If face bounds is less than this it is discarded.

    @return A list of tuples array where each tuple corresponds to face instances. The first tuple entry is the confidence, and the remaining
            four entries are the mininum x coordinate, minimum y coordinate, maximum x coordinate, and maximum y coordinate
            of the faces bounding box.
    """

    (height, width) = image.shape[:2]

    padded_image = np.zeros((height * 2, width * 2, image.shape[2]), dtype=np.uint8)
    padded_image[:height, :width] = image

    blob = blobFromImage(resize(padded_image, (300, 300)), 1, (300, 300), (103.93, 116.77, 123.68), swapRB=True)

    net.setInput(blob)
    detections = net.forward()

    face_bounds_data = []
    possible_faces = detections[0, 0, detections[0, 0, :, 2] > threshold, 2:7]
    possible_faces = possible_faces[0:max_faces]

    # Filter the possible faces to make sure they fit in the non padded image.
    # If part (but not all) of the face is in the padded reason, then clip the bounds so it would lie
    # completely in the non padded region. 
    
    for possible_face in possible_faces:

        (confidence, min_x, min_y, max_x, max_y) = possible_face

        min_x = max(int(min_x * width * 2), 0)
        min_y = max(int(min_y * height * 2), 0) 
        max_x = min(int(max_x * width * 2), width * 2)
        max_y = min(int(max_y * height) * 2, height * 2)

        # Check if minimum face coordinates are in padded section, or if face is to small.
        if min_x > width or min_y > height or max_x - min_x < min_face_size[0] or max_y - min_y < min_face_size[1]:
            continue
        
        face_bounds_data.append((confidence, min_x, min_y, max_x, max_y)) 

    return face_bounds_data


def extract_faces(image, face_bounds_data, reshape_size=(224, 224)):
    """
    Extracts images that contain a face given an image to extract faces from and bounds data used to extract faces.

    @param image Image to extract faces from.
    @param face_bounds_data List of tuples that are of the format (confidence, min x, min y, max x, max y)
    @param reshape_size Shape to resize the extracted faces to.

    @return List of extracted faces of size 'reshape_size'
    """

    extracted_faces = []

    for (_, min_x, min_y, max_x, max_y) in face_bounds_data:
        extracted_faces.append(resize(image[min_y:max_y, min_x:max_x], reshape_size))
    
    return extracted_faces
