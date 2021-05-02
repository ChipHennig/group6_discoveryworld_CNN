from imutils.video import VideoStream
import numpy as np
import argparse
from pathlib import Path
import imutils
import time
from tensorflow.keras.models import load_model
from cv2.dnn import blobFromImage
from cv2 import cvtColor, resize, COLOR_RGB2GRAY

# pip install opencv-python for this to work
import cv2

base_folder = 

proto_text = "deploy.prototxt.txt"
face_extraction_weights = "opencv_face_detector.caffemodel"
net = cv2.dnn.readNetFromCaffe(proto_text, face_extraction_weights)


def get_face_bounding_boxes(image, threshold=0.7, max_faces=400):
    """
    @param image 
    @param threshold
    @param max_faces
    @param min_padding_size The minimum (Width, Height) to pad an image to.
    """

    (height, width) = image.shape[:2]

    padded_image = np.zeros((height * 2, width * 2, image.shape[2]))
    padded_image[:height, :width] = image

    blob = blobFromImage(padded_image, size=(300, 300), mean=(103.93, 116.77, 123.68))

    net.setInput(blob)
    detections = net.forward()

    face_bounds_data = []
    possible_faces = detections[0, 0, detections[0, 0][0:max_faces, 2] > threshold, 2:7]

    # Filter the possible faces to make sure they fit in the non padded image.
    # If part (but not all) of the face is in the padded reason, then clip the bounds so it would lie
    # completely in the non padded region. 
    
    for possible_face in possible_faces:

        (confidence, min_x, min_y, max_x, max_y) = possible_face

        if min_x > 0.5 * width or min_y > 0.5 * height:
             continue
        
        face_bounds_data.append((
            confidence,
            int(min_x * width), 
            int(min_y * height),
            min(int(max_x * width), width),
            min(int(max_y * height), height)
        )) 

    return face_bounds_data


def extract_faces(image, face_bounds_data):

    extracted_faces = []

    for (_, min_x, min_y, max_x, max_y) in face_bounds_data:
        extracted_faces.append(image[min_y:max_y, min_x:max_x])
    
    return extracted_faces
