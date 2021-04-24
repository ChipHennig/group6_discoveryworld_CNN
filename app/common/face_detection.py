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

proto_text = "deploy.prototxt.txt"
face_extraction_weights = "opencv_face_detector.caffemodel"
confidence_threshold = 0.8

net = cv2.dnn.readNetFromCaffe(proto_text, face_extraction_weights)


def get_face_bounding_boxes(frame, threshold):
    """

    :param frame:
    :param threshold:
    :return:
    """

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, size=(300, 300), mean=(103.93, 116.77, 123.68))

    net.setInput(blob)
    detections = net.forward()

    valid_bounds = detections[0, 0, detections[0, 0][:, 2] > threshold, 2:7]
    return (valid_bounds * np.array([1, w, h, w, h])).astype("int")


def extract_faces(frame, bounds):

    faces = []

    for (_, startX, startY, endX, endY) in bounds:

        face = frame[startX:endX, startY:endY]
        faces.append(resize(face, (224, 224)))

    return faces
