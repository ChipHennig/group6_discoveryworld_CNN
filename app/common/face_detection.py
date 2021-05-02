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

base_path = Path(__file__).parent.parent.parent

proto_text = f"{base_path}/app/common/deploy.prototxt.txt"
face_extraction_weights = f"{base_path}/app/common/opencv_face_detector.caffemodel"
# confidence_threshold = 0.8

net = cv2.dnn.readNetFromCaffe(proto_text, face_extraction_weights)


def get_face_bounding_boxes(frame, threshold):
    """

    :param frame:
    :param threshold:
    :return:
    """

    (h, w) = frame.shape[:2]
    print(f"{h} {w}")
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    print(blob.shape)
    print(blob)

    net.setInput(blob)
    detections = net.forward()
    print(detections)

    valid_bounds = detections[0, 0, detections[0, 0][:, 2] > threshold, 2:7]
    print(valid_bounds[0])
    return (valid_bounds * np.array([1, w, h, w, h])).astype("int")


def get_blob(image):
    return cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# def detect_face_from_image(path_to_image, confidence_threshold=0.0):
#     image = cv2.imread(path_to_image)
#     (h, w) = image.shape[:2]
#     blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
#                                  (300, 300), (104.0, 177.0, 123.0))
#     net.setInput(blob)
#     detections = net.forward()

#     print(detections)

#     num_detections = detections.shape[2]
#     confidences = detections[0, 0, 0:num_detections, 2]
#     detections = detections[0, 0, confidences > confidence_threshold, 0:7]

#     # print(detections)

#     if detections.size == 0:
#         return
#     # get the size of each box as the sum of two sides of the rectangle
#     # num_detections = detections.shape[0]
#     # sizes = ((detections[0:num_detections, 5] - detections[0:num_detections, 3])
#     #          + (detections[0:num_detections, 6] - detections[0:num_detections, 4]))
#     box = detections[np.argmax(sizes), 3:7] * np.array([w, h, w, h])
#     (startX, startY, endX, endY) = box.astype("int")

#     print(f"Bounds Found: {startX} {startY} {endX} {endY}")
#     maxY = image.shape[0]
#     maxX = image.shape[1]
#     return image[max(startY - 50, 0):min(endY + 50, maxY), max(startX - 50, 0):min(endX + 50, maxX)]

def detect_face_from_image(path_to_image):

    image = cv2.imread(path_to_image)
    (h, w) = image.shape[:2]

    print(f"H: {h} W: {w}")

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
                                 
    net.setInput(blob)
    detections = net.forward()

    print(f"Raw: {detections[0, 0, 0, 3:7]}")

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")

    print(f"Bounds Found: {startX} {startY} {endX} {endY}")
    return image[startX:endX, startY:endY]


def extract_faces(frame, bounds):

    faces = []

    for (_, startX, startY, endX, endY) in bounds:

        face = frame[startX:endX, startY:endY]
        print(face.shape)
        faces.append(resize(face, (224, 224)))

    return faces
