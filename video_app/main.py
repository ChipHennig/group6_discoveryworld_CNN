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

proto_text = "deploy.prototxt.txt"
face_extraction_weights = "opencv_face_detector.caffemodel"
confidence_threshold = 0.8

model = load_model(str(Path(__file__).parent.absolute()) + "/Model", compile=True)

emotion_map = {
    0: "Neutral",
    1: "Happiness",
    2: "Sadness",
    3: "Surprise",
    4: "Sad",
    5: "Disgust",
    6: "Anger",
    7: "Contempt",
    8: "None",
    9: "Uncertain",
    10: "No-Face",
}


def label_bounds(frame, bounds_data, labels):

    # draw the bounding box of the face along with the associated probability

    for i in range(labels.shape[0]):

        (confidence, startX, startY, endX, endY) = bounds_data[i]

        confidence = bounds_data[i, 0] * 100.0
        text = f"{labels[i]} - {confidence:.2f}"

        cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 255), 2)
        cv2.putText(frame, text, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


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
    print("===")
    print(valid_bounds)
    return (valid_bounds * np.array([1, w, h, w, h])).astype("int")


def predict_emotions(frame, bounds):
    """

    :param frame:
    :param bounds:
    :param size:
    :return:
    """

    faces = []

    for (_, startX, startY, endX, endY) in bounds:

        face = frame[startX:endX, startY:endY]
        faces.append(blobFromImage(cvtColor(resize(face, (224, 224)), COLOR_RGB2GRAY)))

    input_data = np.squeeze(np.array(faces))
    input_data = np.expand_dims(input_data, 0)
    input_data = np.expand_dims(input_data, -1)
    predictions = model.predict(input_datag)

    return np.argmax(predictions)


def process_frame(frame):

    # pass the blob through the network and obtain the detections and predictions
    # load the input image and construct an input blob for the image and resize image to fixed 300x300 pixels and then normalize it

    bounds = get_face_bounding_boxes(frame, 0.8)

    # Faces might not always be detected.
    if bounds.shape[0] > 0:

        print("Labeling")
        emotions = predict_emotions(frame, bounds)
        label_bounds(frame, bounds, emotions)

    return frame


print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(proto_text, face_extraction_weights)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels

    # show the output frame
    cv2.imshow("Frame", process_frame(vs.read()))
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"): break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()