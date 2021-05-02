import cv2
import numpy as np
from pathlib import Path

base_path = Path(__file__).parent.parent.parent

proto_text = f"{base_path}/app/common/deploy.prototxt.txt"
face_extraction_weights = f"{base_path}/app/common/opencv_face_detector.caffemodel"

    def detect_face_from_image(self, image, confidence_threshold=0.7, padding_size=(900, 900)):
        image = cv2.imread(path_to_image)
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        self._net.setInput(blob)
        detections = self._net.forward()

        print(detections.shape)

        num_detections = detections.shape[2]
        confidences = detections[0, 0, 0:num_detections, 2]
        # detections = detections[0, 0, confidences > confidence_threshold, 0:7]

        print(detections[0, 0, :, 2])

        print(detections[0, 0, 0])
        print(detections[0, 0, 1])

        if detections.size == 0:
            return

        # get the size of each box as the sum of two sides of the rectangle
        num_detections = detections.shape[0]
        sizes = ((detections[0:num_detections, 5] - detections[0:num_detections, 3])
                 + (detections[0:num_detections, 6] - detections[0:num_detections, 4]))

        print(f"Bounds: {detections[np.argmax(sizes), 3:7]}")

        box = detections[np.argmax(sizes), 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        maxY = image.shape[0]
        maxX = image.shape[1]

        print(f"Bounds: {startX} {startY} {endX} {endY}")

        return image[startY:endY, startX:endX]


face_detector = FaceDetector()
# image = face_detector.detect_face_from_image(f"/home/lumayer/DataSciencePracticum/app/common/wp1816999.jpg", confidence_threshold=0.0)
image = face_detector.detect_face_from_image(f"/data/datasets/affectNet/val_set/images/2762.jpg", confidence_threshold=0.0)
print(f"Image Shape: {image.shape}")

export_image_to_file("test", image)