import cv2
import numpy as np
import argparse
from scipy.special import comb

# Import the necessary modules for object detection (assuming you're using OpenCV's DNN module)
# Modify this based on the object detection model you want to use
# For example, you can use YOLO, SSD, Faster R-CNN, etc.
# Here we use MobileNet SSD as an example
net = cv2.dnn.readNetFromTensorflow("frozen_inference_graph.pb", "ssd_mobilenet_v2_coco.pbtxt")

# Define the classes for COCO dataset (for MobileNet SSD)
classes = ["background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
           "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
           "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", 
           "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", 
           "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", 
           "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", 
           "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", 
           "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", 
           "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# Args setting
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-i', "--input", help="input file video")
parser.add_argument('--leftPoint', type=int, help="Left rail offset", default=450)
parser.add_argument('--rightPoint', type=int, help="Right rail offset", default=840)
parser.add_argument('--topPoint', type=int, help="Top rail offset", default=330)
args = parser.parse_args()


def main():
    cap = VideoCapture(args.input)

    # Initialization for line detection
    expt_startLeft = args.leftPoint
    expt_startRight = args.rightPoint
    expt_startTop = args.topPoint

    # Value initialize
    left_maxpoint = [0] * 50
    right_maxpoint = [195] * 50

    # Convolution filter
    kernel = np.array([
        [-1, 1, 0, 1, -1],
        [-1, 1, 0, 1, -1],
        [-1, 1, 0, 1, -1],
        [-1, 1, 0, 1, -1],
        [-1, 1, 0, 1, -1]
    ])

    # Next frame availability
    r = True
    first = True

    while r is True:
        r, frame = cap.read()
        if frame is None:
            break

        # Cut away invalid frame area
        valid_frame = frame[expt_startTop:, expt_startLeft:expt_startRight]

        # Gray scale transform
        gray_frame = cv2.cvtColor(valid_frame, cv2.COLOR_BGR2GRAY)

        # Histogram equalization image
        histeqaul_frame = cv2.equalizeHist(gray_frame)

        # Apply Gaussian blur
        blur_frame = cv2.GaussianBlur(histeqaul_frame, (5, 5), 5)

        # Merge current frame and last frame
        if first is True:
            merge_frame = blur_frame
            first = False
            old_valid_frame = merge_frame.copy()
        else:
            merge_frame = cv2.addWeighted(blur_frame, 0.2, old_valid_frame, 0.8, 0)
            old_valid_frame = merge_frame.copy()

        # Convolution filter
        conv_frame = cv2.filter2D(merge_frame, -1, kernel)

        # Initialization for sliding window property
        sliding_window = [20, 190, 200, 370]
        slide_interval = 15
        slide_height = 15
        slide_width = 60

        # Object detection
        objects = detect_objects(valid_frame)

        # Draw bounding boxes around detected objects
        for object in objects:
            label, confidence, bbox = object
            draw_bbox(valid_frame, label, confidence, bbox)

        cv2.imshow('Video', valid_frame)
        cv2.waitKey(1)
    print('finish')


# For reading video
class VideoCapture:
    def __init__(self, path):
        path= 'video/test1.mp4'
        self.video = cv2.VideoCapture(path)

    def __del__(self):
        self.video.release()

    def read(self):
        # Single frame of video
        ret, frame = self.video.read()
        return frame is not None, frame


# Object detection function
def detect_objects(frame):
    # Resize frame to fit the network input size
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    # Set the input to the network
    net.setInput(blob)

    # Forward pass
    detections = net.forward()

    # Parse the detections
    objects = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Filter out weak detections
            class_id = int(detections[0, 0, i, 1])
            label = classes[class_id]
            bbox = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            objects.append((label, confidence, bbox.astype("int")))
    return objects


# Function to draw bounding boxes around detected objects
def draw_bbox(frame, label, confidence, bbox):
    x, y, x2, y2 = bbox
    cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "{}: {:.2f}".format(label, confidence), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


if __name__ == '__main__':
    main()
