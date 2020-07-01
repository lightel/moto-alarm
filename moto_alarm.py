import os
import cv2
import time, datetime
import queue
import threading
import numpy as np
import winsound
import sys
import urllib.request
from pathlib import Path

VIDEO_SOURCE = "./assets/sample1.mp4" # put here URL to the video stream

###########################################################################################################################
# Bufferless VideoCapture
class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # Read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)
        self.cap.release()

    def read(self):
        try:
            return self.q.get(block=True, timeout=5)
        except queue.Empty:
            return None
###########################################################################################################################
#
def getIoU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA) * (yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
        return iou

###########################################################################################################################
# Root directory of the project
ROOT_DIR = Path(".")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Directory to save images
IMAGES_DIR = os.path.join(ROOT_DIR, "images")

# Local path to trained weights file
YOLOV3_WEIGHTS_PATH = os.path.join(ROOT_DIR, "yolov3.weights")

# Local path to the YOLO configuration file
YOLOV3_CFG_PATH = os.path.join(ROOT_DIR, "yolov3.cfg")

# Download YOLO  weights if needed
if not os.path.exists(YOLOV3_WEIGHTS_PATH):
    print(YOLOV3_WEIGHTS_PATH, "has not been found")
    print("Downloading YOLOv3 weights from https://pjreddie.com/media/files/yolov3.weights...")
    urllib.request.urlretrieve("https://pjreddie.com/media/files/yolov3.weights", YOLOV3_WEIGHTS_PATH)
    print("Weights have been downloaded and saved to", YOLOV3_WEIGHTS_PATH)

# Create folder for images
if not os.path.exists(IMAGES_DIR):
    os.mkdir(IMAGES_DIR)

# Load YOLO
net = cv2.dnn.readNet(YOLOV3_WEIGHTS_PATH, YOLOV3_CFG_PATH)

# Defin output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

# Load the video file we want to run detection on
video_capture = VideoCapture(VIDEO_SOURCE)

overlapped_frames = 0

# Loop over each frame of video
# counter = 0
while True:
    original_frame = video_capture.read()

    if original_frame is None:
        break
    
    # Crop the image
    # x = frame.shape[0]-800
    # y = frame.shape[1]-800
    # frame = frame[x:-1, y:-1]
    
    # Resize the image
    # scale = 0.3
    # width = int(original_frame.shape[1] * scale)
    # height = int(original_frame.shape[0] * scale)
    frame = cv2.resize(original_frame, None, fx=0.4, fy=0.3)
    height, width, channels = frame.shape
    #frame = original_frame
        
    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0,0,0), True, False)

    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids=[]
    confidences=[]
    boxes=[]

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                #cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)

    people_boxes = []
    motorcycle_boxes = []
    
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            if class_ids[i]==0: # Get boxes of all detected people
                people_boxes.append([x, y, x+w, y+h])
            elif class_ids[i]==3: # Get boxes of all detected motorcycles
                motorcycle_boxes.append([x, y, x+w, y+h])
    
    print("Motorcycles detected:", len(motorcycle_boxes))
    print("People detected:", len(people_boxes))
    
    # Draw green rectangles around motorcycles
    for box in motorcycle_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # Draw red rectangles around people
    for box in people_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # Assume no overlaps with motorcyles
    is_overlap_detected = False

    # Detecting overlaps
    for motorcycle_box in motorcycle_boxes:
        for people_box in people_boxes:
            if getIoU(motorcycle_box, people_box)>=0.2:
                is_overlap_detected = True
                break
        if is_overlap_detected:
            break
    print("Is overlap detected: ", is_overlap_detected)

    # If overlap is detected save the frame to the file and beep (Windows)
    if is_overlap_detected:
        overlapped_frames += 1
    else:
        # reset overlap counter
        overlapped_frames = 0

    if overlapped_frames==3:
        overlapped_frames = 0
        filename = datetime.datetime.now().strftime('%G-%m-%d_%H-%M-%S_overlap.jpg')
        cv2.imwrite(os.path.join(IMAGES_DIR, filename), original_frame)
        winsound.Beep(2500, 1000)

    # Uncomment the code below to see the result in real time
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break    

cv2.destroyAllWindows()
