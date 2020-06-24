import os
import cv2
import time, datetime
import mrcnn
import mrcnn.config
import mrcnn.utils
import queue
import threading
import numpy as np
import winsound
from mrcnn.model import MaskRCNN
from mrcnn import visualize
from pathlib import Path

VIDEO_SOURCE = "./assets/sample1.mp4" # put here URL to the video stream

###########################################################################################################################
# Configuration that will be used by the Mask-RCNN library
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.6

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
def get_other_objects_boxes(boxes, class_ids):
    object_boxes = []

    for i, box in enumerate(boxes):
        # If the detected object is motorcycle, skip it
        if class_ids[i]!=4:
            object_boxes.append(box)

    return np.array(object_boxes)

def get_motorcycle_boxes(boxes, class_ids, scores):
    motorcycle_boxes = []

    for i, box in enumerate(boxes):
        # If the detected object isn't a motorcycle, skip it
        if class_ids[i]==4 and scores[i]>0.9:
            motorcycle_boxes.append(box)

    return np.array(motorcycle_boxes)

###########################################################################################################################
# Root directory of the project
ROOT_DIR = Path(".")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Directory to save images
IMAGES_DIR = os.path.join(ROOT_DIR, "images")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

# Create folder for images
if not os.path.exists(IMAGES_DIR):
    os.mkdir(IMAGES_DIR)

# Create a Mask-RCNN model in inference mode
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())    

# Load pre-trained model
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Load the video file we want to run detection on
video_capture = VideoCapture(VIDEO_SOURCE)

# Loop over each frame of video
# counter = 0
while True:
    frame = video_capture.read()

    if frame is None:
        break
    
    # Crop the image
    # x = frame.shape[0]-800
    # y = frame.shape[1]-800
    # frame = frame[x:-1, y:-1]
    
    # Convert the image from BGR color (which OpenCV uses) to RGB color
    rgb_image = frame[:, :, ::-1]
    
    # Run the image through the Mask R-CNN model to get results.
    results = model.detect([rgb_image], verbose=0)

    # Mask R-CNN assumes we are running detection on multiple images.
    # We only passed in one image to detect, so only grab the first result.
    r = results[0]

    # The r variable will now have the results of detection:
    # - r['rois'] are the bounding box of each detected object
    # - r['class_ids'] are the class id (type) of each detected object
    # - r['scores'] are the confidence scores for each detection
    # - r['masks'] are the object masks for each detected object (which gives you the object outline)

    # Get boxes of all detected motorcycles
    motorcycle_boxes = get_motorcycle_boxes(r['rois'], r['class_ids'], r['scores'])
    print("Motorcycles detected: ", len(motorcycle_boxes))

    # Get boxes of all other objects
    other_objects_boxes = get_other_objects_boxes(r['rois'], r['class_ids'])
    print("Other objects detected: ", len(other_objects_boxes))
    
    # Draw green rectangles around motorcycles
    for box in motorcycle_boxes:
        y1, x1, y2, x2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # Draw red rectangles around other objects
    for box in other_objects_boxes:
        y1, x1, y2, x2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # See how much those objects overlap with the motorcycles
    overlaps = mrcnn.utils.compute_overlaps(motorcycle_boxes, other_objects_boxes)

    # Assume no overlaps with motorcyles
    is_overlap_detected = False

    for overlap in overlaps:
        # For this motorcycle, find the max amount it was covered by any
        # other objects  that was detected in our image
        max_IoU_overlap = np.max(overlap)
        print("Max overlap: ", max_IoU_overlap)
        if max_IoU_overlap > 0.2:
            is_overlap_detected = True

    print("Is overlap detected: ", is_overlap_detected)

    # If overlap is detected save the frame to the file and beep (Windows)
    if is_overlap_detected:
        filename = datetime.datetime.now().strftime('%G-%m-%d_%H-%M-%S_overlap.jpg')
        cv2.imwrite(os.path.join(IMAGES_DIR, filename), frame)
        winsound.Beep(2500, 1000)

    # Uncomment the lines below to see the result in real time
    # cv2.imshow('frame', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break    

cv2.destroyAllWindows()
