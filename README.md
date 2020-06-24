# Moto Alarm

Every motorcycle owner gets a bit nervous when people touch their toy without asking their permission. This tool is detecting if other objects such as cars and people overlaps with the bounding box of motorcycles and notifies the owner by beeping (Windows only) and storing the image. The tool is built using Mask R-CNN for Object Detection and Segmentation, COCO model and OpenCV.

Limitations:
* The algorithm detects all the motorcycles on the video frame. To recognize a specific motorcyle only need to train the custom model.
* This tool is very slow. It takes about 5 seconds to detect objects for a single frame.
