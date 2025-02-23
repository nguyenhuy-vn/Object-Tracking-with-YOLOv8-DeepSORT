import cv2
import numpy as np
import torch 

from ultralytics import YOLO

from utils import common

class Detector:

  def __init__(self):

    self.configs = common.load_configs()

    self.model = self.load_model(self.configs['model_name'])
    self.classes = self.model.names
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    self.tracked_objects = self.configs['tracked_objects']
    self.confidence_threshold = self.configs['confidence_threshold']  
    
    # FLAGS
    self.DISP_OBJ_DETECT_BOX = self.configs['disp_obj_detect_box']  

  def load_model(self, model_name):

    model = YOLO(model_name)
    return model

  def class_to_label(self, x):

    return self.classes[int(x)]

  def run(self, frame):
    yolo_result = self.model(frame)
    boxes = yolo_result[0].boxes

    labels = boxes.cls.cpu().numpy()  # label (class ID)
    bb_coords = boxes.xyxy.cpu().numpy()  # Bounding box [xmin, ymin, xmax, ymax]
    confidences = boxes.conf.cpu().numpy() # Confidence scores
    
    return labels, bb_coords, confidences

  def extract_detection(self, results, frame):

    labels, bb_coords, confidences = results
    detections = []
    num_objects = len(labels)

    for object_index in range(num_objects):
      row = bb_coords[object_index]
      confidence = confidences[object_index]
      class_id = int(labels[object_index])

      # check the confidence score of bounding box is valid or not
      if confidence < self.confidence_threshold:
        continue

      # check if detected object is in the tracked list or not
      detected_object = self.class_to_label(class_id)
      if detected_object not in self.tracked_objects:
        continue

      xmin, ymin, xmax, ymax = [int(coord) for coord in row]

      if self.DISP_OBJ_DETECT_BOX:
        # Draw the bounding box only if the object is in the lower half of the frame
        if ymax >= int(frame.shape[0] / 2):
          self.plot_box(xmin, ymin, xmax, ymax, detected_object, frame)

      detections.append([[xmin, ymin, xmax-xmin, ymax-ymin], confidence, class_id])
    return detections

  def plot_box(self, xmin, ymin, xmax, ymax, detected_object, frame):

    WHITE = (255, 255, 255)
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    cv2.putText(frame, f"{detected_object}", (xmin, ymin - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 2.1, WHITE, 2)
