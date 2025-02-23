from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import cv2

from utils import common

class DeepSortTracker:

  def __init__(self):

    self.configs = common.load_configs()

    self.object_tracker = DeepSort(
        max_age = int(self.configs['max_age']),
        n_init = int(self.configs['n_init']),
        nms_max_overlap = float(self.configs['nms_max_overlap']),
        max_cosine_distance = float(self.configs['max_cosine_distance'])
    )
    
    self.tracked_ids = set()
    self.counter = {obj: 0 for obj in self.configs['tracked_objects']}
    self.prev_object_name = None
    self.current_object_name = None
    
    self.model = YOLO(self.configs['model_name'])
    self.classes = self.model.names
    
    # FLAGS
    self.DISP_COUNTER = self.configs['disp_counter']
    self.DISP_OBJ_TRACK_BOX = self.configs['disp_obj_track_box']
    
  def display_track(self, tracks, frame):
    height, width, _ = frame.shape
    threshold_y = int(height * 5 / 12)  # Define the threshold line at 5/12 of the frame height

    # Draw the threshold line on the frame
    cv2.line(frame, (0, threshold_y), (width, threshold_y), (0, 255, 255), 2)  

    for track in tracks:
        if not track.is_confirmed():
            continue  # Skip unconfirmed tracks

        track_id = track.track_id
        class_id = track.get_det_class()
        object_name = self.classes[int(class_id)]

        bbox = track.to_tlbr().astype(int)  # Convert bounding box to integer coordinates
        object_y = bbox[3]  # Get the y-coordinate of the bottom of the bounding box

        # Update the counter only when the object crosses the threshold from above
        if track_id not in self.tracked_ids and object_y > threshold_y:
            self.prev_object_name = self.current_object_name
            self.current_object_name = object_name
            self.tracked_ids.add(track_id)
            self.update_counter(object_name)  # Increment the counter

        # Draw the tracking box if enabled
        if self.DISP_OBJ_TRACK_BOX:
            self.plot_box_tracking(frame, bbox, object_name)

    self.display_tracking_info(frame)  # Display tracking information on the frame

  def update_counter(self, object_name):
    self.counter[object_name] = self.counter.get(object_name, 0) + 1

  def plot_box_tracking(self, frame, bbox, object_name):
    """
    Draw bounding boxes and tracking information on the frame.
    """
    color = (0, 255, 0)  # Red color for the box
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    label = f"{object_name}"
    cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
  
  def draw_overlay_box(self, frame, start_x, start_y, width, height, overlay_color, border_color):
    cv2.rectangle(frame, (start_x, start_y),
                  (start_x + width, start_y + height),
                  overlay_color, -1)
    cv2.rectangle(frame, (start_x, start_y),
                  (start_x + width, start_y + height),
                  border_color, 2)
  
  def display_tracking_info(self, frame):
    """
    Display tracking information: previous object, current object, and counter.
    """
    # Background box parameters for the "Detect" section
    overlay_start_x, overlay_start_y = 5, 5
    overlay_width, overlay_height = 320, 110
    overlay_color = (255, 230, 200)  # Background color (light blue)
    border_color = (0, 0, 0)  # Border color (black)

    # Draw the background box for "Detect"
    self.draw_overlay_box(frame, overlay_start_x, overlay_start_y, overlay_width, overlay_height, overlay_color, border_color)

    # Display "Detect" information
    cv2.putText(frame, "        Detect", (overlay_start_x + 10, overlay_start_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Prev Obj: {self.prev_object_name}",
                (overlay_start_x + 10, overlay_start_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
    cv2.putText(frame, f"Current Obj: {self.current_object_name}",
                (overlay_start_x + 10, overlay_start_y + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)

    # Background box parameters for the "Counter" section
    counter_start_y = overlay_start_y + overlay_height + 10
    counter_height = counter_start_y + (len(self.counter) + 1) * 20

    # Draw the background box for "Counter"
    self.draw_overlay_box(frame, overlay_start_x, counter_start_y, overlay_width, counter_height, overlay_color, border_color)

    # Display "Counter" information
    cv2.putText(frame, "        Counter", (overlay_start_x + 10, counter_start_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    y_offset = counter_start_y + 60
    for class_name, count in sorted(self.counter.items()):
        text = f"{class_name}: {count}"
        cv2.putText(frame, text, (overlay_start_x + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
        y_offset += 30

