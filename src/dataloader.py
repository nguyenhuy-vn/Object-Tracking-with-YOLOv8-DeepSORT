import cv2

from utils import common

class Dataloader:

  def __init__(self, data_path = None):

    self.configs = common.load_configs()

    self.DATA_SOURCE = self.configs['data_source']
    self.WEBCAM_ID = self.configs['webcam_id']
    self.DATA_PATH = data_path or self.configs['data_path']

    self.cap = None

  def run(self):

    if self.DATA_SOURCE == 'webcam':
      self.cap = cv2.VideoCapture(self.WEBCAM_ID)
    elif self.DATA_SOURCE == 'video file':
      self.cap = cv2.VideoCapture(self.DATA_PATH)
    else:
      print("Enter correct data source")
      return

    if not self.cap.isOpened():
      print(f"Error: Unable to open the {self.DATA_SOURCE}")
      return

  def get_capture(self):
    return self.cap