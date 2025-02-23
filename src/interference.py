import cv2
import os
import time

from utils import common
from .dataloader import Dataloader
from .detector import Detector
from .tracker import DeepSortTracker

class ModelInterference:
    def __init__(self, data_path = None):
        self.configs = common.load_configs()

        # FLAGS
        self.DISP_FPS = self.configs['disp_fps']
        self.DISP_VIDEO = self.configs['disp_video']  # Flag to show video
        self.SAVE_VIDEO = self.configs['save_video']  # Flag to save video
        self.WINDOW_NAME = self.configs['window_name']

        """ Disable saving video for webcam
        self.DATA_SOURCE = self.configs['data_source']
        if self.DATA_SOURCE == "webcam":
            self.SAVE_VIDEO = False 
        """
        
        # Initialize Dataloader, Detector, and Tracker
        self.dataloader = Dataloader(data_path=data_path)
        self.detector = Detector()
        self.tracker = DeepSortTracker()

        # Video writer (in case need save video)
        self.video_writer = None

        # generate auto output for saving video file
        if self.configs['data_source'] == 'video file' and self.configs['data_path']:
            self.SAVE_PATH = self.generate_save_path(self.configs['data_path'])
            
    def run(self):
        self.dataloader.run()
        capture = self.dataloader.get_capture()

        if not capture:
            print("Error: Unable to initialize the capture source.")
            return

        # initialize video writer if SAVE_VIDEO is enabled
        self.initialize_video_writer(capture)
        
        # Start processing the input stream
        start_time = time.time()
        frame_count = 0

        while True:
            ret, frame = capture.read()

            if not ret:
                print("End of stream or error reading the frame.")
                break
            
            # Object detection
            results = self.detector.run(frame)
            detections = self.detector.extract_detection(results, frame)

            # Object tracking
            tracker_outputs = self.tracker.object_tracker.update_tracks(detections, frame=frame)
            
            # Display tracking results
            self.tracker.display_track(tracker_outputs, frame)

            # Display FPS if enabled
            
            if self.DISP_FPS:
                self.display_fps(frame, frame_count, start_time)

            # save frame into video if SAVE_VIDEO is enabled
            if self.SAVE_VIDEO:
                self.save_video_frame(frame)

            # Display frame if DISP_VIDEO is enabled
            if self.DISP_VIDEO:
                if self.display_video_frame(frame):
                    break
                    
            frame_count += 1

        # release the resources
        capture.release()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()
    
    def generate_save_path(self, data_path):
        file_name = os.path.basename(data_path)
        file_name_without_extension = os.path.splitext(file_name)[0]
        return os.path.join(self.configs['save_path'], f"{file_name_without_extension}_output.avi")
        
    def initialize_video_writer(self, capture):
        if not self.SAVE_VIDEO:
            return
        
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS)) or 30  
        
        self.video_writer = cv2.VideoWriter(
            self.SAVE_PATH,
            cv2.VideoWriter_fourcc(*'XVID'), #CODEC
            fps,
            (frame_width, frame_height)
        )
        
    def display_fps(self, frame, frame_count, start_time):
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(
            frame, f"FPS: {fps:.2f}", (10, frame.shape[0]-50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
    
    def save_video_frame(self, frame):
        self.video_writer.write(frame)
        
    def display_video_frame(self, frame):
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        
        cv2.resizeWindow(self.WINDOW_NAME, 1280, 800)

        cv2.imshow(self.WINDOW_NAME, frame)
        
        # exit by press 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return True
        
        return False