from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject
from imutils.video import VideoStream
from itertools import zip_longest
from utils.mailer import Mailer
from imutils.video import FPS
from utils import thread
import numpy as np
import threading
import argparse
import datetime
import schedule
import logging
import imutils
import time
import dlib
import json
import csv
import cv2

# Import YOLOv8
from ultralytics import YOLO

# Execution start time
start_time = time.time()

# Setup logger
logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")
logger = logging.getLogger(__name__)

# Load configuration
with open("utils/config.json", "r") as file:
    config = json.load(file)


class YOLOv8Detector:
    """
    YOLOv8 Detector wrapper for people detection
    Replaces MobileNet SSD with modern YOLO architecture
    """
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.4):
        """
        Initialize YOLOv8 detector
        
        Args:
            model_path: Path to YOLOv8 weights (.pt file)
            conf_threshold: Confidence threshold for detections
        """
        logger.info(f"Loading YOLOv8 model from {model_path}...")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        logger.info("YOLOv8 model loaded successfully!")
    
    def detect_people(self, frame):
        """
        Detect people in frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of detections: [(x1, y1, x2, y2, confidence), ...]
        """
        # Run YOLOv8 inference
        # classes=[0] -> only detect 'person' class from COCO dataset
        # verbose=False -> suppress output logs
        results = self.model(
            frame, 
            conf=self.conf_threshold,
            classes=[0],  # Person class only
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                # Get confidence score
                conf = box.conf[0].cpu().numpy()
                
                detections.append({
                    'box': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': float(conf)
                })
        
        return detections


def parse_arguments():
    """Parse command line arguments"""
    ap = argparse.ArgumentParser(description="People Counter with YOLOv8n")
    
    # YOLOv8 model weights
    ap.add_argument("-w", "--weights", 
                    type=str, 
                    default="yolov8n.pt",
                    help="path to YOLOv8 weights file (default: yolov8n.pt)")
    
    # Input/Output
    ap.add_argument("-i", "--input", 
                    type=str,
                    help="path to optional input video file")
    ap.add_argument("-o", "--output", 
                    type=str,
                    help="path to optional output video file")
    
    # Detection parameters
    ap.add_argument("-c", "--confidence", 
                    type=float, 
                    default=0.4,
                    help="minimum probability to filter weak detections (default: 0.4)")
    ap.add_argument("-s", "--skip-frames", 
                    type=int, 
                    default=30,
                    help="number of skip frames between detections (default: 30)")
    
    args = vars(ap.parse_args())
    return args


def send_mail():
    """Send email alerts when threshold exceeded"""
    try:
        Mailer().send(config["Email_Receive"])
    except Exception as e:
        logger.error(f"Failed to send email: {e}")


def log_data(move_in, in_time, move_out, out_time):
    """
    Log counting data to CSV file
    
    Args:
        move_in: List of people entering
        in_time: List of entry timestamps
        move_out: List of people exiting
        out_time: List of exit timestamps
    """
    data = [move_in, in_time, move_out, out_time]
    export_data = zip_longest(*data, fillvalue='')

    with open('utils/data/logs/counting_data.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        if myfile.tell() == 0:  # Write header if file is empty
            wr.writerow(("Move In", "In Time", "Move Out", "Out Time"))
            wr.writerows(export_data)


def people_counter():
    """
    Main people counting function with YOLOv8
    """
    # Parse arguments
    args = parse_arguments()
    
    # Initialize YOLOv8 detector
    detector = YOLOv8Detector(
        model_path=args["weights"],
        conf_threshold=args["confidence"]
    )
    
    # Initialize video stream
    if not args.get("input", False):
        logger.info("Starting the live stream...")
        vs = VideoStream(config["url"]).start()
        time.sleep(2.0)
    else:
        logger.info("Starting the video...")
        vs = cv2.VideoCapture(args["input"])
    
    # Initialize video writer
    writer = None
    W = None
    H = None
    
    # Initialize centroid tracker
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}
    
    # Initialize counters
    totalFrames = 0
    totalDown = 0  # People entering
    totalUp = 0    # People exiting
    total = []
    move_out = []
    move_in = []
    out_time = []
    in_time = []
    
    # Start FPS counter
    fps = FPS().start()
    
    # Use threading if enabled
    if config["Thread"]:
        vs = thread.ThreadingClass(config["url"])
    
    # Main processing loop
    while True:
        # Read frame
        frame = vs.read()
        frame = frame[1] if args.get("input", False) else frame
        
        # Check if video ended
        if args["input"] is not None and frame is None:
            break
        
        # Resize frame for faster processing
        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get frame dimensions
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        
        # Initialize video writer
        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)
        
        # Initialize status and rects
        status = "Waiting"
        rects = []
        
        # Perform detection every N frames
        if totalFrames % args["skip_frames"] == 0:
            status = "Detecting"
            trackers = []
            
            # Detect people using YOLOv8
            detections = detector.detect_people(frame)
            
            # Create dlib tracker for each detection
            for detection in detections:
                (startX, startY, endX, endY) = detection['box']
                
                # Initialize dlib correlation tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)
                
                trackers.append(tracker)
        
        # Otherwise, use existing trackers
        else:
            for tracker in trackers:
                status = "Tracking"
                
                # Update tracker
                tracker.update(rgb)
                pos = tracker.get_position()
                
                # Get bounding box coordinates
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                
                rects.append((startX, startY, endX, endY))
        
        # Draw horizontal line (entrance border)
        cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 0), 3)
        cv2.putText(frame, "-Prediction border - Entrance-", 
                   (10, H - 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Update centroid tracker
        objects = ct.update(rects)
        
        # Loop over tracked objects
        for (objectID, centroid) in objects.items():
            # Get trackable object
            to = trackableObjects.get(objectID, None)
            
            # Create new trackable object if needed
            if to is None:
                to = TrackableObject(objectID, centroid)
            
            # Otherwise, track the object
            else:
                # Calculate direction
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)
                
                # Check if object needs to be counted
                if not to.counted:
                    # Moving UP (exiting)
                    if direction < 0 and centroid[1] < H // 2:
                        totalUp += 1
                        date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                        move_out.append(totalUp)
                        out_time.append(date_time)
                        to.counted = True
                    
                    # Moving DOWN (entering)
                    elif direction > 0 and centroid[1] > H // 2:
                        totalDown += 1
                        date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                        move_in.append(totalDown)
                        in_time.append(date_time)
                        
                        # Check threshold for alert
                        total_inside = len(move_in) - len(move_out)
                        if total_inside >= config["Threshold"]:
                            cv2.putText(frame, "-ALERT: People limit exceeded-", 
                                       (10, frame.shape[0] - 80),
                                       cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
                            
                            # Send email alert
                            if config["ALERT"]:
                                logger.info("Sending email alert...")
                                email_thread = threading.Thread(target=send_mail)
                                email_thread.daemon = True
                                email_thread.start()
                                logger.info("Alert sent!")
                        
                        to.counted = True
                        
                        # Update total
                        total = []
                        total.append(len(move_in) - len(move_out))
            
            # Store trackable object
            trackableObjects[objectID] = to
            
            # Draw tracking info
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)
        
        # Prepare info for display
        info_status = [
            ("Exit", totalUp),
            ("Enter", totalDown),
            ("Status", status),
            ("Detector", "YOLOv8n"),
        ]
        
        info_total = [
            ("Total people inside", ', '.join(map(str, total))),
        ]
        
        # Display status info
        for (i, (k, v)) in enumerate(info_status):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Display total info
        for (i, (k, v)) in enumerate(info_total):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (265, H - ((i * 20) + 60)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Log data if enabled
        if config["Log"]:
            log_data(move_in, in_time, move_out, out_time)
        
        # Write frame to output video
        if writer is not None:
            writer.write(frame)
        
        # Display frame
        cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
        key = cv2.waitKey(1) & 0xFF
        
        # Break on 'q' key
        if key == ord("q"):
            break
        
        # Update counters
        totalFrames += 1
        fps.update()
        
        # Timer to auto-stop (8 hours)
        if config["Timer"]:
            end_time = time.time()
            num_seconds = (end_time - start_time)
            if num_seconds > 28800:  # 8 hours
                break
    
    # Stop FPS counter
    fps.stop()
    logger.info("Elapsed time: {:.2f}".format(fps.elapsed()))
    logger.info("Approx. FPS: {:.2f}".format(fps.fps()))
    
    # Release resources
    if config["Thread"]:
        vs.release()
    
    # Close windows
    cv2.destroyAllWindows()


# Main execution
if __name__ == "__main__":
    # Check if scheduler is enabled
    if config["Scheduler"]:
        # Run at 09:00 AM every day
        schedule.every().day.at("09:00").do(people_counter)
        logger.info("Scheduler enabled. Waiting for scheduled time...")
        while True:
            schedule.run_pending()
            time.sleep(1)
    else:
        # Run immediately
        people_counter()
