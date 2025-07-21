"""Main coffee cup counter class."""

import cv2
import numpy as np
import time
from ultralytics import YOLO
from shapely.geometry import Polygon

from .config import Config
from .utils import point_in_zone, get_object_center, create_zone_state, validate_zone_points
from .visualizer import Visualizer


class CoffeeCupCounter:
    """Main class for counting coffee cups in defined zones."""
    
    def __init__(self, model_path, zone1_points, zone2_points, 
                 confidence_threshold=None, config=None):
        """
        Initialize the coffee cup counter.

        Args:
            model_path: Path to YOLO model file
            zone1_points: List of 4 points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] for zone 1
            zone2_points: List of 4 points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] for zone 2
            confidence_threshold: Minimum confidence for detection (default from config)
            config: Custom Config instance (optional)
        """
        self.config = config or Config()
        
        # Validate inputs
        validate_zone_points(zone1_points)
        validate_zone_points(zone2_points)
        
        # Initialize model
        self.model = YOLO(model_path)
        
        # Setup zones
        self.zone1_polygon = Polygon(zone1_points)
        self.zone2_polygon = Polygon(zone2_points)
        self.zone1_points = np.array(zone1_points, dtype=np.int32)
        self.zone2_points = np.array(zone2_points, dtype=np.int32)
        
        # Set confidence threshold
        self.confidence_threshold = (confidence_threshold or 
                                   self.config.DEFAULT_CONFIDENCE_THRESHOLD)
        
        # Initialize zone states
        self.zone1_state = create_zone_state()
        self.zone2_state = create_zone_state()
        
        # Initialize visualizer
        self.visualizer = Visualizer(self.config)
        
        # Video timing
        self.video_start_time = None
        self.frame_count = 0
        self.fps = None
        
        self._print_initialization_info(model_path, zone1_points, zone2_points)

    def _print_initialization_info(self, model_path, zone1_points, zone2_points):
        """Print initialization information."""
        print(f"Coffee Cup Counter Initialized")
        print(f"Model: {model_path}")
        print(f"Zone 1 points: {zone1_points}")
        print(f"Zone 2 points: {zone2_points}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print("Classes: 0=Empty Cup, 1=Filled Cup")

    def get_video_time(self):
        """Get accurate time based on video frame count and FPS."""
        if self.fps and self.video_start_time:
            return self.video_start_time + (self.frame_count / self.fps)
        return time.time()

    def process_zone_detections(self, zone_state, current_time):
        """
        Process detections for a specific zone.
        
        Args:
            zone_state: Zone state dictionary
            current_time: Current timestamp
            
        Returns:
            bool: True if a count was made
        """
        filled_cups = zone_state['current_detections']['filled']
        empty_cups = zone_state['current_detections']['empty']

        # Check if cooldown has expired
        if zone_state['in_cooldown']:
            time_since_cooldown = current_time - zone_state['cooldown_start_time']
            if time_since_cooldown >= self.config.COOLDOWN_TIME:
                zone_state['in_cooldown'] = False

        # Handle empty cup detection
        if empty_cups:
            zone_state['last_empty_detected'] = current_time
            zone_state['first_filled_detected'] = None
            if zone_state['in_cooldown']:
                zone_state['in_cooldown'] = False

        # Handle filled cup detection
        if filled_cups and not zone_state['in_cooldown']:
            if zone_state['first_filled_detected'] is None:
                zone_state['first_filled_detected'] = current_time

            time_since_first_filled = current_time - zone_state['first_filled_detected']

            if time_since_first_filled >= self.config.DWELL_TIME:
                # Count the cup
                zone_state['count'] += 1
                zone_state['last_count_time'] = current_time
                zone_state['first_filled_detected'] = None

                # Start cooldown
                zone_state['in_cooldown'] = True
                zone_state['cooldown_start_time'] = current_time
                
                return True

        # Reset timer if no filled cups
        if not filled_cups:
            zone_state['first_filled_detected'] = None

        return False

    def process_detections(self, detections, current_time):
        """
        Process YOLO detections and update zone states.
        
        Args:
            detections: YOLO detection results
            current_time: Current timestamp
        """
        # Clear current detections
        self.zone1_state['current_detections'] = {'empty': [], 'filled': []}
        self.zone2_state['current_detections'] = {'empty': [], 'filled': []}

        # Process each detection
        for detection in detections:
            if detection.conf < self.confidence_threshold:
                continue

            bbox = detection.xyxy[0].cpu().numpy()
            cls = int(detection.cls[0].cpu().numpy())
            center = get_object_center(bbox)
            cup_type = 'empty' if cls == self.config.CLASS_EMPTY_CUP else 'filled'

            detection_data = {
                'bbox': bbox,
                'center': center,
                'conf': detection.conf.item(),
                'cls': cls
            }

            # Assign to appropriate zone
            if point_in_zone(center, self.zone1_polygon):
                self.zone1_state['current_detections'][cup_type].append(detection_data)
            elif point_in_zone(center, self.zone2_polygon):
                self.zone2_state['current_detections'][cup_type].append(detection_data)

        # Process zone counting logic
        zone1_counted = self.process_zone_detections(self.zone1_state, current_time)
        zone2_counted = self.process_zone_detections(self.zone2_state, current_time)

        # Log counts
        if zone1_counted:
            print(f"Zone 1: Cup counted! Total: {self.zone1_state['count']}")
        if zone2_counted:
            print(f"Zone 2: Cup counted! Total: {self.zone2_state['count']}")

    def process_frame(self, frame):
        """
        Process a single frame.
        
        Args:
            frame: OpenCV frame
            
        Returns:
            tuple: (processed_frame, zone1_count, zone2_count)
        """
        current_time = self.get_video_time()

        # Run YOLO detection
        results = self.model(frame, verbose=False)

        # Process detections if any found
        if results[0].boxes is not None:
            self.process_detections(results[0].boxes, current_time)

        # Draw visualizations
        self.visualizer.draw_zones(frame, self.zone1_points, self.zone2_points)
        self.visualizer.draw_detections(
            frame, 
            self.zone1_state['current_detections'],
            self.zone2_state['current_detections']
        )
        self.visualizer.draw_status_panel(
            frame, 
            self.zone1_state['count'], 
            self.zone2_state['count'],
            current_time,
            self.zone1_state,
            self.zone2_state
        )

        self.frame_count += 1
        return frame, self.zone1_state['count'], self.zone2_state['count']

    def process_video(self, input_path, output_path=None):
        """
        Process entire video file.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video (optional)
            
        Returns:
            dict: Processing results
        """
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.video_start_time = time.time()
        self.frame_count = 0

        # Setup video writer if output path provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))

        print(f"Processing video: {width}x{height} @ {self.fps} FPS")
        start_time = time.time()

        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, _, _ = self.process_frame(frame)

            if out:
                out.write(processed_frame)

            # Progress logging
            if self.frame_count % 30 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {self.frame_count} frames in {elapsed:.2f}s")

        # Cleanup
        cap.release()
        if out:
            out.release()

        # Return results
        results = {
            'total_frames': self.frame_count,
            'zone1_count': self.zone1_state['count'],
            'zone2_count': self.zone2_state['count'],
            'total_count': self.zone1_state['count'] + self.zone2_state['count'],
            'processing_time': time.time() - start_time
        }

        print(f"\nVideo processing completed!")
        print(f"Total customers: {results['total_count']}")
        print(f"Zone 1: {results['zone1_count']}, Zone 2: {results['zone2_count']}")
        if output_path:
            print(f"Output saved to: {output_path}")

        return results

    def get_counts(self):
        """
        Get current counts.
        
        Returns:
            dict: Current counting statistics
        """
        return {
            'zone1_count': self.zone1_state['count'],
            'zone2_count': self.zone2_state['count'],
            'total_count': self.zone1_state['count'] + self.zone2_state['count']
        }

    def reset_counts(self):
        """Reset all counts to zero."""
        self.zone1_state = create_zone_state()
        self.zone2_state = create_zone_state()
        print("Counts reset to zero")