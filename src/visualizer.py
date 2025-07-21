"""Visualization functions for coffee cup counter."""

import cv2
import numpy as np
from .config import Config


class Visualizer:
    """Handles all visualization and drawing operations."""
    
    def __init__(self, config=None):
        self.config = config or Config()
    
    def draw_zones(self, frame, zone1_points, zone2_points):
        """
        Draw zone outlines and fills on frame.
        
        Args:
            frame: OpenCV frame
            zone1_points: Zone 1 points as numpy array
            zone2_points: Zone 2 points as numpy array
        """
        zone1_pts = zone1_points.reshape((-1, 1, 2))
        zone2_pts = zone2_points.reshape((-1, 1, 2))

        # Draw zone outlines
        cv2.polylines(frame, [zone1_pts], True, self.config.COLORS['zone1'], 2)
        cv2.polylines(frame, [zone2_pts], True, self.config.COLORS['zone2'], 2)

        # Add subtle fill
        overlay = frame.copy()
        cv2.fillPoly(overlay, [zone1_pts], self.config.COLORS['zone1'])
        cv2.fillPoly(overlay, [zone2_pts], self.config.COLORS['zone2'])
        cv2.addWeighted(overlay, self.config.OVERLAY_ALPHA, frame, 0.9, 0, frame)

        # Zone labels
        zone1_label_pos = (zone1_points[0][0], zone1_points[0][1] - 10)
        zone2_label_pos = (zone2_points[0][0], zone2_points[0][1] - 10)

        cv2.putText(frame, "Zone 1", zone1_label_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.config.COLORS['zone1'], 2)
        cv2.putText(frame, "Zone 2", zone2_label_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.config.COLORS['zone2'], 2)

    def draw_status_panel(self, frame, zone1_count, zone2_count, current_time, zone1_state, zone2_state):
        """
        Draw status information panel.
        
        Args:
            frame: OpenCV frame
            zone1_count: Zone 1 count
            zone2_count: Zone 2 count
            current_time: Current time
            zone1_state: Zone 1 state dictionary
            zone2_state: Zone 2 state dictionary
        """
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), 
                     (self.config.PANEL_WIDTH, self.config.PANEL_HEIGHT), 
                     self.config.COLORS['panel_bg'], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Border
        cv2.rectangle(frame, (10, 10), 
                     (self.config.PANEL_WIDTH, self.config.PANEL_HEIGHT), 
                     self.config.COLORS['text'], 1)

        # Total customers
        total_customers = zone1_count + zone2_count
        cv2.putText(frame, f"Customers prepared {total_customers} today", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.config.COLORS['text'], 2)
        
        # Separator line
        cv2.line(frame, (20, 45), (self.config.PANEL_WIDTH - 20, 45), 
                self.config.COLORS['text'], 1)

      
    def draw_detections(self, frame, zone1_detections, zone2_detections):
        """
        Draw detection boxes and labels.
        
        Args:
            frame: OpenCV frame
            zone1_detections: Zone 1 detection data
            zone2_detections: Zone 2 detection data
        """
        # Zone 1 detections
        for cup_type, detections in zone1_detections.items():
            color = self.config.COLORS['zone1_empty'] if cup_type == 'empty' else self.config.COLORS['zone1_filled']
            self._draw_detection_boxes(frame, detections, color, cup_type)

        # Zone 2 detections
        for cup_type, detections in zone2_detections.items():
            color = self.config.COLORS['zone2_empty'] if cup_type == 'empty' else self.config.COLORS['zone2_filled']
            self._draw_detection_boxes(frame, detections, color, cup_type)

    def _draw_detection_boxes(self, frame, detections, color, cup_type):
        """Draw detection boxes for a specific cup type."""
        for det in detections:
            bbox = det['bbox'].astype(int)
            x1, y1, x2, y2 = bbox

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

            # Label
            label = f"{'E' if cup_type == 'empty' else 'F'}{det['conf']:.2f}"
            cv2.putText(frame, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Center dot
            center = (int(det['center'][0]), int(det['center'][1]))
            cv2.circle(frame, center, 2, color, -1)