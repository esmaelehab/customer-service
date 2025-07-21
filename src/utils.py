"""Utility functions for coffee cup counter."""

from shapely.geometry import Point, Polygon


def point_in_zone(point, zone_polygon):
    """
    Check if a point is inside a zone polygon.
    
    Args:
        point: Tuple of (x, y) coordinates
        zone_polygon: Shapely Polygon object
        
    Returns:
        bool: True if point is inside the polygon
    """
    return zone_polygon.contains(Point(point))


def get_object_center(bbox):
    """
    Get center point of bounding box.
    
    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        
    Returns:
        tuple: Center coordinates (x, y)
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def create_zone_state():
    """
    Create initial state dictionary for a zone.
    
    Returns:
        dict: Initial zone state
    """
    return {
        'first_filled_detected': None,
        'last_empty_detected': None,
        'last_count_time': 0,
        'count': 0,
        'current_detections': {'empty': [], 'filled': []},
        'in_cooldown': False,
        'cooldown_start_time': 0
    }


def validate_zone_points(zone_points):
    """
    Validate zone points format.
    
    Args:
        zone_points: List of points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        
    Returns:
        bool: True if valid
        
    Raises:
        ValueError: If points format is invalid
    """
    if not isinstance(zone_points, list) or len(zone_points) != 4:
        raise ValueError("Zone points must be a list of 4 points")
    
    for point in zone_points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("Each point must be a tuple/list of 2 coordinates")
    
    return True