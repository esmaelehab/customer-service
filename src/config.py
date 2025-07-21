"""Configuration settings for coffee cup counter."""

class Config:
    """Configuration class for coffee cup counter settings."""
    
    # Detection settings
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5
    
    # Timing settings
    DWELL_TIME = 0.5  # Time in seconds a filled cup must be present before counting
    COOLDOWN_TIME = 25.0  # Cooldown period in seconds after counting
    
    # Class mappings
    CLASS_EMPTY_CUP = 0
    CLASS_FILLED_CUP = 1
    
    # Visualization colors (BGR format for OpenCV)
    COLORS = {
        'zone1': (0, 255, 0),      # Green
        'zone2': (255, 0, 0),      # Blue
        'zone1_empty': (0, 255, 255),  # Yellow
        'zone1_filled': (0, 255, 0),   # Green
        'zone2_empty': (255, 255, 0),  # Cyan
        'zone2_filled': (255, 0, 0),   # Blue
        'text': (255, 255, 255),       # White
        'panel_bg': (0, 0, 0),         # Black
    }
    
    # UI settings
    PANEL_HEIGHT = 170
    PANEL_WIDTH = 450
    OVERLAY_ALPHA = 0.1