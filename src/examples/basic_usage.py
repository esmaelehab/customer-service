"""Basic usage example for coffee cup counter."""

from src.counter import CoffeeCupCounter


def main():
    """Basic usage example."""
    
    # Configuration
    model_path = 'C:\\Users\\ismail\\Downloads\\best (7).pt'
    input_video = 'C:\\Users\\ismail\\Downloads\\self customer service short 6.mp4'
    output_video = 'output_video.mp4'

    # Define zones (4 points each, clockwise from top-left)
    zone1_points = [[214, 394], [253, 434], [222, 460], [186, 419]]
    zone2_points = [[217, 395], [191, 369], [162, 395], [186, 423]]

    # Create counter
    counter = CoffeeCupCounter(
        model_path=model_path,
        zone1_points=zone1_points,
        zone2_points=zone2_points,
        confidence_threshold=0.5
    )

    # Process video
    results = counter.process_video(input_video, output_video)
    
    # Print results
    print(f"Processing complete!")
    print(f"Total cups counted: {results['total_count']}")
    print(f"Zone 1: {results['zone1_count']}")
    print(f"Zone 2: {results['zone2_count']}")


if __name__ == "__main__":
    main()