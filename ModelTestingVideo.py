import os
import cv2
import argparse
import time
from ultralytics import YOLO

def load_model(model_path):
    """
    Load a trained YOLOv8 model.
    
    Args:
        model_path: Path to the trained model (.pt file)
        
    Returns:
        Loaded YOLO model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = YOLO(model_path)
    print(f"Model loaded successfully from {model_path}")
    return model

def process_video(model, video_path, output_path=None, conf_threshold=0.25, show_video=True):
    """
    Process a video with the trained model.
    
    Args:
        model: Loaded YOLO model
        video_path: Path to the video file
        output_path: Path to save the output video (optional)
        conf_threshold: Confidence threshold for detections
        show_video: Whether to display video during processing
        
    Returns:
        None
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found at {video_path}")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video at {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height} at {fps} FPS, {total_frames} frames")
    
    # Create video writer if output path is specified
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    else:
        out = None
    
    # Create a named window in fullscreen mode if showing video
    if show_video:
        window_name = "Detection Results"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Process the video frame by frame
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Increment frame counter
        frame_count += 1
        
        # Calculate and display progress
        if frame_count % 10 == 0:
            elapsed_time = time.time() - start_time
            frames_left = total_frames - frame_count
            if elapsed_time > 0 and frames_left > 0:
                estimated_time = (elapsed_time / frame_count) * frames_left
                print(f"Processing: {frame_count}/{total_frames} frames " +
                      f"({frame_count/total_frames*100:.1f}%) - " +
                      f"ETA: {estimated_time:.1f}s", end='\r')
        
        # Run inference on the frame
        results = model.predict(
            source=frame,
            conf=conf_threshold,
            save=False,
            verbose=False
        )
        
        # Plot results on the frame
        result_frame = results[0].plot()
        
        # Write frame to output video
        if out:
            out.write(result_frame)
        
        # Display the frame in fullscreen window
        if show_video:
            cv2.imshow(window_name, result_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release resources
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    total_time = time.time() - start_time
    fps_processing = frame_count / total_time
    print(f"\nProcessed {frame_count} frames in {total_time:.2f} seconds ({fps_processing:.2f} FPS)")
    
    if output_path:
        print(f"Output video saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Test YOLOv8 model on videos")
    parser.add_argument("--model", default="Biological E PPE Detection/runs/ppe_detection/weights/best.pt", 
                        help="Path to the trained model")
    parser.add_argument("--video", required=True, 
                        help="Path to the video file to test")
    parser.add_argument("--conf", type=float, default=0.25, 
                        help="Confidence threshold for detections")
    parser.add_argument("--output", 
                        help="Path to save the output video (optional)")
    parser.add_argument("--no-display", action="store_true", 
                        help="Do not display video during processing")
    
    args = parser.parse_args()
    
    try:
        # Load the model
        model = load_model(args.model)
        
        # Process the video
        process_video(
            model, 
            args.video, 
            args.output, 
            args.conf, 
            not args.no_display
        )
        
        print("Video processing completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 