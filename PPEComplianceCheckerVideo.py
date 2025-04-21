import os
import cv2
import argparse
import time
import numpy as np
from ultralytics import YOLO
from datetime import datetime

# # Define required PPE items and non-compliant indicators
# REQUIRED_PPE = {
#     'surgical-mask': False,     # Whether person is wearing a mask
#     'surgical-gloves': False,   # Whether person is wearing gloves
#     'surgical-gown': False,     # Whether person is wearing a gown
#     'surgical-cap': False,      # Whether person is wearing a cap
#     'face-shield': False        # Whether person is wearing a face shield
# }

# Define required PPE items and non-compliant indicators
REQUIRED_PPE = {
    'boot_secured': False,      # Whether boots are secured
    'gloves_fitted_securely': False,  # Whether gloves are fitted securely
    'headgear_secured': False,  # Whether headgear is secured
    'zip_secured': False        # Whether zip is secured
}

# Non-compliant indicators (presence of these means non-compliance)
NON_COMPLIANT_INDICATORS = [
    'boot_not_secured',
    'gloves_not_fitted_securely',
    'gown_torn',
    'headgear_not_secured_backside',
    'zip_not_secured'
]

# Required minimum number of PPE items (adjust as needed)
MIN_REQUIRED_PPE = 4  # boots, gloves, headgear, zip

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

def check_ppe_compliance(results):
    """
    Check if the detected person is compliant with PPE requirements.
    
    Args:
        results: Detection results from YOLO model
        
    Returns:
        is_compliant: Boolean indicating whether person is compliant
        compliance_details: Dictionary with details about compliance
    """
    # Reset PPE status
    ppe_status = REQUIRED_PPE.copy()
    for key in ppe_status:
        ppe_status[key] = False
    
    # Count detected PPE items
    detected_classes = []
    non_compliant_detected = False
    
    if hasattr(results, 'boxes') and len(results.boxes) > 0:
        # Get class names
        class_names = results.names if hasattr(results, 'names') else {}
        
        # Check each detection
        for box in results.boxes:
            cls_id = int(box.cls.item())
            cls_name = class_names.get(cls_id, f"Class {cls_id}")
            conf = box.conf.item()
            
            detected_classes.append(cls_name)
            
            # Update PPE status
            if cls_name in ppe_status:
                ppe_status[cls_name] = True
            
            # Check if non-compliant indicator is detected
            if cls_name in NON_COMPLIANT_INDICATORS:
                non_compliant_detected = True
    
    # Calculate number of PPE items worn
    num_ppe_worn = sum(1 for status in ppe_status.values() if status)
    
    # Determine overall compliance
    is_compliant = (num_ppe_worn >= MIN_REQUIRED_PPE) and not non_compliant_detected
    
    # Create compliance details
    compliance_details = {
        'ppe_status': ppe_status,
        'num_ppe_worn': num_ppe_worn,
        'min_required': MIN_REQUIRED_PPE,
        'non_compliant_indicator_detected': non_compliant_detected,
        'detected_classes': detected_classes
    }
    
    return is_compliant, compliance_details

def process_video(model, video_path, output_path=None, conf_threshold=0.25, show_video=True, playback_speed=1.0):
    """
    Process a video with the trained model and check PPE compliance.
    
    Args:
        model: Loaded YOLO model
        video_path: Path to the video file
        output_path: Path to save the output video (optional)
        conf_threshold: Confidence threshold for detections
        show_video: Whether to display video during processing
        playback_speed: Speed multiplier for video playback (1.0 = normal speed)
        
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
    
    # Calculate delay between frames based on playback speed
    if show_video:
        frame_delay = int(1000 / (fps * playback_speed))
        frame_delay = max(1, frame_delay)  # Ensure delay is at least 1ms
    else:
        frame_delay = 1
    
    # Create video writer if output path is specified
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    else:
        out = None
    
    # Create a named window in fullscreen mode if showing video
    if show_video:
        window_name = "PPE Compliance (Press 'q' to quit, 'p' to pause/resume, 'f' to toggle fullscreen)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Process the video frame by frame
    frame_count = 0
    start_time = time.time()
    paused = False
    
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Increment frame counter
            frame_count += 1
        
        # Calculate and display progress
        if frame_count % 10 == 0 and not paused:
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
        
        # Check PPE compliance
        is_compliant, compliance_details = check_ppe_compliance(results[0])
        
        # Plot results on the frame
        result_frame = results[0].plot()
        
        # Add compliance status to the frame with a background
        status_text = "ALLOWED" if is_compliant else "NOT ALLOWED"
        status_color = (0, 255, 0) if is_compliant else (0, 0, 255)  # Green for allowed, Red for not allowed
        
        # Create background for status text
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
        text_x = (width - text_size[0]) // 2
        text_y = height - 50
        
        # Draw background rectangle
        cv2.rectangle(
            result_frame,
            (text_x - 10, text_y - text_size[1] - 10),
            (text_x + text_size[0] + 10, text_y + 10),
            (0, 0, 0),
            -1
        )
        
        # Draw status text
        cv2.putText(
            result_frame,
            status_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            status_color,
            3
        )
        
        # Add PPE status details on the right side
        y_offset = 80
        for ppe, status in compliance_details['ppe_status'].items():
            status_text = f"{ppe}: {'✓' if status else '✗'}"
            color = (0, 255, 0) if status else (0, 0, 255)
            
            cv2.putText(
                result_frame,
                status_text,
                (width - 350, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )
            y_offset += 40
        
        # Add frame counter
        cv2.putText(
            result_frame,
            f"Frame: {frame_count}/{total_frames}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        
        # Add pause indicator if paused
        if paused:
            cv2.putText(
                result_frame,
                "PAUSED",
                (width - 150, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 165, 255),
                2
            )
        
        # Write frame to output video
        if out:
            out.write(result_frame)
        
        # Display the frame
        if show_video:
            cv2.imshow(window_name, result_frame)
            
            # Wait for key press
            key = cv2.waitKey(0 if paused else frame_delay) & 0xFF
            
            # 'q' to quit
            if key == ord('q'):
                break
                
            # 'p' to pause/resume
            elif key == ord('p'):
                paused = not paused
                print("Video playback paused" if paused else "Video playback resumed")
                
            # 'f' to toggle fullscreen
            elif key == ord('f'):
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    
            # '+' to speed up playback
            elif key == ord('+') or key == ord('='):
                playback_speed *= 1.5
                frame_delay = int(1000 / (fps * playback_speed))
                frame_delay = max(1, frame_delay)
                print(f"Playback speed: {playback_speed:.1f}x")
                
            # '-' to slow down playback
            elif key == ord('-'):
                playback_speed /= 1.5
                frame_delay = int(1000 / (fps * playback_speed))
                print(f"Playback speed: {playback_speed:.1f}x")
    
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
    # Global declaration at the beginning of the function
    global MIN_REQUIRED_PPE
    
    parser = argparse.ArgumentParser(description="PPE Compliance Checker for Videos using YOLOv8 model")
    parser.add_argument("--model", default="PPE Detection/runs/ppe_detection/weights/best.pt", 
                        help="Path to the trained model")
    parser.add_argument("--video", required=True, 
                        help="Path to the video file to process")
    parser.add_argument("--conf", type=float, default=0.25, 
                        help="Confidence threshold for detections")
    parser.add_argument("--output", 
                        help="Path to save the output video (optional)")
    parser.add_argument("--no-display", action="store_true", 
                        help="Do not display video during processing")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (default: 1.0)")
    parser.add_argument("--min-ppe", type=int, default=MIN_REQUIRED_PPE,
                        help=f"Minimum number of PPE items required (default: {MIN_REQUIRED_PPE})")
    
    args = parser.parse_args()
    
    # Update minimum required PPE if specified
    MIN_REQUIRED_PPE = args.min_ppe
    
    try:
        # Load the model
        model = load_model(args.model)
        
        # Process the video
        process_video(
            model, 
            args.video, 
            args.output, 
            args.conf, 
            not args.no_display,
            args.speed
        )
        
        print("PPE compliance checking on video completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 