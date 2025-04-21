import os
import cv2
import argparse
import time
import numpy as np
from ultralytics import YOLO
from datetime import datetime

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

def process_webcam(model, camera_id=0, output_path=None, conf_threshold=0.25):
    """
    Process webcam feed with the trained model and check PPE compliance.
    
    Args:
        model: Loaded YOLO model
        camera_id: Webcam ID (0 for default camera)
        output_path: Path to save the output video (optional)
        conf_threshold: Confidence threshold for detections
    """
    # Open the webcam
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise ValueError(f"Failed to open webcam with ID {camera_id}")
    
    # Try to set higher resolution for the webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30  # Assume 30 FPS for webcam
    
    print(f"Webcam properties: {width}x{height} at {fps} FPS")
    
    # Create video writer if output path is specified
    if output_path:
        # If output_path doesn't have an extension, add timestamp
        if not os.path.splitext(output_path)[1]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{output_path}_{timestamp}.mp4"
            
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Recording to {output_path}")
    else:
        out = None
    
    # Create a named window in fullscreen mode
    window_name = "PPE Compliance Checker (Press 'q' to quit, 'r' to start/stop recording, 'f' to toggle fullscreen)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    fps_display_time = start_time
    fps_display_value = 0
    
    # Variable for recording toggle
    is_recording = output_path is not None
    
    print("PPE compliance checking started. Press 'q' to quit, 'r' to start/stop recording, 'f' to toggle fullscreen.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam")
            break
        
        # Increment frame counter
        frame_count += 1
        
        # Calculate FPS every second
        current_time = time.time()
        if current_time - fps_display_time >= 1.0:
            fps_display_value = frame_count / (current_time - start_time)
            fps_display_time = current_time
        
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
        
        # Add FPS information to the frame
        cv2.putText(
            result_frame, 
            f"FPS: {fps_display_value:.1f}", 
            (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
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
        
        # Add recording indicator if recording
        if is_recording:
            cv2.putText(
                result_frame, 
                "REC", 
                (width - 100, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 255), 
                2
            )
            
            # Write frame to output video if recording
            if out:
                out.write(result_frame)
        
        # Display the frame
        cv2.imshow(window_name, result_frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        
        # 'q' to quit
        if key == ord('q'):
            break
            
        # 'r' to toggle recording
        elif key == ord('r'):
            is_recording = not is_recording
            
            if is_recording and out is None:
                # Start new recording
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_output_path = f"PPE Detection/recordings/compliance_{timestamp}.mp4"
                os.makedirs(os.path.dirname(os.path.abspath(new_output_path)), exist_ok=True)
                out = cv2.VideoWriter(new_output_path, fourcc, fps, (width, height))
                print(f"Started recording to {new_output_path}")
            elif not is_recording and out is not None:
                # Stop recording
                out.release()
                out = None
                print("Stopped recording")
            
        # 'f' to toggle fullscreen
        elif key == ord('f'):
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Release resources
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    total_time = time.time() - start_time
    fps_processing = frame_count / total_time
    print(f"Processed {frame_count} frames in {total_time:.2f} seconds ({fps_processing:.2f} FPS)")

def main():
    # Move the global declaration to the beginning of the function
    global MIN_REQUIRED_PPE
    
    parser = argparse.ArgumentParser(description="PPE Compliance Checker using YOLOv8 model on webcam feed")
    parser.add_argument("--model", default="PPE Detection/runs/ppe_detection/weights/best.pt", 
                        help="Path to the trained model")
    parser.add_argument("--camera", type=int, default=0, 
                        help="Camera ID (usually 0 for the default webcam)")
    parser.add_argument("--conf", type=float, default=0.25, 
                        help="Confidence threshold for detections")
    parser.add_argument("--output", 
                        help="Path to save the output video (optional)")
    parser.add_argument("--min-ppe", type=int, default=MIN_REQUIRED_PPE,
                        help=f"Minimum number of PPE items required (default: {MIN_REQUIRED_PPE})")
    
    args = parser.parse_args()
    
    # Update minimum required PPE if specified
    MIN_REQUIRED_PPE = args.min_ppe
    
    try:
        # Load the model
        model = load_model(args.model)
        
        # Process the webcam feed
        process_webcam(
            model, 
            args.camera, 
            args.output, 
            args.conf
        )
        
        print("PPE compliance checking completed")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 