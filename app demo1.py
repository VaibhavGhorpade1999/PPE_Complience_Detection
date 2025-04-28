import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from PIL import Image
from ultralytics import YOLO
from pathlib import Path

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

# Define which PPE items are only visible from certain angles
ANGLE_DEPENDENT_ITEMS = {
    'zip_secured': True,  # Only visible from front
    'headgear_secured_backside': True,  # Only visible from back
    'boot_secured': False,  # Visible from multiple angles
    'gloves_fitted_securely': False  # Visible from multiple angles
}

# Define for how many frames we should remember detections
PERSISTENCE_FRAMES = 30  # Remember detections for ~3 seconds at 10 fps

# Function to load model (only once)
@st.cache_resource
def load_model(model_path):
    """Load a trained YOLOv8 model."""
    model = YOLO(model_path)
    return model

def check_ppe_compliance(results, item_memory=None):
    """
    Check if the detected person is compliant with PPE requirements.
    
    Args:
        results: Detection results from YOLO model
        item_memory: Dictionary tracking persistent detections for angle-dependent items
        
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
            
            detected_classes.append(cls_name)
            
            # Update PPE status
            if cls_name in ppe_status:
                ppe_status[cls_name] = True
            
            # Check if non-compliant indicator is detected
            if cls_name in NON_COMPLIANT_INDICATORS:
                non_compliant_detected = True
    
    # Apply memory for angle-dependent items
    if item_memory is not None:
        for item, frames_left in item_memory.items():
            if frames_left > 0:
                # If we remembered this item as detected recently, consider it detected
                ppe_status[item] = True
    
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

def display_compliance_metrics(compliance_details, is_compliant):
    """Display the compliance metrics in Streamlit UI"""
    st.subheader("Compliance Details")
    
    # Create a container for the overall status
    status_container = st.container()
    if is_compliant:
        status_container.success("✅ ALLOWED - All PPE requirements met")
    else:
        status_container.error("❌ NOT ALLOWED - PPE requirements not met")
    
    # Display detected PPE items
    st.write("PPE Status:")
    for ppe, status in compliance_details['ppe_status'].items():
        if status:
            st.success(f"✅ {ppe}")
        else:
            st.error(f"❌ {ppe}")
    
    # Display any non-compliant indicators
    if compliance_details['non_compliant_indicator_detected']:
        st.error("Non-compliant items detected")
        if 'detected_classes' in compliance_details:
            non_compliant_items = [cls for cls in compliance_details['detected_classes'] 
                                  if cls in NON_COMPLIANT_INDICATORS]
            for item in non_compliant_items:
                st.error(f"❌ {item}")

def process_image(model, image, conf_threshold=0.25):
    """Process an image and check PPE compliance"""
    # Convert PIL Image to RGB and then to numpy array
    if isinstance(image, Image.Image):
        image = image.convert('RGB')
        image_np = np.array(image)
    else:
        if len(image.shape) == 3 and image.shape[2] == 4:
            image_np = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            image_np = image
    
    # Run inference
    results = model.predict(
        source=image_np,
        conf=conf_threshold,
        save=False,
        verbose=False
    )
    
    # Check compliance
    is_compliant, compliance_details = check_ppe_compliance(results[0])
    
    # Create a copy of the image for drawing
    result_image = image_np.copy()
    
    # Draw detections with custom colors
    if hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            cls_name = results[0].names[cls_id]
            
            # Determine color based on class type
            if cls_name in NON_COMPLIANT_INDICATORS:
                color = (255, 0, 0)  # Red for non-compliant indicators
            else:
                color = (0, 255, 0)  # Green for compliant PPE
            
            # Draw rectangle
            cv2.rectangle(result_image, 
                         (int(x1), int(y1)), 
                         (int(x2), int(y2)), 
                         color, 2)
            
            # Add label without confidence
            label = cls_name  # Removed confidence value
            font_size = 0.8  # Increased font size (was 0.5)
            font_thickness = 2  # Increased thickness for better visibility
            
            (label_width, label_height), _ = cv2.getTextSize(label, 
                                                           cv2.FONT_HERSHEY_SIMPLEX, 
                                                           font_size, font_thickness)
            
            # Draw label background
            cv2.rectangle(result_image, 
                         (int(x1), int(y1) - label_height - 10),  # Adjusted padding
                         (int(x1) + label_width, int(y1)), 
                         color, -1)
            
            # Draw label text with black color
            cv2.putText(result_image, label,
                       (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), font_thickness)
    
    return result_image, is_compliant, compliance_details

def image_compliance_checker():
    """UI for checking compliance on uploaded images"""
    # Use less vertical space for the header
    st.write("### Image PPE Compliance Checker")
    
    # Create a more efficient layout with all elements in one row
    col1, col2, col3 = st.columns([1, 1.5, 1.5])
    
    # File uploader in first column
    with col1:
        uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        process_button = st.button("Check Compliance", use_container_width=True)
        
        # Create a placeholder for status information
        status_placeholder = st.empty()
    
    # Column for original image    
    with col2:
        original_placeholder = st.empty()
    
    # Column for result image
    with col3:
        result_placeholder = st.empty()
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        
        # Calculate aspect ratio and set a fixed height
        img_width, img_height = image.size
        aspect_ratio = img_width / img_height
        display_height = 800  # Fixed display height
        
        # Display original image with fixed height
        original_placeholder.markdown("**Original Image**")
        original_placeholder.image(image, width=int(display_height * aspect_ratio))
        
        # Process and display result if button clicked
        if process_button:
            with st.spinner("Processing..."):
                # Process image
                result_image, is_compliant, compliance_details = process_image(
                    model, image, 0.25)
                
                # Display result image with the same fixed height
                result_placeholder.markdown("**Detection Result**")
                result_placeholder.image(result_image, width=int(display_height * aspect_ratio))
                
                # Create a compact status display
                with status_placeholder.container():
                    # Overall status
                    st.markdown(f"**Status:** {'✅ ALLOWED' if is_compliant else '❌ NOT ALLOWED'}")
                    
                    # Display PPE items vertically (one below the other)
                    for ppe, status in compliance_details['ppe_status'].items():
                        icon = "✅" if status else "❌"
                        ppe_name = ppe.replace('_', ' ').title()
                        st.markdown(f"{icon} {ppe_name}")
                    
                    # Show non-compliant indicators if any
                    if compliance_details['non_compliant_indicator_detected']:
                        non_compliant_items = [cls for cls in compliance_details['detected_classes'] 
                                              if cls in NON_COMPLIANT_INDICATORS]
                        if non_compliant_items:
                            st.error(f"Non-compliant items: {', '.join(non_compliant_items)}")
    else:
        # Display placeholders when no image is uploaded
        original_placeholder.markdown("**Original Image will appear here**")
        result_placeholder.markdown("**Detection Result will appear here**")

def process_video_frame(frame, model, conf_threshold=0.25, item_memory=None):
    """Process a single video frame and check PPE compliance"""
    # Ensure frame is in RGB format (3 channels)
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    
    # Run inference
    results = model.predict(
        source=frame,
        conf=conf_threshold,
        save=False,
        verbose=False
    )
    
    # Check compliance
    is_compliant, compliance_details = check_ppe_compliance(results[0], item_memory)
    
    # Create a copy of the frame for drawing
    result_frame = frame.copy()
    
    # Draw detections with custom colors
    if hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            cls_name = results[0].names[cls_id]
            
            # Determine color based on class type
            if cls_name in NON_COMPLIANT_INDICATORS:
                color = (0, 0, 255)  # Red for non-compliant indicators (BGR format)
            else:
                color = (0, 255, 0)  # Green for compliant PPE
            
            # Draw rectangle
            cv2.rectangle(result_frame, 
                         (int(x1), int(y1)), 
                         (int(x2), int(y2)), 
                         color, 2)
            
            # Add label without confidence
            label = cls_name  # Removed confidence value
            font_size = 0.8  # Increased font size (was 0.5)
            font_thickness = 2  # Increased thickness for better visibility
            
            (label_width, label_height), _ = cv2.getTextSize(label, 
                                                           cv2.FONT_HERSHEY_SIMPLEX, 
                                                           font_size, font_thickness)
            
            # Draw label background
            cv2.rectangle(result_frame, 
                         (int(x1), int(y1) - label_height - 10),  # Adjusted padding
                         (int(x1) + label_width, int(y1)), 
                         color, -1)
            
            # Draw label text with black color
            cv2.putText(result_frame, label,
                       (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), font_thickness)
    
    # Add compliance status to the frame
    status_text = "ALLOWED" if is_compliant else "NOT ALLOWED"
    status_color = (0, 255, 0) if is_compliant else (255, 0, 0)
    
    # Create background for status text
    text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = frame.shape[0] - 50
    
    # Draw background rectangle and status text
    cv2.rectangle(result_frame,
                 (text_x - 10, text_y - text_size[1] - 10),
                 (text_x + text_size[0] + 10, text_y + 10),
                 (0, 0, 0), -1)
    cv2.putText(result_frame, status_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 2, status_color, 3)
    
    return result_frame, is_compliant, compliance_details

def video_compliance_checker():
    """UI for checking compliance on uploaded videos"""
    # Use less vertical space for the header
    st.write("### Video PPE Compliance Checker")
    
    # Create a two-column layout (similar to image compliance)
    upload_col, video_col = st.columns([1, 3])
    
    # Upload controls in first column
    with upload_col:
        uploaded_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov"], label_visibility="collapsed")
        
        # Container for video info
        info_container = st.container()
        
        # Add process button
        process_button = st.button("Process Video", use_container_width=True)
        
        # Progress indicators
        progress_bar = st.empty()
        status_text = st.empty()
        
        # Create a placeholder for status information (will show after processing)
        status_placeholder = st.empty()
    
    # Video display in larger column
    with video_col:
        video_placeholder = st.empty()
    
    if uploaded_file is not None:
        # Save the uploaded file to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        temp_video_path = tfile.name
        tfile.close()
        
        # Open the video file
        cap = cv2.VideoCapture(temp_video_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        with info_container:
            st.info(f"Video: {width}x{height}, {fps} FPS", icon="ℹ️")
        
        # Show a preview frame
        ret, preview_frame = cap.read()
        if ret:
            # Reset video position
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Convert BGR to RGB for display
            preview_rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
            
            # Calculate aspect ratio for display
            aspect_ratio = width / height
            display_height = 700  # Fixed display height
            display_width = int(display_height * aspect_ratio)
            
            # Display preview frame
            video_placeholder.image(preview_rgb, width=display_width, caption="Preview - Click Process to analyze")
        
        if process_button:
            # Variables to track compliance statistics
            compliant_frames = 0
            non_compliant_frames = 0
            processed_frames = 0
            compliance_summary = {}
            
            # Track non-compliant indicators separately
            non_compliant_indicators_count = {indicator: 0 for indicator in NON_COMPLIANT_INDICATORS}
            
            # Add memory for angle-dependent items
            item_memory = {item: 0 for item in REQUIRED_PPE if ANGLE_DEPENDENT_ITEMS.get(item, False)}
            
            # Process the video
            frame_count = 0
            start_time = time.time()
            
            # Sample frames for processing (to improve performance)
            frame_step = max(1, int(fps / 5))  # Process ~5 frames per second
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Update progress
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing: {frame_count}/{total_frames}")
                
                # Process only every n-th frame
                if frame_count % frame_step == 0:
                    processed_frames += 1
                    
                    # Process the frame
                    result_frame, is_compliant, compliance_details = process_video_frame(
                        frame, model, 0.75, item_memory)
                    
                    # Update memory for angle-dependent items
                    for item, is_detected in compliance_details['ppe_status'].items():
                        if item in item_memory:
                            if is_detected:
                                # If detected, set memory to max persistence
                                item_memory[item] = PERSISTENCE_FRAMES
                            elif item_memory[item] > 0:
                                # If not detected but in memory, decrement memory
                                item_memory[item] -= 1
                    
                    # Update compliance stats
                    if is_compliant:
                        compliant_frames += 1
                    else:
                        non_compliant_frames += 1
                    
                    # Track compliance details for summary
                    for ppe, status in compliance_details['ppe_status'].items():
                        if ppe not in compliance_summary:
                            compliance_summary[ppe] = {'compliant': 0, 'non_compliant': 0}
                        
                        if status:
                            compliance_summary[ppe]['compliant'] += 1
                        else:
                            compliance_summary[ppe]['non_compliant'] += 1
                    
                    # Track non-compliant indicators
                    for cls in compliance_details['detected_classes']:
                        if cls in NON_COMPLIANT_INDICATORS:
                            non_compliant_indicators_count[cls] += 1
                    
                    # Convert BGR to RGB
                    result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                    
                    # Display the frame
                    video_placeholder.image(result_frame_rgb, width=display_width)
            
            # Clean up
            cap.release()
            os.unlink(temp_video_path)
            
            # Final status
            elapsed_time = time.time() - start_time
            status_text.text(f"Completed in {elapsed_time:.1f}s")
            
            # Determine overall compliance
            compliance_percentage = (compliant_frames / processed_frames * 100) if processed_frames > 0 else 0
            overall_compliant = compliance_percentage >= 65  # Consider compliant if 35% or more frames are compliant
            
            # Calculate PPE status based on majority of frames
            final_ppe_status = {}
            for ppe, stats in compliance_summary.items():
                total = stats['compliant'] + stats['non_compliant']
                if total > 0:
                    final_ppe_status[ppe] = (stats['compliant'] / total >= 0.5)  # Considered compliant if >50% frames show compliance
            
            # Display final status similar to image compliance checker
            with status_placeholder.container():
                # Overall status
                st.markdown(f"**Status:** {'✅ ALLOWED' if overall_compliant else '❌ NOT ALLOWED'}")
                
                # Display PPE items vertically (one below the other)
                for ppe, status in final_ppe_status.items():
                    icon = "✅" if status else "❌"
                    ppe_name = ppe.replace('_', ' ').title()
                    st.markdown(f"{icon} {ppe_name}")
                
                # Show non-compliant indicators that were detected in frames
                detected_non_compliant = [
                    indicator for indicator, count in non_compliant_indicators_count.items() 
                    if count > 0
                ]
                
                if detected_non_compliant:
                    # Format the indicators for display
                    formatted_indicators = [indicator.replace('_', ' ').title() for indicator in detected_non_compliant]
                    st.error(f"Non-compliant items detected: {', '.join(formatted_indicators)}")
                
                # Add compliance rate
                #st.info(f"Overall compliance rate: {compliance_percentage:.1f}%")

def webcam_compliance_checker():
    """UI for checking compliance using webcam"""
    st.subheader("Webcam PPE Compliance Checker")
    
    # Create a two-column layout for controls and webcam display
    control_col, webcam_col = st.columns([1, 2])
    
    with control_col:
        # Confidence threshold slider
        conf_threshold = st.slider("Confidence", 0.1, 1.0, 0.25, 0.05)
        
        # Initialize session state variables for webcam control
        if 'webcam_running' not in st.session_state:
            st.session_state.webcam_running = False
        if 'webcam_results' not in st.session_state:
            st.session_state.webcam_results = None
        
        # Functions to control webcam state
        def start_webcam():
            # Try to initialize the webcam
            try:
                camera = cv2.VideoCapture(0)
                if not camera or not camera.isOpened():
                    st.error("Failed to initialize webcam. Please check your camera connection.")
                    return
                
                # If successful, store in session state
                st.session_state.cap = camera
                st.session_state.webcam_running = True
                
                # Initialize other session variables
                st.session_state.item_memory = {item: 0 for item in REQUIRED_PPE if ANGLE_DEPENDENT_ITEMS.get(item, False)}
                st.session_state.compliant_frames = 0
                st.session_state.non_compliant_frames = 0
                st.session_state.total_frames = 0
                st.session_state.compliance_summary = {}
                st.session_state.start_time = time.time()
            except Exception as e:
                st.error(f"Error initializing webcam: {e}")
        
        def stop_webcam():
            if 'cap' in st.session_state and st.session_state.cap is not None:
                try:
                    st.session_state.cap.release()
                except:
                    pass  # Ignore errors during release
                st.session_state.cap = None
            st.session_state.webcam_running = False
            
        # Start/Stop buttons
        if not st.session_state.webcam_running:
            st.button("Start Webcam", on_click=start_webcam, key="start_button")
        else:
            st.button("Stop Webcam", on_click=stop_webcam, key="stop_button")
        
        # Create a container for status
        status_container = st.container()
    
    # Webcam display area in the larger column
    with webcam_col:
        # Create a placeholder for the webcam feed
        webcam_placeholder = st.empty()
    
    # Run webcam if it should be running
    if st.session_state.webcam_running and 'cap' in st.session_state and st.session_state.cap is not None:
        cap = st.session_state.cap
        
        # Verify the cap is still valid
        try:
            is_opened = cap.isOpened()
        except Exception:
            is_opened = False
            
        if not is_opened:
            with control_col:
                st.error("Webcam connection lost. Please try again.")
                stop_webcam()  # Call the stop function to clean up
        else:
            # Try to set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Capture a single frame
            ret, frame = cap.read()
            
            if ret:
                st.session_state.total_frames += 1
                
                # Process the frame
                result_frame, is_compliant, compliance_details = process_video_frame(
                    frame, model, conf_threshold, st.session_state.item_memory)
                
                # Update memory for angle-dependent items
                for item, is_detected in compliance_details['ppe_status'].items():
                    if item in st.session_state.item_memory:
                        if is_detected:
                            # If detected, set memory to max persistence
                            st.session_state.item_memory[item] = PERSISTENCE_FRAMES
                        elif st.session_state.item_memory[item] > 0:
                            # If not detected but in memory, decrement memory
                            st.session_state.item_memory[item] -= 1
                
                # Update compliance stats
                if is_compliant:
                    st.session_state.compliant_frames += 1
                else:
                    st.session_state.non_compliant_frames += 1
                
                # Track compliance details for summary
                for ppe, status in compliance_details['ppe_status'].items():
                    if ppe not in st.session_state.compliance_summary:
                        st.session_state.compliance_summary[ppe] = {'compliant': 0, 'non_compliant': 0}
                    
                    if status:
                        st.session_state.compliance_summary[ppe]['compliant'] += 1
                    else:
                        st.session_state.compliance_summary[ppe]['non_compliant'] += 1
                
                # Convert BGR to RGB
                result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                
                # Display the frame
                webcam_placeholder.image(result_frame_rgb, use_container_width=True)
                
                # Update status in the control column
                with status_container:
                    # Show current status
                    if is_compliant:
                        st.success("✅ ALLOWED")
                    else:
                        st.error("❌ NOT ALLOWED")
                    
                    # Show compliance details (just a summary to save space)
                    status_cols = st.columns(len(compliance_details['ppe_status']))
                    for i, (ppe, status) in enumerate(compliance_details['ppe_status'].items()):
                        with status_cols[i]:
                            if status:
                                st.success(f"✅ {ppe}")
                            else:
                                st.error(f"❌ {ppe}")
                
                # Store current results in session state for reuse
                st.session_state.webcam_results = {
                    "is_compliant": is_compliant,
                    "compliance_details": compliance_details
                }
            else:
                st.error("Failed to capture frame from webcam")
                stop_webcam()
    else:
        # Clean up webcam resources when stopped
        if 'cap' in st.session_state and st.session_state.cap is not None:
            try:
                st.session_state.cap.release()
            except:
                pass
            st.session_state.cap = None
            
            # Show final statistics if any frames were processed
            if 'total_frames' in st.session_state and st.session_state.total_frames > 0:
                with status_container:
                    elapsed_time = time.time() - st.session_state.start_time
                    
                    st.markdown("### Session Summary")
                    st.write(f"✅ Compliant frames: {st.session_state.compliant_frames} ({st.session_state.compliant_frames/st.session_state.total_frames*100:.1f}%)")
                    st.write(f"❌ Non-compliant frames: {st.session_state.non_compliant_frames} ({st.session_state.non_compliant_frames/st.session_state.total_frames*100:.1f}%)")
                    st.write(f"⏱️ Session duration: {elapsed_time:.2f} seconds")
                    
                    # Add explanation about angle-dependent items
                    st.info("""
                    **Note on viewing angle:** 
                    Some PPE items like a secured zip are only visible from certain angles. 
                    Once detected, these items are considered present for a short time even if the 
                    person turns and the item is no longer visible in the camera.
                    """)
                    
                    # Calculate overall compliance
                    compliance_percentage = (st.session_state.compliant_frames / st.session_state.total_frames * 100)
                    
                    if compliance_percentage >= 40:
                        st.success(f"Overall: ALLOWED ({compliance_percentage:.1f}% compliance, threshold: 40%)")
                        st.write("Reason: Sufficient frames showed proper PPE compliance")
                    else:
                        st.error(f"Overall: NOT ALLOWED ({compliance_percentage:.1f}% compliance, threshold: 40%)")
                        st.write("Reason: Insufficient frames showed proper PPE compliance")

def main():
    st.set_page_config(
        page_title="PPE Compliance Checker",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Create a layout with columns for logo and title
    logo_col, title_col = st.columns([1, 4])
    
    # Display logo in the first column (left side)
    with logo_col:
        # Replace 'path_to_your_logo.png' with the actual path to your logo file
        logo_path = "image.png"
        try:
            st.image(logo_path, width=250)  # Adjust width as needed
        except Exception as e:
            st.error(f"Error loading logo: {e}")
    
    # Display title in the second column
    with title_col:
        st.title("PPE Compliance Checker")
        st.markdown("""
        This app checks compliance with Personal Protective Equipment (PPE) requirements.
        Select an option from the sidebar to get started.
        """)
    
    # Sidebar for navigation
    st.sidebar.title("PPE Compliance Checker")
    # app_mode = st.sidebar.selectbox(
    #     "Choose the mode",
    #     ["Image Compliance Checker", "Video Compliance Checker", "Webcam Compliance Checker"]
    # )
    app_mode = st.sidebar.selectbox(
        "Select Mode",
        ["Image Compliance Checker", "Video Compliance Checker"]
    )
    
    # Model selection
    # model_path = st.sidebar.text_input(
    #     "Model Path", 
    #     "runs/ppe_detection/weights/best.pt"
    # )

    
    # Load the model
    global model
    try:
        model = load_model("runs/ppe_detection/weights/best.pt")
        #st.sidebar.success(f"Model loaded successfully: {model_path}")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        # st.error(f"Failed to load model from {model_path}. Please check the path and try again.")
        return
    
    # Display the selected mode
    if app_mode == "Image Compliance Checker":
        image_compliance_checker()
    elif app_mode == "Video Compliance Checker":
        video_compliance_checker()
    elif app_mode == "Webcam Compliance Checker":
        webcam_compliance_checker()

if __name__ == "__main__":
    main() 