import os
import cv2
import numpy as np
import argparse
from ultralytics import YOLO
from PIL import Image

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

def process_image(model, image_path, conf_threshold=0.25):
    """
    Process an image with the trained model.
    
    Args:
        model: Loaded YOLO model
        image_path: Path to the image to process
        conf_threshold: Confidence threshold for detections
        
    Returns:
        Original image and results from the model
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image at {image_path}")
    
    # Run inference
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=False,
        verbose=False
    )
    
    return image, results[0]

def display_results(image, results, output_path=None):
    """
    Display the results on the image and optionally save it.
    
    Args:
        image: Original image
        results: Detection results from the model
        output_path: Path to save the output image (optional)
    """
    # Get the plotted image with detections from results
    result_image = results.plot()
    
    # Convert from BGR to RGB for display
    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    
    # Display detection info
    print("\n--- Detection Results ---")
    if len(results.boxes) == 0:
        print("No objects detected.")
    else:
        print(f"Detected {len(results.boxes)} objects:")
        
        # Get class names if available
        class_names = results.names if hasattr(results, 'names') else None
        
        # Process each detection
        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls.item())
            cls_name = class_names[cls_id] if class_names else f"Class {cls_id}"
            conf = box.conf.item()
            bbox = box.xyxy[0].tolist()  # x1, y1, x2, y2
            
            print(f"  {i+1}. {cls_name}: {conf:.2f} confidence, Bbox: {[round(x, 2) for x in bbox]}")
    
    # Save result if output path is provided
    if output_path:
        cv2.imwrite(output_path, result_image)
        print(f"Result saved to {output_path}")
    
    # Display the image
    cv2.imshow("Detection Results", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Test YOLOv8 model on images")
    parser.add_argument("--model", default="Biological E PPE Detection/runs/ppe_detection/weights/best.pt", 
                        help="Path to the trained model")
    parser.add_argument("--image", required=True, 
                        help="Path to the image to test")
    parser.add_argument("--conf", type=float, default=0.25, 
                        help="Confidence threshold for detections")
    parser.add_argument("--output", 
                        help="Path to save the output image (optional)")
    
    args = parser.parse_args()
    
    try:
        # Load the model
        model = load_model(args.model)
        
        # Process the image
        image, results = process_image(model, args.image, args.conf)
        
        # Display results
        display_results(image, results, args.output)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 