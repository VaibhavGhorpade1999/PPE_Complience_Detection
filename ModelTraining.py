import os
import yaml
from ultralytics import YOLO
import shutil
from pathlib import Path

# Define paths
DATASET_PATH = r"D:\Biological E POC\Biological E PPE Detection\MPPE Dataset"
OUTPUT_PATH = r"Biological E PPE Detection\runs"
os.makedirs(OUTPUT_PATH, exist_ok=True)

def create_dataset_yaml():
    """Create a YAML file to define the dataset structure for YOLOv8."""
    
    # Get the class names from the data.yaml if it exists in the dataset directory
    dataset_yaml_path = os.path.join(DATASET_PATH, "data.yaml")
    
    if os.path.exists(dataset_yaml_path):
        print(f"Using existing data.yaml from {dataset_yaml_path}")
        return dataset_yaml_path
    
    # If no data.yaml exists, create one
    train_path = os.path.join(DATASET_PATH, "train")
    val_path = os.path.join(DATASET_PATH, "valid")
    test_path = os.path.join(DATASET_PATH, "test")
    
    # Try to detect class names from labels in train folder
    class_names = []
    try:
        # Get one label file to extract class names
        label_files = [f for f in os.listdir(os.path.join(train_path, "labels")) if f.endswith(".txt")]
        if label_files:
            sample_label = os.path.join(train_path, "labels", label_files[0])
            with open(sample_label, 'r') as f:
                lines = f.readlines()
            
            # Get unique class IDs from the label file
            unique_classes = set()
            for line in lines:
                class_id = int(line.strip().split()[0])
                unique_classes.add(class_id)
            
            # Create placeholder class names (user should edit these later)
            class_names = [f"class_{i}" for i in range(max(unique_classes) + 1)]
            print(f"Detected {len(class_names)} classes. Please edit the data.yaml file to add proper class names.")
        else:
            print("No label files found. Please manually edit the data.yaml file.")
            class_names = ["class_0"]  # Default placeholder
    except Exception as e:
        print(f"Error detecting classes: {e}")
        print("Using default class name. Please edit the data.yaml file.")
        class_names = ["class_0"]  # Default placeholder
    
    # Create the dataset YAML
    custom_dataset_yaml = os.path.join(OUTPUT_PATH, "dataset.yaml")
    dataset_config = {
        'path': DATASET_PATH,
        'train': os.path.join(DATASET_PATH, "train", "images"),
        'val': os.path.join(DATASET_PATH, "valid", "images"),
        'test': os.path.join(DATASET_PATH, "test", "images"),
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    with open(custom_dataset_yaml, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"Created dataset YAML at {custom_dataset_yaml}")
    print("Please review and edit the class names in this file if needed.")
    
    return custom_dataset_yaml

def train_model(dataset_yaml, pretrained_model=r"D:\Biological E POC\Yolo-Weights\yolov8n.pt"):
    """Train YOLOv8 model on the dataset."""
    
    # Load a pretrained YOLOv8 model
    model = YOLO(pretrained_model)
    
    # Set training parameters
    results = model.train(
        data=dataset_yaml,
        epochs=32,  # You may want to adjust this based on your needs
        imgsz=640,
        patience=20,  # Early stopping patience
        batch=8,     # Adjust based on your GPU memory
        save=True,
        device='cpu',   # Use '0' for first GPU, 'cpu' for CPU, '0,1' for multiple GPUs
        project=OUTPUT_PATH,
        name="ppe_detection",
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        verbose=True,
        seed=42
    )
    
    return results

def validate_model(model_path):
    """Validate the trained model on the validation set."""
    model = YOLO(model_path)
    
    # Run validation
    results = model.val(
        data=dataset_yaml,
        split="val"
    )
    
    print(f"Validation results: {results}")
    return results

def export_model(model_path, format="onnx"):
    """Export the model to different formats for deployment."""
    model = YOLO(model_path)
    
    # Export the model
    model.export(format=format)
    print(f"Model exported to {format} format")

if __name__ == "__main__":
    # 1. Create dataset YAML
    dataset_yaml = create_dataset_yaml()
    
    # 2. Train the model
    print("Starting model training...")
    results = train_model(dataset_yaml)
    
    # 3. Get the path to the best model
    best_model_path = os.path.join(OUTPUT_PATH, "ppe_detection", "weights", "best.pt")
    if os.path.exists(best_model_path):
        print(f"Best model saved at: {best_model_path}")
        
        # 4. Validate the model
        print("Validating the model...")
        val_results = validate_model(best_model_path)
        
        # 5. Export the model for inference
        print("Exporting the model...")
        export_model(best_model_path)
        
        print("Training pipeline completed successfully!")
    else:
        print("Training may have failed. Check the logs for details.")
