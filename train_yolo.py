from ultralytics import YOLO
import os

# Load a model
# Using a pretrained model like yolo26n-seg.pt is recommended for faster convergence
model = YOLO(
	os.path.join("models", "yolo26s-seg.pt"),
	task="segment"
)

# Train the model on the Crack Segmentation dataset
DATASET_PATH = os.path.join("datasets", "segmentation")
DATASET_YAML = os.path.join(DATASET_PATH, "crack_sgementation.yaml")
results = model.train(data=DATASET_YAML, epochs=50, imgsz=200, device="cuda")