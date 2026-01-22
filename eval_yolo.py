from ultralytics import YOLO
from yolo_utils import compute_inference_time
import torch
import os

# Load a model
# Using a pretrained model like yolo26n-seg.pt is recommended for faster convergence
MODELS_PATH = os.path.join("runs", "segment", "train", "weights", "best.pt")
DATASET_PATH = os.path.join("datasets", "segmentation")
DATASET_YAML = os.path.join(DATASET_PATH, "crack_sgementation.yaml")
DATASET_TEST = os.path.join(DATASET_PATH, "images", "test")
model = YOLO(MODELS_PATH)

# Evaluate the model on the Crack Segmentation dataset
metrics = model.val(
	data=DATASET_YAML, 
	imgsz=200, 
	device="cpu",
	iou = 0.5
)

print("Mean average precision for different IoU thresholds:", metrics.box.map)
print("Mean recall:", metrics.box.mr)


results = model.predict(
	source=DATASET_TEST, 
	imgsz=200, 
	device="cpu"
)

def check_crack_state(results, threshold=0.05):
	"""
		Analyze YOLO segmentation results to determine if maintenance is required based on crack area ratio.
		The threshold parameter defines the minimum ratio of crack area to total image area to trigger maintenance.
			- results: List of YOLO prediction results
			- threshold: float, ratio threshold to trigger maintenance
	"""
	n_maitenance = 0
	for result in results:
		if len(result.boxes) != 0:
			crack_area = torch.sum(result.masks.data, dim=0) > 0
			crack_ratio = torch.sum(crack_area).item() / crack_area.numel()
			if crack_ratio > threshold:
				n_maitenance += 1
				print(f"Large crack detected, triggering maintenance for image {result.path} (crack ratio: {crack_ratio:.2%})")
	print(f"Total images requiring maintenance: {n_maitenance} out of {len(results)}")

check_crack_state(results, threshold=0.05)