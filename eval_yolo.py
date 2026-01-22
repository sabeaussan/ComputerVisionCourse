from ultralytics import YOLO
from yolo_utils import compute_inference_time
import torch

# Load a model
# Using a pretrained model like yolo26n-seg.pt is recommended for faster convergence
model = YOLO("runs/segment/train/weights/best.pt")

# Evaluate the model on the Crack Segmentation dataset
metrics = model.val(
	data="datasets/segmentation/crack_sgementation.yaml", 
	imgsz=200, 
	device="cpu",
	iou = 0.5
)

print("Mean average precision for different IoU thresholds:", metrics.box.map)
print("Mean recall:", metrics.box.mr)


results = model.predict(
	source="datasets/segmentation/images/train", 
	imgsz=200, 
	device="cpu"
)

def check_crack_state(results, threshold=0.05):
	n_maitenance = 0
	for result in results:
		if len(result.boxes) != 0:
			crack_area = torch.sum(result.masks.data, dim=0) > 0
			crack_ratio = torch.sum(crack_area).item() / crack_area.numel()
			print(crack_ratio)
			if crack_ratio > threshold:
				n_maitenance += 1
				print(f"Large crack detected, triggering maintenance for image {result.path} (crack ratio: {crack_ratio:.2%})")
	print(f"Total images requiring maintenance: {n_maitenance} out of {len(results)}")

check_crack_state(results, threshold=0.05)