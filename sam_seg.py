from ultralytics.models.sam import SAM3SemanticPredictor
import os
import torch
from plot_utils import plot_binary_mask
from yolo_utils import compute_inference_time, filter_NMS

DATA_DIR = os.path.join("datasets", "classification")
nms = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Initialize predictor with configuration
overrides = dict(
    conf=0.3,
    task="segment",
    mode="predict",
    model="models/sam3.pt",
    half=device.type == "cuda",  # Use FP16 for faster inference
    save=False,
	imgsz=200
)
predictor = SAM3SemanticPredictor(overrides=overrides) 

# Set image once for multiple queries
path = os.path.join(os.path.join(DATA_DIR, "val"), "positive", "1 (201).jpg")
predictor.set_image(path)  # Example with the first test image

# Query with text prompt
result = predictor(text=["crack"])[0]
compute_inference_time(result)

if len(result.boxes) == 0:
    exit()
    
result.show()
mask = torch.sum(result.masks.data, dim=0) > 0  # Combine masks if multiple found
plot_binary_mask(mask)

# optional NMS filtering
if nms:
	filter_NMS(result, iou_threshold=0.4)
	result.show()






