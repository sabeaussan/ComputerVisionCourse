from ultralytics import YOLO

# Load a model
# Using a pretrained model like yolo26n-seg.pt is recommended for faster convergence
model = YOLO(
	"yolo26s-seg.pt",
	task="segment"
)

# Train the model on the Crack Segmentation dataset
results = model.train(data="datasets/segmentation/crack_sgementation.yaml", epochs=50, imgsz=200, device="cuda")