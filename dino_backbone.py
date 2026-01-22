from transformers import AutoModel
import torch
import os

MODEL_DIR = "models"
DINO_ID = {
	"v2-small": os.path.join(MODEL_DIR, "dinov2-small", "snapshots", "ed25f3a31f01632728cabb09d1542f84ab7b0056"),
	"v3-small": os.path.join(MODEL_DIR, "dinov3-vits16-pretrain-lvd1689m", "snapshots", "114c1379950215c8b35dfcd4e90a5c251dde0d32")
}

class DINOBackbone:
	def __init__(self, device, dtype, model_id="v2-small"):
		model_id = DINO_ID.get(model_id, "v2-small")
		self.model = AutoModel.from_pretrained(
			model_id,
			torch_dtype=dtype,
			attn_implementation="sdpa",
		).eval().to(device)

	def extract_features(self, inputs):
		""" Extract features from input images using the DINO model.
			- inputs: Tensor of shape (batch_size, 3, H, W)
			- returns: Tensor of shape (batch_size, feature_dim)"""
		with torch.no_grad():
			outputs = self.model(pixel_values=inputs)
		# CLS token features
		features = outputs.last_hidden_state[:, 0, :].cpu()
		return features