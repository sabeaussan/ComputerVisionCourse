from PIL import Image
from pathlib import Path
from transformers import AutoImageProcessor
import numpy as np
import os
from dino_backbone import DINO_ID

class ReadJpgDataset:

	def __init__(self, data_path, model_id="v2-small", device="cpu"):
		model_id = DINO_ID.get(model_id, "v2-small")
		self.data_path = data_path
		# Used to preprocess images for DINO model
		self.processor = AutoImageProcessor.from_pretrained(model_id)
		self.device = device
		self.images, self.labels = self._build_data()
		if self.labels:
			self.labels = np.array(self.labels)
		
	def _list_images(self):
		root_path = Path(self.data_path)
		self.paths = [p for p in root_path.rglob("*") if p.suffix==".jpg"]
	
	def _build_data(self):
		""" 
			Build dataset by loading images and their corresponding labels from directory structure. 
				- returns: Tuple (images_data, labels)
				- images_data: Tensor of preprocessed images
				- labels: List of integer labels (1 for positive, 0 for negative)
		"""
		self._list_images()
		labels = []
		images = []
		for p in self.paths:
			s = p.parts[-2]
			if s == "positive":
				labels.append(1)
			elif s == "negative":
				labels.append(0)
			else:
				pass
			img = Image.open(p).convert("RGB")
			images.append(img)
		# preprocess images (normalize, resize, etc.)
		images_data = self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
		return images_data, labels
	
	def __len__(self):
		return len(self.images)

def save_class_predictions(output_dir, test_set, predictions):
	pos_dir = os.path.join(output_dir, "positive")
	neg_dir = os.path.join(output_dir, "negative")
	os.makedirs(pos_dir, exist_ok=True)
	os.makedirs(neg_dir, exist_ok=True)
	for i, pred in enumerate(predictions):
		img_path = test_set.paths[i]
		img = Image.open(img_path)
		pred_label = "positive" if pred == 1 else "negative"
		output_path = os.path.join(output_dir, pred_label, f"{img_path.parts[-1]}")
		img.save(output_path)