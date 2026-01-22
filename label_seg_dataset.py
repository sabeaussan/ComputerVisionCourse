import os
from pathlib import Path
import shutil

import torch
from PIL import Image
from tqdm import tqdm
from ultralytics.models.sam import SAM3SemanticPredictor
from ultralytics.engine.results import Masks
import numpy as np

# def _save_mask(masks, combined_mask_tensor, output_path):
# 	mask_np = combined_mask_tensor.detach().cpu().numpy().astype("uint8") * 255
# 	Image.fromarray(mask_np).save(output_path)
# 	mask_yolo = Masks(masks=mask_tensor.unsqueeze(0), orig_shape=mask_tensor.shape)
# 	class_ = 0
# 	label = np.concatenate(([class_], mask_yolo.xyn[0].reshape(-1)), axis=0).reshape(-1, 1)
# 	np.savetxt(
# 		str(output_path.with_suffix(".txt")),
# 		label,
# 		fmt="%.6f",
# 	)

def _save_mask(masks, combined_mask_tensor, output_path):
	mask_np = combined_mask_tensor.detach().cpu().numpy().astype("uint8") * 255
	Image.fromarray(mask_np).save(output_path)

	mask_yolo = Masks(masks=combined_mask_tensor.unsqueeze(0), orig_shape=combined_mask_tensor.shape)
	# mask_yolo = Masks(masks=masks, orig_shape=combined_mask_tensor.shape)
	class_ = 0
	lines = []
	for polygon in mask_yolo.xyn:
		coords = polygon.reshape(-1)
		if coords.size < 4:
			continue
		if not np.isfinite(coords).all():
			continue
		line = np.concatenate(([class_], coords), axis=0)
		lines.append(" ".join(f"{v:.6f}" for v in line))

	label_path = output_path.with_suffix(".txt")
	with open(label_path, "w", encoding="utf-8") as f:
		f.write("\n".join(lines))
	


def _segment_directory(predictor, images_dir, labels_dir):
	images_dir = Path(images_dir)
	labels_dir = Path(labels_dir)
	labels_dir.mkdir(parents=True, exist_ok=True)

	image_paths = [p for p in images_dir.rglob("*.jpg")]
	for image_path in tqdm(image_paths, desc=f"Segmenting {images_dir}", unit="img"):
		predictor.set_image(str(image_path))
		results = predictor(text=["crack"])
		if not results or results[0].masks is None:
			continue
		masks = results[0].masks.data
		if masks.numel() == 0:
			continue
		combined = torch.sum(masks, dim=0) > 0
		output_path = labels_dir / f"{image_path.stem}.png"
		_save_mask(masks, combined, output_path)

	# Remove images without corresponding masks
	for image_path in image_paths:
		label_path = labels_dir / f"{image_path.stem}.txt"
		if not label_path.exists():
			image_path.unlink()


def _copy_positive_dir(source_dir, target_dir):
	if not os.path.exists(source_dir):
		return False
	if os.path.exists(target_dir):
		return True
	os.makedirs(os.path.dirname(target_dir), exist_ok=True)
	shutil.copytree(source_dir, target_dir)
	return True


def segment_positive_instances():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	overrides = dict(
		conf=0.2,
		task="segment",
		mode="predict",
		model="models/sam3.pt",
		half=device.type == "cuda",
		save=False,
		imgsz=200,
	)
	predictor = SAM3SemanticPredictor(overrides=overrides)

	split_map = [
		("knn_predictions", "train"),
		("train", "test"),
		("val", "val"),
	]

	segmentation_roots = []
	for source_split, target_split in split_map:
		source_dir = os.path.join("datasets", "classification", source_split, "positive")
		target_dir = os.path.join("datasets", "segmentation", "images", target_split)
		if _copy_positive_dir(source_dir, target_dir):
			segmentation_roots.append(
				(
					target_dir,
					os.path.join("datasets", "segmentation", "labels", target_split),
				)
			)

	for images_dir, labels_dir in segmentation_roots:
		if os.path.exists(images_dir):
			_segment_directory(predictor, images_dir, labels_dir)


if __name__ == "__main__":
	segment_positive_instances()
