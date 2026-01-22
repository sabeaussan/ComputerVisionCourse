import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


def _to_numpy(array):
	if isinstance(array, np.ndarray):
		return array
	try:
		return array.detach().cpu().numpy()
	except AttributeError:
		return np.asarray(array)


def plot_pca(
	train_features,
	train_labels,
	val_features,
	val_labels,
	output_path=None,
	show=False,
	title="DINO PCA (train/val)",
):
	train_features = _to_numpy(train_features)
	val_features = _to_numpy(val_features)
	train_labels = _to_numpy(train_labels)
	val_labels = _to_numpy(val_labels)

	all_features = np.concatenate([train_features, val_features], axis=0)
	pca = PCA(n_components=2, random_state=0)
	components = pca.fit_transform(all_features)

	n_train = train_features.shape[0]
	train_components = components[:n_train]
	val_components = components[n_train:]

	fig, ax = plt.subplots(figsize=(8, 6))

	groups = [
		("train", "positive", train_components, train_labels == 1, "o", "tab:red"),
		("train", "negative", train_components, train_labels == 0, "o", "tab:blue"),
		("val", "positive", val_components, val_labels == 1, "^", "tab:red"),
		("val", "negative", val_components, val_labels == 0, "^", "tab:blue"),
	]

	for split_name, class_name, comps, mask, marker, color in groups:
		if not np.any(mask):
			continue
		ax.scatter(
			comps[mask, 0],
			comps[mask, 1],
			label=f"{split_name} {class_name}",
			alpha=0.7,
			s=18,
			marker=marker,
			color=color,
			edgecolors="none",
		)

	explained = pca.explained_variance_ratio_
	ax.set_xlabel(f"PC1")
	ax.set_ylabel(f"PC2")
	ax.set_title(title)
	ax.legend(frameon=False, fontsize=9)
	ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

	fig.tight_layout()
	if output_path:
		os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
		fig.savefig(output_path, dpi=150)
	if show:
		plt.show()
	plt.close(fig)


def plot_binary_mask(mask, output_path=None, show=True, title="Binary Mask"):
	mask_np = _to_numpy(mask)
	if mask_np.ndim > 2:
		mask_np = np.squeeze(mask_np)
	if mask_np.ndim != 2:
		raise ValueError(f"Expected 2D mask after squeeze, got shape {mask_np.shape}")

	fig, ax = plt.subplots(figsize=(6, 6))
	ax.imshow(mask_np.astype(float), cmap="gray", vmin=0.0, vmax=1.0)
	ax.set_title(title)
	ax.axis("off")

	fig.tight_layout()
	if output_path:
		os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
		fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0)
	if show:
		plt.show()
	plt.close(fig)
