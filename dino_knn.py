import os
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score
from sklearn.neighbors import KNeighborsClassifier
from io_utils import ReadJpgDataset, save_class_predictions
from dino_backbone import DINOBackbone
from plot_utils import plot_pca

def compute_classication_metrics(true_labels, predicted_labels):
	"""
		Compute and print classification metrics: accuracy, confusion matrix, recall, precision.
			- true_labels: List or array of true class labels
			- predicted_labels: List or array of predicted class labels
	"""
	accuracy = accuracy_score(true_labels, predicted_labels)
	conf_mat = confusion_matrix(true_labels, predicted_labels)
	recall = recall_score(true_labels, predicted_labels)
	precision = precision_score(true_labels, predicted_labels)
	print("Confusion Matrix:")
	print(conf_mat)
	print(f"Recall: {recall*100:.2f}%") # Number of missed positive samples, should be low in our case
	print(f"Precision: {precision*100:.2f}%")
	print(f"KNN classification accuracy: {accuracy*100:.2f}%")
	

# -----------------------
# Config
# -----------------------
DATA_DIR = os.path.join("datasets", "classification")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32


train_root = os.path.join(DATA_DIR, "train")
val_root = os.path.join(DATA_DIR, "val")
test_root = os.path.join(DATA_DIR, "test")

train_dataset = ReadJpgDataset(train_root, device=device)
val_dataset = ReadJpgDataset(val_root, device=device)
test_dataset = ReadJpgDataset(test_root, device=device)


model = DINOBackbone(
	device=device, 
	dtype=dtype, 
	model_id="v2-small"
) # Instantiate DINO backbone

with torch.inference_mode():
	train_features = model.extract_features(train_dataset.images)
	val_features = model.extract_features(val_dataset.images)
	test_features = model.extract_features(test_dataset.images)
	
	# Normalize
	train_features = torch.nn.functional.normalize(train_features, p=2, dim=1)
	val_features = torch.nn.functional.normalize(val_features, p=2, dim=1)
	test_features = torch.nn.functional.normalize(test_features, p=2, dim=1)


# Train and validate KNN classifier
k_neighbors = 3
classifier = KNeighborsClassifier(
	n_neighbors=k_neighbors,
	metric="cosine",
	weights="distance",
	algorithm="brute",
)

classifier.fit(train_features.numpy(), train_dataset.labels)
preds = classifier.predict(val_features.numpy())
compute_classication_metrics(val_dataset.labels, preds)

# Test KNN classifier on unlabeled test set
test_preds = classifier.predict(test_features.numpy())

# Save predictions
output_dir = os.path.join(DATA_DIR, "knn_predictions")
save_class_predictions(output_dir, test_dataset, test_preds)

# PCA visualization for train/val with positive/negative labels
pca_output_path = os.path.join(DATA_DIR, "pca_train_val.png")
plot_pca(
	train_features,
	train_dataset.labels,
	val_features,
	val_dataset.labels,
	output_path=pca_output_path,
)
