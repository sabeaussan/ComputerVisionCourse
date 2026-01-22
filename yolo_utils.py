import numpy as np
from ultralytics.utils.nms import TorchNMS
import torch

def compute_inference_time(result):
    inference_time = np.sum(list(result.speed.values()))
    fps = 1000.0 / inference_time if inference_time > 0 else 0.0
    desc = f"Inference took: {inference_time:.1f} ms, running at {fps:.1f} FPS"
    print(desc)
    
def filter_NMS(result, iou_threshold=0.4):
	bboxes = result.boxes.xyxy  # Bounding boxes
	scores = result.boxes.conf  # Confidence scores
	if len(bboxes) == 0:
		return result
	keep = TorchNMS.fast_nms(bboxes, scores, iou_threshold=iou_threshold)
	kept_bboxes = bboxes[keep] if len(bboxes) else bboxes
	kept_scores = scores[keep] if len(scores) else scores
	class_ = torch.zeros_like(kept_scores, dtype=torch.int64)  # Assuming single class "crack"

	filtered_boxes = torch.cat([kept_bboxes, kept_scores.unsqueeze(1), class_.unsqueeze(1)], dim=1)
	filtered_masks = result.masks.data[keep] if len(keep) else result.masks.data
	result.update(boxes=filtered_boxes, masks=filtered_masks)