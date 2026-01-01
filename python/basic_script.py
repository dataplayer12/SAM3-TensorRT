from transformers.models.sam3 import Sam3Processor, Sam3Model
import torch
from PIL import Image, ImageDraw, ImageFont
import requests
import pdb
import os
from typing import Union, Sequence, Dict, Any
import numpy as np
from visualize import visualize_sam3_results

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

# Load image
image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
# image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
image = Image.open("boxes.jpg").convert("RGB")

# Segment using text prompt
inputs = processor(images=image, text="box", return_tensors="pt").to(device)

print(inputs.keys())

with torch.no_grad():
    outputs = model(**inputs)

# Post-process results
results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)[0]


# Example usage:
# visualize_sam3_results(image, results, out_path="sam3_viz.png")

# pdb.set_trace()
visualize_sam3_results(
    image,
    results,
    out_path = 'results.jpg',
    mask_alpha = 0.45,
    mask_threshold = 0.5,
    box_width = 3,
)

print(f"Found {len(results['masks'])} objects")
# Results contain:
# - masks: Binary masks resized to original image size
# - boxes: Bounding boxes in absolute pixel coordinates (xyxy format)
# - scores: Confidence scores
