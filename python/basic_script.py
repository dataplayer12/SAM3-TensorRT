from transformers.models.sam3 import Sam3Processor, Sam3Model
import torch
from PIL import Image, ImageDraw, ImageFont
import requests
import pdb
import os
from typing import Union, Sequence, Dict, Any
import numpy as np
from visualize import visualize_sam3_results
import time
import sys

device = "cuda" if torch.cuda.is_available() else "cpu"
prompt='snow'
benchmark = True

model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")
indir = sys.argv[1]

num_frames_read=0
millis_elapsed=0.0

for fname in os.listdir(indir):
    image = Image.open(os.path.join(indir, fname)).convert("RGB")

    start=time.time()
    # Segment using text prompt
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]

    visualize_sam3_results(
        image,
        results,
        out_path = 'results.jpg',
        mask_alpha = 0.45,
        mask_threshold = 0.5,
        box_width = 3,
        benchmark=benchmark
    )
    end = time.time()
    num_frames_read+=1
    millis_elapsed+= (end-start)*1000
    if num_frames_read%10==0 and num_frames_read>0:
        print(f"Processed {num_frames_read} images at {millis_elapsed/num_frames_read :.2f} msec/image")