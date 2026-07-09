import torch
from pathlib import Path
from transformers.models.sam3 import Sam3Processor, Sam3Model
from PIL import Image
import requests

device = "cpu" # for onnx export we use CPU for maximum compatibility

# 1. Load model & processor
model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

model.eval()

prompt="person"

# 2. Build a sample batch (same as your example)
image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

pixel_values = inputs["pixel_values"]
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

print("input_ids", input_ids.shape, input_ids.dtype)
print(input_ids)
print()
print("attention_mask", attention_mask.shape, attention_mask.dtype)
print(attention_mask)

# 3. Wrap Sam3Model so the ONNX graph has clean inputs/outputs
class Sam3ONNXWrapper(torch.nn.Module):
    def __init__(self, sam3):
        super().__init__()
        self.sam3 = sam3

    def forward(self, pixel_values, input_ids, attention_mask):
        outputs = self.sam3(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask)
        
        return outputs.pred_masks, outputs.semantic_seg, outputs.pred_boxes, outputs.pred_logits

wrapper = Sam3ONNXWrapper(model).to(device).eval()

# 5. Export to ONNX
output_dir = Path(f"onnx_weights")
output_dir.mkdir(exist_ok=True)
onnx_path = str(output_dir / f"sam3_dynamic.onnx")

torch.onnx.export(
    wrapper,
    (pixel_values, input_ids, attention_mask),
    onnx_path,
    input_names=["pixel_values", "input_ids", "attention_mask"],
    output_names=["instance_masks", "semantic_seg", "pred_boxes", "pred_logits"],
    dynamo=False,
    opset_version=17,
)
print(f"Exported to {onnx_path}")
