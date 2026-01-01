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

prompt="dog"

# 2. Build a sample batch (same as your example)
image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

pixel_values = inputs["pixel_values"]
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
print(input_ids.shape)

# 3. Wrap Sam3Model so the ONNX graph has clean inputs/outputs
class Sam3ONNXWrapper(torch.nn.Module):
    def __init__(self, sam3, input_ids, attention_mask):
        super().__init__()
        self.sam3 = sam3
        self.register_buffer("const_input_ids", input_ids.to(torch.int64).cpu())
        self.register_buffer("const_attention_mask", attention_mask.to(torch.int64).cpu())

    def forward(self, pixel_values):
        outputs = self.sam3(
            pixel_values=pixel_values,
            input_ids=self.const_input_ids,
            attention_mask=self.const_attention_mask,
        )
        # Typical useful outputs
        instance_masks = torch.sigmoid(outputs.pred_masks)  # [B, Q, H, W]
        semantic_seg = outputs.semantic_seg                 # [B, 1, H, W]
        return outputs.pred_masks, outputs.semantic_seg # instance_masks, semantic_seg

wrapper = Sam3ONNXWrapper(model, input_ids, attention_mask).to(device).eval()

# 5. Export to ONNX
output_dir = Path(f"onnx_weights")
output_dir.mkdir(exist_ok=True)
onnx_path = str(output_dir / f"sam3_static.onnx")

torch.onnx.export(
    wrapper,
    (pixel_values),
    onnx_path,
    input_names=["pixel_values"],
    output_names=["instance_masks", "semantic_seg"],
    dynamo=False,
    opset_version=17,
)
print(f"Exported to {onnx_path}")
