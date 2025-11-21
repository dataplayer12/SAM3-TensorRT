# SAM3 → TensorRT

Export Meta AI's newest Segment Anything 3 (SAM-3) model to ONNX and then build a TensorRT engine you can deploy for real-time segmentation.

## Why this repo
- Minimal, readable export script using Hugging Face `transformers`' `Sam3Model` and `Sam3Processor`
- Clean ONNX graph with instance masks (`pred_masks`) and semantic mask outputs
- TensorRT-ready: plug the produced ONNX into `trtexec` or your own build pipeline
- CPU-friendly export for maximum compatibility (flip `device` to use GPU if you prefer)

## Quickstart
1) **Request access to the gated model**
   - Visit https://huggingface.co/facebook/sam3 and click “Access repository” (approval is required before downloads succeed). Make sure your `HF_TOKEN` has that access.

2) **Install dependencies**
```bash
pip install torch transformers pillow requests

# Optional: If after a pip install the ONNX export script complains that it cannot find SAM3, it's just because SAM-3 is very new and a formal release of `transformers` has not added support yet. In this case, please install `transformers` using git

git clone https://github.com/huggingface/transformers.git
cd transformers
pip install '.[torch]'
# or uv pip install '.[torch]'
```
Use a PyTorch wheel that matches your CUDA if you plan to export on GPU. By default a GPU is not required to export to ONNX.


3) **Export to ONNX**
```bash
export HF_TOKEN=<YOUR TOKEN>
python onnxexport.py
```

This downloads `facebook/sam3`, runs a sample prompt (`"ear"`), and writes `onnx_weights/sam3_static.onnx` plus associated weight shards. SAM-3 is ~3.2 GB, so the ONNX exporter will create external data files; if you build TensorRT on another machine you must copy the entire `onnx_weights/` directory (not just the .onnx).


4) **Build a TensorRT engine**
Use `trtexec` (or your favorite builder) on the generated ONNX:

```bash
trtexec --onnx=onnx_weights/sam3_static.onnx --saveEngine=sam3_fp16.plan --fp16 --verbose # fp16
trtexec --onnx=onnx_weights/sam3_static.onnx --saveEngine=sam3_int8.plan --int8 --verbose # int8
trtexec --onnx=onnx_weights/sam3_static.onnx --saveEngine=sam3_fp8.plan --fp8 --verbose # fp8
trtexec --onnx=onnx_weights/sam3_static.onnx --saveEngine=sam3_int4.plan --int4 --verbose # int4
```


5) **Validate** (optional)
- Run `onnxruntime` on `onnx_weights/sam3_static.onnx` to confirm outputs
- Benchmark `sam3_fp16.plan` with `trtexec --loadEngine=sam3_fp16.plan`

## Adapting the export
- Change the prompt/image in `onnxexport.py` to better reflect your production use case.
- Swap `device` to `"cuda"` for GPU-side export if your environment supports it.
- Add more outputs from `Sam3Model` (e.g., scores) by extending `Sam3ONNXWrapper`.

## Project layout
- `onnxexport.py` — single, documented script that pulls SAM-3, builds clean inputs, and exports ONNX
- `LICENSE` — MIT

If this saved you time, drop a ⭐ so others can find it and ship SAM-3 faster.
