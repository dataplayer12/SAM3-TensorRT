# SAM3 → TensorRT

Export Meta AI's Segment Anything 3 (SAM3) model to ONNX, then build a TensorRT engine for real-time segmentation. This repo includes a CUDA inference library and demo apps for semantic and instance segmentation.

## Table of Contents
- [Project Overview](#project-overview)
- [Benchmarks](#benchmarks)
- [Demos](#demos)
- [Repo Layout](#repo-layout)
- [Quickstart](#quickstart)
  - [On x86](#on-x86)
  - [On Jetson/Spark](#on-jetsonspark)
- [Extensions](#extensions)
- [Troubleshooting](#troubleshooting)
- [Development guide](#development-guide)
  - [CUDA Library Notes](#cuda-library-notes)
  - [Benchmarking](#benchmarking)
  - [ONNX Export Details](#onnx-export-details)
  - [TensorRT Notes](#tensorrt-notes)
  - [License](#license)

## Project Overview
- Python tooling to export SAM3 to a clean ONNX graph.
- TensorRT-ready workflows for building optimized engines.
- A C++/CUDA library for high-performance inference with demo apps.
- Support for Promptable concept segmentation (PCS), the latest feautre in SAM3.
- Zero-copy support on unified-memory platforms (Jetson, DGX Spark). Great for robotics/real-time interaction.
- Everything runs inside a reproducible docker environment (x86, Jetson, Spark).
- MIT license for the love of everything nice :)

## Benchmarks
Benchmarks from this project. The numbers show end to end image processing latency excluding image load/save time.

| Hardware | HF+PyTorch | TensorRT+CUDA | Speedup | Notes |
| --- | --- | --- | --- | --- |
| RTX 3090 | 437.76 ms/image | 75.17 ms/image | 5.82x |  |
| Jetson Orin NX | 6600 ms/image | 950 ms/image | 6.95x |  |
| DGX Spark | TBD | TBD | TBD | TBD |
| Jetson Nano | TBD | TBD | TBD | TBD |

Please contribute your results and I will be happy to add them here. Use [Benchmarking](#benchmarking) to run the benchmarks yourself.

## Demos
Video demo (click to play):
[![Semantic segmentation demo video](https://img.youtube.com/vi/hHvhQ514Evs/maxresdefault.jpg)](https://youtube.com/shorts/hHvhQ514Evs?feature=share)

Semantic segmentation produced by the C++ demo app (`prompt='dog'`)

<img src="demo/semantic_puppies.png" width="640" alt="Semantic segmentation demo">

Instance segmentation results (`prompt='box'`)

<img src="demo/instance_box.jpeg" width="800" alt="Instance segmentation demo">

## Repo Layout
- `python/` - ONNX export and visualization scripts.
- `cpp/` - C++/CUDA library and apps (TensorRT inference).
- `docker/` - Container setup (`Dockerfile.x86`, with an aarch64 variant expected).
- `demo/` - Example outputs from the C++ demo app.

## Quickstart

1) Request access to the gated model
   - Visit https://huggingface.co/facebook/sam3 and request access.
   - Ensure your `HF_TOKEN` has permission.
   - Set `HF_TOKEN` as environment variable in the host. Docker will pick it up from there.

2) Build the Docker container for your platform (all commands below run inside it)

### On x86
```bash
docker build -t sam3-trt -f docker/Dockerfile.x86 .
```

### On Jetson/Spark

For aarch64 platforms with shared CPU/GPU memory, the C++ library in this repo supports zero-copy inference paths.

Build and run the aarch64 container:
```bash
docker build -t sam3-trt-aarch64 -f docker/Dockerfile.aarch64 .
```

3) Export `HF_TOKEN` and run the docker container 

```bash
export HF_TOKEN=<YOUR TOKEN>
docker run -it --rm \
  --network=host \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --runtime=nvidia \
  --env HF_TOKEN \
  -v "$PWD":/workspace \
  -w /workspace \
  sam3-trt bash
```

4) Export to ONNX
```bash
python python/onnxexport.py
```
This produces `onnx_weights/sam3_static.onnx` plus external weight shards.

5) Build a TensorRT engine
```bash
trtexec --onnx=onnx_weights/sam3_static.onnx --saveEngine=sam3_fp16.plan --fp16 --verbose
```

6) Build the C++/CUDA library and sample app
```bash
mkdir cpp/build && cd cpp/build
cmake ..
make
```

7) Run the demo app
```bash
./sam3_pcs_app <image_dir> <engine_path.engine>
```

Results are written to a `results/` folder.


## Extensions
This is a very raw project and provides the crucial backend TensorRT/CUDA bits necessary for anything. From here, please feel free to fan out into any application you like. Pull requests are very welcome! Here are some ideas I can think of:
- ROS2 wrapper for real-time robotics pipelines.
- Interactive voice-based segmentation app. Have someone speak into a microphone, use a TTS model to transcribe it and feed into the engine, which then produces the segmentation mask live. I don't have the time to build it but I hope you can.
- Live camera input and overlays. You will need a beefy GPU. SAM3 doesn't run realtime on a Jetson nano.

## Troubleshooting
- **Access errors:** Make sure your `HF_TOKEN` has access to `facebook/sam3`.
- **ONNX export fails:** Install `transformers` from source if SAM3 is missing.
- **TensorRT parse errors:** Ensure the full `onnx_weights/` directory is copied (external data is required).
- **C++ build errors:** Confirm CUDA, TensorRT, and OpenCV are installed and discoverable via `pkg-config`.

## Development guide

### CUDA Library Notes
- The shared library target is `sam3_trt`.
- Demo app: `sam3_pcs_app` (semantic/instance visualization modes).
- Outputs include semantic segmentation and instance segmentation mask logits. If you choose `SAM3_VISUALIZATION::VIS_NONE` in your application, you need to apply sigmoid yourself.
- The library does not support building engines. Use `trtexec` instead.

### Benchmarking
Use the same image directory and prompt for all runs. Both paths time the model pipeline and exclude image load/save.

Huggingface + PyTorch:
```bash
python python/basic_script.py <image_dir>
```

TensorRT + CUDA (benchmark mode disables output writes):
```bash
./sam3_pcs_app <image_dir> <engine_path.engine> 1
```

### ONNX Export Details
- Default export runs on CPU for compatibility (switch `device` to `cuda` if desired).
- SAM3 is large and exports with external weight shards; keep the entire `onnx_weights/` directory together.

### TensorRT Notes
- Use `trtexec` for quick engine builds and benchmarking.
- FP16 is the usual starting point; INT8/FP8/INT4 require calibration or compatible tooling.

### License
- MIT (see `LICENSE`).

If this saved you time, drop a ⭐ so others can find it and ship SAM-3 faster.
