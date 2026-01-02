import os
from typing import Union, Sequence, Dict, Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

def visualize_sam3_results(
    image: Image.Image,
    results: Dict[str, Any],
    out_path: str,
    mask_alpha: float = 0.45,
    mask_threshold: float = 0.5,
    box_width: int = 3,
    benchmark: bool = False,
) -> None:
    """
    Visualize SAM3 instance segmentation results and write to disk.

    Args:
        image: PIL image (RGB recommended).
        results: dict with keys: 'scores', 'boxes', 'masks' (as returned by
                 processor.post_process_instance_segmentation(...)[0]).
        out_path: output filepath (e.g., "out.png" or "out.jpg").
        mask_alpha: transparency of mask overlay (0..1).
        mask_threshold: threshold to binarize mask if mask is float.
        box_width: rectangle line width in pixels.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # --- normalize inputs to CPU numpy ---
    img_rgba = image.convert("RGBA")
    W, H = img_rgba.size

    # in benchmarking mode we only visualize segmentation masks 
    # to stay on par with the CUDA implementation
    scores_t = [] if benchmark else results["scores"]
    boxes_t = [] if benchmark else results.get("boxes",[])
    masks_obj = results["masks"]

    if isinstance(scores_t, torch.Tensor):
        scores = scores_t.detach().float().cpu().numpy()
    else:
        scores = np.asarray(scores_t, dtype=np.float32)

    if isinstance(boxes_t, torch.Tensor):
        boxes = boxes_t.detach().float().cpu().numpy()
    else:
        boxes = np.asarray(boxes_t, dtype=np.float32)

    # masks can be: list[tensor(H,W)] OR tensor(N,H,W)
    if isinstance(masks_obj, torch.Tensor):
        masks_list = [masks_obj[i] for i in range(masks_obj.shape[0])]
    else:
        masks_list = list(masks_obj)

    # --- color palette (repeats if many instances) ---
    palette = [
        (255,  99,  71),   # tomato
        ( 30, 144, 255),   # dodgerblue
        ( 50, 205,  50),   # limegreen
        (255, 215,   0),   # gold
        (138,  43, 226),   # blueviolet
        (255, 105, 180),   # hotpink
        ( 64, 224, 208),   # turquoise
        (255, 165,   0),   # orange
    ]

    # --- build translucent mask overlay ---
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    for i, m in enumerate(masks_list):
        if isinstance(m, torch.Tensor):
            m_np = m.detach().float().cpu().numpy()
        else:
            m_np = np.asarray(m, dtype=np.float32)

        # binarize if needed
        if m_np.dtype != np.bool_:
            m_bin = m_np > mask_threshold
        else:
            m_bin = m_np

        if m_bin.shape != (H, W):
            # safety: resize mask to image size (nearest)
            m_pil = Image.fromarray((m_bin.astype(np.uint8) * 255), mode="L")
            m_pil = m_pil.resize((W, H), resample=Image.NEAREST)
        else:
            m_pil = Image.fromarray((m_bin.astype(np.uint8) * 255), mode="L")

        r, g, b = palette[i % len(palette)]
        color_img = Image.new("RGBA", (W, H), (r, g, b, int(255 * mask_alpha)))
        overlay.paste(color_img, (0, 0), m_pil)

    composed = Image.alpha_composite(img_rgba, overlay)

    # --- draw boxes + score text on top ---
    draw = ImageDraw.Draw(composed)

    # font: try a nicer default if available, otherwise fallback
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=18)
    except Exception:
        font = ImageFont.load_default()

    for i in range(min(len(scores), len(boxes))):
        x1, y1, x2, y2 = boxes[i].tolist()
        x1, y1, x2, y2 = map(int, [round(x1), round(y1), round(x2), round(y2)])

        # clamp
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W - 1, x2))
        y2 = max(0, min(H - 1, y2))

        r, g, b = palette[i % len(palette)]
        box_color = (r, g, b, 255)

        # rectangle
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=box_width)

        # score label with background
        label = f"{float(scores[i]):.3f}"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]

        pad = 3
        tx, ty = x1, max(0, y1 - th - 2 * pad)
        bg = (0, 0, 0, 180)
        draw.rectangle([tx, ty, tx + tw + 2 * pad, ty + th + 2 * pad], fill=bg)
        draw.text((tx + pad, ty + pad), label, fill=(255, 255, 255, 255), font=font)

    if benchmark:
        return None # in benchmarking mode, we dont save output images

    # --- save ---
    # If saving as JPEG, must drop alpha
    ext = os.path.splitext(out_path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        composed.convert("RGB").save(out_path, quality=95)
    else:
        composed.save(out_path)