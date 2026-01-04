"""Script to compute and visualize similarity maps from DINOv3."""

import argparse
import os

import numpy as np
import torch
from PIL import Image

from experiments import SimilarityExperiment
from utils import load_dinov3_model, process_image, upsample_to_original_size

DEFAULT_IMAGE_PATH = "http://images.cocodataset.org/val2017/000000039769.jpg"


def _validate_patch_idx(patch_idx: int, inputs: dict, model, orig_size: tuple) -> tuple:
    """Validate patch index and compute patch coordinates."""
    patch_size = model.config.patch_size
    h_in = int(inputs["pixel_values"].shape[-2])
    w_in = int(inputs["pixel_values"].shape[-1])
    grid_h = h_in // patch_size
    grid_w = w_in // patch_size
    max_idx = grid_h * grid_w - 1

    # Handle CLS token (patch_idx == -1)
    if patch_idx == -1:
        patch_idx = 0

    if patch_idx < 0 or patch_idx > max_idx:
        raise ValueError(
            f"patch_idx must be in [0, {max_idx}] for grid {grid_h}x{grid_w} "
            f"(input tensor {h_in}x{w_in}, patch_size={patch_size})"
        )

    row = patch_idx // grid_w
    col = patch_idx % grid_w

    sx = orig_size[0] / float(w_in)
    sy = orig_size[1] / float(h_in)

    x0 = int(col * patch_size * sx)
    x1 = int((col + 1) * patch_size * sx)
    y0 = int(row * patch_size * sy)
    y1 = int((row + 1) * patch_size * sy)

    x0 = max(0, min(orig_size[0], x0))
    x1 = max(0, min(orig_size[0], x1))
    y0 = max(0, min(orig_size[1], y0))
    y1 = max(0, min(orig_size[1], y1))

    return (x0, y0, x1, y1)


def main():
    """Run similarity experiment and save visualization."""
    parser = argparse.ArgumentParser(
        description="Compute and save similarity map as an image, upsampled to original size."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save the similarity map image.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=DEFAULT_IMAGE_PATH,
        help="Path or URL to the input image.",
    )
    parser.add_argument(
        "--patch_idx",
        type=int,
        required=True,
        help="Selected patch index. Use -1 for CLS token.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["mps", "cuda", "cpu"],
        help="Torch device.",
    )
    args = parser.parse_args()

    # Load and process image
    image, inputs = process_image(args.input_path, device=args.device)
    orig_size = image.size

    # Load model and create experiment
    model = load_dinov3_model(device=args.device)
    experiment = SimilarityExperiment(dinov3_model=model, device=args.device, name="similarity")

    # Validate patch index
    x0, y0, x1, y1 = _validate_patch_idx(args.patch_idx, inputs, model, orig_size)

    # Run experiment
    outputs = experiment.run_dict(preprocessed_inputs=inputs, patch_idx=args.patch_idx)
    feat_map = outputs["similarity"].cpu().squeeze().numpy()

    # Upsample to original size
    feat_map_t = torch.from_numpy(feat_map).unsqueeze(0).unsqueeze(0)
    upsampled = upsample_to_original_size(feat_map_t, orig_size, mode="nearest")
    feat_map_img = np.clip(upsampled.squeeze().numpy() * 255, 0, 255).astype(np.uint8)

    # Create RGB image and highlight selected patch
    rgb = np.stack([feat_map_img, feat_map_img, feat_map_img], axis=-1)
    rgb[y0:y1, x0:x1, 0] = 255  # Red channel
    rgb[y0:y1, x0:x1, 1] = 0  # Green channel
    rgb[y0:y1, x0:x1, 2] = 0  # Blue channel

    img_out = Image.fromarray(rgb)

    # Save output
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "similarity.png")
    img_out.save(out_path)
    print(f"Saved similarity image to {out_path}")


if __name__ == "__main__":
    main()
