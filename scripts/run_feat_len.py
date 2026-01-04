"""Script to compute and visualize feature length maps from DINOv3."""

import argparse
import os

import numpy as np
import torch
from PIL import Image

from experiments import FeaturesLengthExperiment
from utils import load_dinov3_model, process_image, upsample_to_original_size

DEFAULT_IMAGE_PATH = "http://images.cocodataset.org/val2017/000000039769.jpg"


def main():
    """Run feature length experiment and save visualization."""
    parser = argparse.ArgumentParser(
        description="Compute and save normalized feature length map as an image, upsampled to original size."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save the feature length map image.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=DEFAULT_IMAGE_PATH,
        help="Path or URL to the input image.",
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
    experiment = FeaturesLengthExperiment(dinov3_model=model, device=args.device, name="features_length")

    # Run experiment
    outputs = experiment.run_dict(inputs)
    feat_map = outputs["norm_feat_lens"].cpu().squeeze().numpy()

    # Upsample to original size
    feat_map_t = torch.from_numpy(feat_map).unsqueeze(0).unsqueeze(0)
    upsampled = upsample_to_original_size(feat_map_t, orig_size, mode="nearest")
    feat_map_img = np.clip(upsampled.squeeze().numpy() * 255, 0, 255).astype(np.uint8)
    img_out = Image.fromarray(feat_map_img)

    # Save output
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "norm_feat_lens.png")
    img_out.save(out_path)
    print(f"Saved norm_feat_lens image to {out_path}")


if __name__ == "__main__":
    main()