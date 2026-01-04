"""Script to compute and visualize PCA visualization from DINOv3 patch tokens."""

import argparse
import os

import torch
from PIL import Image

from experiments import PCAExperiment
from utils import load_dinov3_model, process_image, upsample_to_original_size

DEFAULT_IMAGE_PATH = "http://images.cocodataset.org/val2017/000000039769.jpg"


def main():
    """Run PCA experiment and save visualization."""
    parser = argparse.ArgumentParser(
        description="Compute and save PCA visualization as an RGB image, upsampled to original size."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save the PCA map image.",
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
    experiment = PCAExperiment(dinov3_model=model, device=args.device, name="pca")

    # Run experiment
    outputs = experiment.run_dict(inputs)
    pca_map = outputs["pca"].cpu().squeeze().numpy()

    # Upsample to original size
    pca_map_t = torch.from_numpy(pca_map[None])
    upsampled = upsample_to_original_size(pca_map_t, orig_size, mode="nearest")
    pca_hwc = upsampled.permute(1, 2, 0)

    # Convert to image and save
    pca_map_img = (pca_hwc.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    img_out = Image.fromarray(pca_map_img, mode="RGB")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "pca.png")
    img_out.save(out_path)
    print(f"Saved PCA image to {out_path}")


if __name__ == "__main__":
    main()