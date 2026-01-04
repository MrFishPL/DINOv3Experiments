"""Script to visualize DINOv3 patch matching between two images."""

import argparse
import os
import random

import torch
from PIL import Image, ImageDraw
from transformers.image_utils import load_image

from experiments import MatchingExperiment
from utils import (
    load_dinov3_model,
    load_dinov3_processor,
    patch_grid_from_inputs,
    resize_to_height,
)

DEFAULT_PATH1 = (
    "https://static.vecteezy.com/system/resources/thumbnails/012/395/175/small/"
    "confused-cartoon-cat-free-vector.jpg"
)
DEFAULT_PATH2 = (
    "https://i.guim.co.uk/img/media/327aa3f0c3b8e40ab03b4ae80319064e401c6fbc/"
    "377_133_3542_2834/master/3542.jpg?width=1200&height=1200&quality=85&auto=format&fit=crop&s=34d32522f47e4a67286f9894fc81c863"
)


def main():
    """Run matching experiment and save visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize DINOv3 patch matching between two images."
    )
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--input_path1", type=str, default=DEFAULT_PATH1)
    parser.add_argument("--input_path2", type=str, default=DEFAULT_PATH2)
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["mps", "cuda", "cpu"]
    )
    parser.add_argument("--out_name", type=str, default="matching_vis.png")
    parser.add_argument(
        "--max_lines", type=int, default=800, help="Maximum number of lines to draw."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="Draw only top K matches by score. 0 means use all or max_lines.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    args = parser.parse_args()

    device = args.device

    # Load images
    image1 = load_image(args.input_path1)
    image2 = load_image(args.input_path2)

    # Load model and processor
    model = load_dinov3_model(device=device)
    processor = load_dinov3_processor()
    ps = model.config.patch_size

    # Process images
    inputs1 = processor(
        images=image1, return_tensors="pt", do_resize=False, do_center_crop=False
    ).to(device)
    inputs2 = processor(
        images=image2, return_tensors="pt", do_resize=False, do_center_crop=False
    ).to(device)

    # Determine which image is smaller
    _, _, Hp1, Wp1, N1 = patch_grid_from_inputs(inputs1, ps)
    _, _, Hp2, Wp2, N2 = patch_grid_from_inputs(inputs2, ps)

    if N1 <= N2:
        img_small, img_large = image1, image2
        inp_small, inp_large = inputs1, inputs2
        Hp_s, Wp_s = Hp1, Wp1
        Hp_t, Wp_t = Hp2, Wp2
        small_label = "image1"
        large_label = "image2"
    else:
        img_small, img_large = image2, image1
        inp_small, inp_large = inputs2, inputs1
        Hp_s, Wp_s = Hp2, Wp2
        Hp_t, Wp_t = Hp1, Wp1
        small_label = "image2"
        large_label = "image1"

    # Run experiment
    experiment = MatchingExperiment(dinov3_model=model, device=device, name="matching")
    outputs = experiment.run_dict(
        preprocessed_inputs1=inp_small, preprocessed_inputs2=inp_large
    )
    mapping = outputs["matching"].detach().cpu().squeeze(0)
    scores = outputs["scores"].detach().cpu().squeeze(0)

    # Validate outputs
    total = Hp_s * Wp_s
    if mapping.numel() != total or scores.numel() != total:
        raise RuntimeError(
            f"Unexpected output length: mapping={mapping.numel()} "
            f"scores={scores.numel()} expected={total}"
        )

    # Prepare images for visualization
    large_h = img_large.height
    small_resized = resize_to_height(img_small, large_h)
    large_resized = img_large

    scale_s = small_resized.height / img_small.height
    scale_t = 1.0

    # Create canvas
    canvas_w = small_resized.width + large_resized.width
    canvas_h = large_h

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 255))
    canvas.paste(small_resized.convert("RGBA"), (0, 0))
    canvas.paste(large_resized.convert("RGBA"), (small_resized.width, 0))

    draw = ImageDraw.Draw(canvas, "RGBA")

    # Select patches to visualize
    if args.top_k and args.top_k > 0:
        k = min(args.top_k, total)
        idxs = torch.topk(scores, k=k, largest=True).indices.tolist()
        idxs.sort()
    else:
        random.seed(args.seed)
        if args.max_lines is not None and total > args.max_lines:
            idxs = sorted(random.sample(range(total), args.max_lines))
        else:
            idxs = list(range(total))

    # Draw matching lines
    x_offset_large = small_resized.width

    for i in idxs:
        j = int(mapping[i].item())

        rs, cs = divmod(i, Wp_s)
        rt, ct = divmod(j, Wp_t)

        xs = (cs + 0.5) * ps * scale_s
        ys = (rs + 0.5) * ps * scale_s

        xt = x_offset_large + (ct + 0.5) * ps * scale_t
        yt = (rt + 0.5) * ps * scale_t

        draw.line([(xs, ys), (xt, yt)], fill=(255, 0, 0, 110), width=3)

    # Add label
    draw.rectangle([0, 0, canvas_w, 24], fill=(0, 0, 0, 160))
    draw.text(
        (8, 5),
        f"SMALL={small_label} (left, scaled)  ->  LARGE={large_label} (right)",
        fill=(255, 255, 255, 220),
    )

    # Save output
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.out_name)
    canvas.convert("RGB").save(out_path)
    print(f"Saved matching visualization to {out_path}")


if __name__ == "__main__":
    main()