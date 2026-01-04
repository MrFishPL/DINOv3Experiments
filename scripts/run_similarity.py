import argparse
import os
import torch
from experiments import SimilarityExperiment
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
import numpy as np
from PIL import Image

def main():
    DEFAULT_PATH = "http://images.cocodataset.org/val2017/000000039769.jpg"

    parser = argparse.ArgumentParser(description="Compute and save norm_feat_lens map as an image from a ViT model input, upsampled to original size.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save the feature length map image.")
    parser.add_argument("--input_path", type=str, default=DEFAULT_PATH, help="Path or URL to the input image.")
    parser.add_argument("--patch_idx", type=int, required=True, help="Selected patch index.")
    args = parser.parse_args()

    DEVICE = "mps"

    input_path = args.input_path
    image = load_image(input_path)

    processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
    model = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
    model.set_attn_implementation('eager')

    experiment = SimilarityExperiment(dinov3_model=model, device=DEVICE, name="similarity")
    orig_size = image.size

    inputs = processor(
        images=image, 
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    ).to(DEVICE)
    
    outputs = experiment.run_dict(preprocessed_inputs=inputs, patch_idx=args.patch_idx)
    feat_map = outputs["similarity"].cpu().squeeze().numpy()

    feat_map_t = torch.from_numpy(feat_map).unsqueeze(0).unsqueeze(0)
    upsampled = torch.nn.functional.interpolate(feat_map_t, size=(orig_size[1], orig_size[0]), mode="nearest")
    feat_map_img = np.clip(upsampled.squeeze().numpy() * 255, 0, 255).astype(np.uint8)

    rgb = np.stack([feat_map_img, feat_map_img, feat_map_img], axis=-1)

    patch_size = getattr(getattr(model, "config", None), "patch_size", 16)
    h_in = int(inputs["pixel_values"].shape[-2])
    w_in = int(inputs["pixel_values"].shape[-1])
    grid_h = h_in // patch_size
    grid_w = w_in // patch_size
    max_idx = grid_h * grid_w - 1

    # for cls token
    if args.patch_idx == -1:
        args.patch_idx = 0
        
    if args.patch_idx < 0 or args.patch_idx > max_idx:
        raise ValueError(f"patch_idx must be in [0, {max_idx}] for grid {grid_h}x{grid_w} (input tensor {h_in}x{w_in}, patch_size={patch_size})")

    row = args.patch_idx // grid_w
    col = args.patch_idx % grid_w

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

    rgb[y0:y1, x0:x1, 0] = 255
    rgb[y0:y1, x0:x1, 1] = 0
    rgb[y0:y1, x0:x1, 2] = 0

    img_out = Image.fromarray(rgb)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "similarity.png")
    img_out.save(out_path)
    print(f"Saved similarity image to {out_path}")

if __name__ == "__main__":
    main()
