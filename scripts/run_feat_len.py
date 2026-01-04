import argparse
import os
import torch
from experiments import FeaturesLengthExperiment
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
import numpy as np
from PIL import Image

def main():
    DEFAULT_PATH = "http://images.cocodataset.org/val2017/000000039769.jpg"

    parser = argparse.ArgumentParser(description="Compute and save norm_feat_lens map as an image from a ViT model input, upsampled to original size.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save the feature length map image.")
    parser.add_argument("--input_path", type=str, default=DEFAULT_PATH, help="Path or URL to the input image.")
    parser.add_argument("--device", type=str, default="cuda", choices=["mps", "cuda", "cpu"], help="Torch device.")
    args = parser.parse_args()

    input_path = args.input_path
    image = load_image(input_path)

    processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
    model = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
    model.set_attn_implementation('eager')

    experiment = FeaturesLengthExperiment(dinov3_model=model, device=args.device, name="features_length")
    orig_size = image.size

    inputs = processor(
        images=image, 
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    ).to(args.device)
    
    outputs = experiment.run_dict(inputs)
    feat_map = outputs["norm_feat_lens"].cpu().squeeze().numpy()

    feat_map_t = torch.from_numpy(feat_map).unsqueeze(0).unsqueeze(0)
    upsampled = torch.nn.functional.interpolate(feat_map_t, size=(orig_size[1], orig_size[0]), mode="nearest")
    feat_map_img = np.clip(upsampled.squeeze().numpy() * 255, 0, 255).astype(np.uint8)
    img_out = Image.fromarray(feat_map_img)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "norm_feat_lens.png")
    img_out.save(out_path)
    print(f"Saved norm_feat_lens image to {out_path}")

if __name__ == "__main__":
    main()