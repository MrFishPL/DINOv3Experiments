import argparse
import os
import torch
from experiments import PCAExperiment
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
import numpy as np
from PIL import Image

def main():
    DEFAULT_PATH = "http://images.cocodataset.org/val2017/000000039769.jpg"

    parser = argparse.ArgumentParser(description="Compute and save pca map as an image from a ViT model input, upsampled to original size.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save the pca map image.")
    parser.add_argument("--input_path", type=str, default=DEFAULT_PATH, help="Path or URL to the input image.")
    parser.add_argument("--device", type=str, default="cuda", choices=["mps", "cuda", "cpu"], help="Torch device.")
    args = parser.parse_args()

    input_path = args.input_path
    image = load_image(input_path)

    processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
    model = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
    model.set_attn_implementation('eager')

    experiment = PCAExperiment(dinov3_model=model, device=args.device, name="pca")
    orig_size = image.size

    inputs = processor(
        images=image, 
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    ).to(args.device)
    
    outputs = experiment.run_dict(inputs)
    pca_map = outputs["pca"].cpu().squeeze().numpy()

    pca_map_t = torch.from_numpy(pca_map[None])
    
    upsampled = torch.nn.functional.interpolate(
        pca_map_t,
        size=(orig_size[1], orig_size[0]),
        mode="nearest",
    )

    pca_hwc = upsampled.squeeze(0).permute(1, 2, 0)

    pca_map_img = (pca_hwc.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    img_out = Image.fromarray(pca_map_img, mode="RGB")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "pca.png")
    img_out.save(out_path)
    print(f"Saved pca image to {out_path}")

if __name__ == "__main__":
    main()