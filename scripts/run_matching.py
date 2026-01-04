import argparse
import os
import random
import torch
from experiments import MatchingExperiment
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
from PIL import Image, ImageDraw


def get_device(requested: str) -> str:
    # prosta, praktyczna detekcja
    req = requested.lower()
    if req == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if req == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def patch_grid_from_inputs(inputs, patch_size: int):
    # inputs.pixel_values: [B, 3, H, W]
    _, _, H, W = inputs.pixel_values.shape
    Hp, Wp = H // patch_size, W // patch_size
    N = Hp * Wp
    return H, W, Hp, Wp, N


def resize_to_height(img: Image.Image, target_h: int) -> Image.Image:
    if img.height == target_h:
        return img
    scale = target_h / img.height
    new_w = max(1, int(round(img.width * scale)))
    return img.resize((new_w, target_h), Image.BICUBIC)


def main():
    DEFAULT_PATH1 = "https://static.vecteezy.com/system/resources/thumbnails/012/395/175/small/confused-cartoon-cat-free-vector.jpg"
    DEFAULT_PATH2 = "https://i.guim.co.uk/img/media/327aa3f0c3b8e40ab03b4ae80319064e401c6fbc/377_133_3542_2834/master/3542.jpg?width=1200&height=1200&quality=85&auto=format&fit=crop&s=34d32522f47e4a67286f9894fc81c863"

    parser = argparse.ArgumentParser(description="Visualize DINOv3 patch matching between two images.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save output image.")
    parser.add_argument("--input_path1", type=str, default=DEFAULT_PATH1, help="Path or URL to the first input image.")
    parser.add_argument("--input_path2", type=str, default=DEFAULT_PATH2, help="Path or URL to the second input image.")
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cuda", "cpu"], help="Torch device.")
    parser.add_argument("--out_name", type=str, default="matching_vis.png", help="Output filename.")
    parser.add_argument("--max_lines", type=int, default=800, help="Max number of lines to draw (subsample if too many).")
    parser.add_argument("--seed", type=int, default=0, help="Seed for subsampling.")
    args = parser.parse_args()

    device = get_device(args.device)

    image1 = load_image(args.input_path1)
    image2 = load_image(args.input_path2)

    processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
    model = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
    model.set_attn_implementation("eager")
    model.to(device)
    model.eval()

    ps = model.config.patch_size

    inputs1 = processor(images=image1, return_tensors="pt", do_resize=False, do_center_crop=False).to(device)
    inputs2 = processor(images=image2, return_tensors="pt", do_resize=False, do_center_crop=False).to(device)

    H1, W1, Hp1, Wp1, N1 = patch_grid_from_inputs(inputs1, ps)
    H2, W2, Hp2, Wp2, N2 = patch_grid_from_inputs(inputs2, ps)

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

    experiment = MatchingExperiment(dinov3_model=model, device=device, name="matching")
    outputs = experiment.run_dict(preprocessed_inputs1=inp_small, preprocessed_inputs2=inp_large)
    mapping = outputs["matching"].detach().cpu().squeeze(0)

    if mapping.numel() != Hp_s * Wp_s:
        raise RuntimeError(
            f"Unexpected mapping length: got {mapping.numel()}, expected {Hp_s * Wp_s}."
        )

    large_h = img_large.height
    small_resized = resize_to_height(img_small, large_h)
    large_resized = img_large
    
    scale_s = small_resized.height / img_small.height
    scale_t = 1.0

    canvas_w = small_resized.width + large_resized.width
    canvas_h = large_h

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 255))
    canvas.paste(small_resized.convert("RGBA"), (0, 0))
    canvas.paste(large_resized.convert("RGBA"), (small_resized.width, 0))

    draw = ImageDraw.Draw(canvas, "RGBA")

    random.seed(args.seed)
    total = Hp_s * Wp_s
    if args.max_lines is not None and total > args.max_lines:
        idxs = sorted(random.sample(range(total), args.max_lines))
    else:
        idxs = range(total)

    x_offset_large = small_resized.width

    for i in idxs:
        j = int(mapping[i].item())

        rs, cs = divmod(i, Wp_s)
        rt, ct = divmod(j, Wp_t)

        xs = (cs + 0.5) * ps * scale_s
        ys = (rs + 0.5) * ps * scale_s

        xt = x_offset_large + (ct + 0.5) * ps * scale_t
        yt = (rt + 0.5) * ps * scale_t

        draw.line([(xs, ys), (xt, yt)], fill=(255, 0, 0, 110), width=5)

    draw.rectangle([0, 0, canvas_w, 24], fill=(0, 0, 0, 160))
    draw.text((8, 5), f"SMALL={small_label} (left, scaled)  ->  LARGE={large_label} (right)", fill=(255, 255, 255, 220))

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.out_name)
    canvas.convert("RGB").save(out_path)
    print(f"Saved matching visualization to {out_path}")


if __name__ == "__main__":
    main()
