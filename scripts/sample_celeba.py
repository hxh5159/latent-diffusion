import argparse, os, sys, glob, datetime, yaml
import torch
import time
import numpy as np
from tqdm import tqdm

from omegaconf import OmegaConf
from PIL import Image

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config


def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def load_model(config, ckpt_path):
    print(f"Loading model from {ckpt_path}")
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    global_step = pl_sd.get("global_step", 0)
    if "state_dict" in pl_sd:
        pl_sd = pl_sd["state_dict"]
    model = load_model_from_config(config.model, pl_sd)
    return model, global_step


@torch.no_grad()
def sample_celeba(model, batch_size, ddim_steps, eta, n_samples, out_dir):
    ddim = DDIMSampler(model)
    shape = (model.channels, model.image_size, model.image_size)

    with model.ema_scope("Sampling"):
        total_batches = (n_samples + batch_size - 1) // batch_size
        n_saved = 0
        for _ in tqdm(range(total_batches), desc="Sampling"):
            current_bs = min(batch_size, n_samples - n_saved)
            samples, _ = ddim.sample(ddim_steps, batch_size=current_bs, shape=shape,
                                     eta=eta, verbose=False)
            x_samples = model.decode_first_stage(samples)
            for j in range(x_samples.shape[0]):
                img = custom_to_pil(x_samples[j])
                img.save(os.path.join(out_dir, f"sample_{n_saved:06d}.png"))
                n_saved += 1

    print(f"Saved {n_saved} samples to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, required=True,
                        help="Path to trained checkpoint (.ckpt)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml (auto-detected from --resume if omitted)")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--eta", type=float, default=1.0,
                        help="DDIM eta (0.0 = deterministic, 1.0 = stochastic)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: auto-generated under samples/)")
    opt = parser.parse_args()

    sys.path.append(os.getcwd())

    if opt.config is None:
        logdir = os.path.dirname(os.path.dirname(opt.resume))
        config_path = glob.glob(os.path.join(logdir, "configs", "*project.yaml"))
        if not config_path:
            raise ValueError(f"Cannot auto-detect config in {logdir}. Use --config.")
        opt.config = config_path[0]
        print(f"Auto-detected config: {opt.config}")

    config = OmegaConf.load(opt.config)
    model, global_step = load_model(config, opt.resume)
    print(f"Global step: {global_step}")

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if opt.out_dir is None:
        logdir = os.path.dirname(os.path.dirname(opt.resume))
        opt.out_dir = os.path.join(logdir, "samples", f"step_{global_step:08d}", now)
    os.makedirs(opt.out_dir, exist_ok=True)

    print(f"Output: {opt.out_dir}")
    print(f"DDIM steps: {opt.ddim_steps}, eta: {opt.eta}")
    print(f"Generating {opt.n_samples} samples...")

    sample_celeba(model, opt.batch_size, opt.ddim_steps, opt.eta,
                  opt.n_samples, opt.out_dir)

    # Save a grid image
    all_images = sorted(glob.glob(os.path.join(opt.out_dir, "*.png")))
    if len(all_images) >= 16:
        grid_imgs = [Image.open(p) for p in all_images[:64]]
        grid_size = int(np.ceil(np.sqrt(len(grid_imgs))))
        cell_w, cell_h = grid_imgs[0].size
        grid = Image.new("RGB", (grid_size * cell_w, grid_size * cell_h))
        for idx, img in enumerate(grid_imgs):
            row, col = idx // grid_size, idx % grid_size
            grid.paste(img, (col * cell_w, row * cell_h))
        grid_path = os.path.join(opt.out_dir, "grid.png")
        grid.save(grid_path)
        print(f"Grid saved to {grid_path}")

    print("Done.")


if __name__ == "__main__":
    main()
