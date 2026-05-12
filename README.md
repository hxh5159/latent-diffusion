

# Latent Diffusion复现代码说明

## 前置说明

复现与训练过程中遇到的问题：

1、加载数据集过程困难，数据集大，上传、解压缩时间长、困难，最终通过linux命令将其他文件夹下的数据集直接复制到自己的文件夹下使用（没有删改学长学姐的数据集，只复制以便快速开始训练）

2、安装依赖总是冲突，与cuda环境冲突，pytorch与torchvison冲突，安装taming-transformers也常报错，总结下来下次需要从环境开始仔细研究，一开始就配置好，不然把报错反复问ai，然后按ai的指示改会越改越乱

3、对张量运算，文件操作等不熟悉，理解了计算过程但需要vibe coding

---

## 补充代码说明

---

### 新增文件：`ldm/data/celeba.py`   CelebA 数据集类

目的： 定义celeba 数据集类。

主要操作：

把celebaresize 到 256×256

像素归一化到 [-1, 1]（通过transforms实现）



**代码：**

```python
import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CelebABase(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 split="train",
                 train_size=162770,
                 ):
        self.data_root = data_root
        self.image_paths = sorted([
            f for f in os.listdir(data_root)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        if split == "train":
            self.image_paths = self.image_paths[:train_size]
        elif split == "val":
            self.image_paths = self.image_paths[train_size:]

        self._length = len(self.image_paths)
        print(f"CelebA {split} set: {self._length} images")

        self.size = size
        self.interpolation = {
            "linear": PIL.Image.BILINEAR,   
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        img_path = os.path.join(self.data_root, self.image_paths[i])
        image = Image.open(img_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        img = np.array(image).astype(np.uint8)
        h, w = img.shape[0], img.shape[1]

        crop_size = min(h, w)
        img = img[(h - crop_size) // 2:(h + crop_size) // 2,
                  (w - crop_size) // 2:(w + crop_size) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        return {"image": image}


class CelebATrain(CelebABase):
    def __init__(self, **kwargs):
        super().__init__(split="train", flip_p=0.5, **kwargs)


class CelebAValidation(CelebABase):
    def __init__(self, **kwargs):
        super().__init__(split="val", flip_p=0.0, **kwargs)
```

---

### 新增文件：`configs/latent-diffusion/celeba-ldm-vq-4.yaml`  训练配置（复现要求的二阶段隐空间扩散模型）

目的： 基于 `celebahq-ldm-vq-4.yaml` 模板，将数据集切换为 celeba，其他模型参数保持不变。



`data.train.target` → `ldm.data.celeba.CelebATrain`

`data.validation.target` → `ldm.data.celeba.CelebAValidation`

`data_root` → `data/celeba`  此步为了适配复制数据集的操作

代码：

```yaml
model:
  base_learning_rate: 2.0e-06
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.0015
    linear_end: 0.0195
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    first_stage_key: image
    image_size: 64
    channels: 3
    monitor: val/loss_simple_ema

    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 64
        in_channels: 3
        out_channels: 3
        model_channels: 224
        attention_resolutions:
        - 8
        - 4
        - 2
        num_res_blocks: 2
        channel_mult:
        - 1
        - 2
        - 3
        - 4
        num_head_channels: 32
    first_stage_config:
      target: ldm.models.autoencoder.VQModelInterface
      params:
        embed_dim: 3
        n_embed: 8192
        ckpt_path: models/first_stage_models/vq-f4/model.ckpt
        ddconfig:
          double_z: false
          z_channels: 3
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult:
          - 1
          - 2
          - 4
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity
    cond_stage_config: __is_unconditional__
data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 24
    num_workers: 5
    wrap: false
    train:
      target: ldm.data.celeba.CelebATrain
      params:
        size: 256
        data_root: data/celeba
    validation:
      target: ldm.data.celeba.CelebAValidation
      params:
        size: 256
        data_root: data/celeba

lightning:
  callbacks:
    image_logger:
      target: main.ImageLogger
      params:
        batch_frequency: 5000
        max_images: 8
        increase_log_steps: False

  trainer:
    benchmark: True
```

---

### 新增文件：`scripts/download_vq_f4.py` — VQ-F4 预训练权重下载

目的： 下载 VQ-F4 自编码器的预训练权重（作为一阶段编解码器）。

代码：

```python
import os
import urllib.request
import zipfile

URL = "https://ommer-lab.com/files/latent-diffusion/vq-f4.zip"
DST_DIR = "models/first_stage_models/vq-f4"

os.makedirs(DST_DIR, exist_ok=True)
zip_path = os.path.join(DST_DIR, "model.zip")

print(f"Downloading VQ-F4 from {URL}")
urllib.request.urlretrieve(URL, zip_path)
print("Download complete.")

print(f"Extracting to {DST_DIR}")
with zipfile.ZipFile(zip_path, 'r') as zf:
    zf.extractall(DST_DIR)
os.remove(zip_path)

ckpt = os.path.join(DST_DIR, "model.ckpt")
if os.path.exists(ckpt):
    print(f"Done. Checkpoint at {ckpt}")
else:
    print("WARNING: model.ckpt not found after extraction. Check the zip contents.")
    for f in os.listdir(DST_DIR):
        print(f"  {f}")
```

---

### 新增文件：`scripts/sample_celeba.py` — 推理采样脚本

目的： 加载训练好的 LDM checkpoint，使用 DDIM 采样生成人脸图像。基于 `scripts/sample_diffusion.py` 修改。

加载训练好的 LDM checkpoint + 训练时的 config.yaml

进行 DDIM 采样

通过 VQ-F4 decoder 将潜变量解码为 256×256 图像

**代码：**

```python
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
    pl_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
```



## 完整运行流程

### 环境准备

```bash
# 安装依赖
pip install pytorch-lightning==1.4.2 omegaconf==2.1.1 einops==0.3.0 test-tube
pip install git+https://github.com/CompVis/taming-transformers.git@master
pip install git+https://github.com/openai/CLIP.git@main
pip install -e .
```

### 下载 VQ-F4 预训练权重

```bash
python scripts/download_vq_f4.py
```

### 准备 CelebA 数据集

理想中，从 [CelebA 官网](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 下载 `img_align_celeba.zip`，解压到 `data/celeba/`，但实际上复制了其他文件夹下的数据集

### 训练

```bash
# 进入项目目录
cd houwenzhe/latent-diffusion


# GPU 训练（单卡）
python main.py --base configs/latent-diffusion/celeba-ldm-vq-4.yaml -t --gpus 0
```




