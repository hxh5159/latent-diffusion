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
            "linear": PIL.Image.LINEAR,
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

        # CelebA images are 178x218, center crop to 178x178
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
