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
