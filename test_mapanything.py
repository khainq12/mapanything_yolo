# test_mapanything.py
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


import torch

from mapanything.models import MapAnything
from mapanything.utils.image import load_images

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Device: {device}")

# Tải model (lần đầu sẽ download ~2-3GB từ HuggingFace)
print("📥 Đang tải model...")
model = MapAnything.from_pretrained("facebook/map-anything").to(device)
print("✅ Model đã tải xong!")

# Dùng ảnh mẫu trong repo
import glob

demo_imgs = glob.glob("assets/*.png") + glob.glob("assets/*.jpg")
if not demo_imgs:
    # Tạo ảnh test ngẫu nhiên nếu không có
    import numpy as np
    from PIL import Image

    os.makedirs("test_images", exist_ok=True)
    for i in range(2):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        Image.fromarray(img).save(f"test_images/img_{i}.jpg")
    demo_imgs = ["test_images/img_0.jpg", "test_images/img_1.jpg"]

print(f"📸 Sử dụng {len(demo_imgs)} ảnh: {demo_imgs}")

# Load & chạy inference
views = load_images(demo_imgs)
print("🚀 Đang chạy inference...")
predictions = model.infer(
    views,
    memory_efficient_inference=True,
    use_amp=True,
    amp_dtype="bf16",
    apply_mask=True,
    mask_edges=True,
)

for i, pred in enumerate(predictions):
    print(f"📊 View {i}: pts3d={pred['pts3d'].shape}, depth={pred['depth_z'].shape}")

print("🎉 MapAnything chạy thành công!")
