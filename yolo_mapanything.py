#!/usr/bin/env python3
"""
YOLO + MapAnything: Semantic 3D Map
- MapAnything: tạo 3D point cloud từ ảnh/video
- YOLO: detect object trong từng frame
- Kết hợp: gán nhãn 3D points theo YOLO detection → Semantic 3D Map
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


import cv2
import numpy as np
import torch

# ====== YOLO imports ======
from ultralytics import YOLO

# ====== MapAnything imports ======
from mapanything.models import MapAnything
from mapanything.utils.geometry import depthmap_to_world_frame
from mapanything.utils.image import load_images
from mapanything.utils.viz import predictions_to_glb

# Class → Color mapping (màu nổi bật cho mỗi loại vật thể)
CLASS_COLORS = {
    "person": [255, 0, 0],  # Đỏ
    "car": [0, 255, 0],  # Xanh lá
    "truck": [0, 200, 0],
    "bus": [0, 150, 0],
    "bicycle": [0, 0, 255],  # Xanh dương
    "motorbike": [0, 50, 200],
    "chair": [255, 255, 0],  # Vàng
    "diningtable": [200, 200, 0],
    "table": [200, 200, 0],
    "couch": [150, 150, 0],
    "tv": [255, 0, 255],  # Hồng
    "laptop": [200, 0, 200],
    "cell phone": [150, 0, 150],
    "phone": [150, 0, 150],
    "bottle": [0, 255, 255],  # Cyan
    "cup": [0, 200, 200],
    "dog": [255, 128, 0],  # Cam
    "cat": [200, 100, 0],
    "refrigerator": [128, 0, 255],  # Tím
    "keyboard": [255, 165, 0],  # Orange
    "mouse": [0, 128, 128],  # Teal
    "sink": [64, 224, 208],  # Turquoise
    "handbag": [255, 192, 203],  # Pink
    "traffic light": [255, 215, 0],  # Gold
    "train": [165, 42, 42],  # Brown
}
DEFAULT_OBJECT_COLOR = [128, 128, 128]  # Grey
NO_CLASS = -1


def extract_frames(video_path, output_dir, interval_sec=1.0):
    """Extract frames từ video, trả về list kích thước ảnh gốc"""
    os.makedirs(output_dir, exist_ok=True)
    vs = cv2.VideoCapture(video_path)
    fps = vs.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps * interval_sec))

    count, saved = 0, 0
    orig_sizes = []  # Lưu kích thước ảnh gốc (H, W)
    while True:
        gotit, frame = vs.read()
        if not gotit:
            break
        count += 1
        if count % frame_interval == 0:
            h, w = frame.shape[:2]
            orig_sizes.append((h, w))
            cv2.imwrite(os.path.join(output_dir, f"{saved:06d}.png"), frame)
            saved += 1
    vs.release()
    print(f"Extracted {saved} frames")
    return saved, orig_sizes


def get_frame_sizes(frames_dir):
    """Đọc kích thước của tất cả frames trong thư mục"""
    sizes = []
    exts = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]
    files = sorted(
        [f for f in os.listdir(frames_dir) if any(f.endswith(e) for e in exts)]
    )
    for fname in files:
        img = cv2.imread(os.path.join(frames_dir, fname))
        if img is not None:
            sizes.append((img.shape[0], img.shape[1]))  # (H, W)
        else:
            sizes.append((0, 0))
    return sizes


def run_yolo_on_frames(frames_dir, yolo_model, conf_threshold=0.3):
    """
    Chạy YOLO trên tất cả frames → trả về detections per frame.
    Bounding box coords ở KÍCH THƯỚC ẢNH GỐC (chưa scale).
    """
    results = yolo_model.predict(
        source=frames_dir,
        conf=conf_threshold,
        save=False,
        verbose=False,
    )

    all_detections = []
    for r in results:
        frame_dets = []
        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = yolo_model.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                frame_dets.append(
                    {
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": conf,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    }
                )
        all_detections.append(frame_dets)

    return all_detections


def colorize_images_with_yolo(
    outputs, yolo_detections, yolo_model_names, orig_frame_sizes
):
    """
    Tạo bản images đã tô màu theo YOLO detection.

    QUAN TRỌNG: YOLO bbox coords ở kích thước ảnh gốc (vd 1920x1080),
    nhưng MapAnything output img_no_norm ở model resolution (vd 518x291).
    Phải scale bbox từ gốc → model resolution.

    Returns:
        colored_images: list of (H, W, 3) numpy arrays
        label_maps: list of (H, W) numpy arrays
    """
    colored_images = []
    label_maps = []
    class_counts = {}

    for view_idx, pred in enumerate(outputs):
        img = pred["img_no_norm"][0].cpu().numpy().copy()  # (H_model, W_model, 3)
        H_model, W_model = img.shape[:2]

        # Label map mặc định
        label_map = np.full((H_model, W_model), NO_CLASS, dtype=np.int32)
        colored = img.copy()

        # Lấy kích thước ảnh gốc cho frame này
        if view_idx < len(orig_frame_sizes):
            H_orig, W_orig = orig_frame_sizes[view_idx]
        else:
            H_orig, W_orig = H_model, W_model  # fallback

        # Scale factor: từ ảnh gốc → model resolution
        scale_x = W_model / W_orig if W_orig > 0 else 1.0
        scale_y = H_model / H_orig if H_orig > 0 else 1.0

        if view_idx < len(yolo_detections):
            for det in yolo_detections[view_idx]:
                x1, y1, x2, y2 = det["bbox"]
                cls_id = det["class_id"]
                cls_name = det["class_name"]

                # ====== SCALE bbox từ ảnh gốc → model resolution ======
                sx1 = int(x1 * scale_x)
                sy1 = int(y1 * scale_y)
                sx2 = int(x2 * scale_x)
                sy2 = int(y2 * scale_y)

                # Clamp vào ảnh
                sx1 = max(0, min(sx1, W_model - 1))
                sy1 = max(0, min(sy1, H_model - 1))
                sx2 = max(0, min(sx2, W_model - 1))
                sy2 = max(0, min(sy2, H_model - 1))

                # Skip nếu bbox quá nhỏ sau khi scale
                if sx2 <= sx1 or sy2 <= sy1:
                    continue

                # Gán label cho pixels trong bbox
                label_map[sy1:sy2, sx1:sx2] = cls_id

                # Tô màu vùng bbox
                color = CLASS_COLORS.get(cls_name, DEFAULT_OBJECT_COLOR)
                colored[sy1:sy2, sx1:sx2] = color

                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        colored_images.append(colored)
        label_maps.append(label_map)

    print("\n📊 Detection Summary:")
    for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"   {cls_name}: {count} detections across frames")

    return colored_images, label_maps


def save_semantic_ply_with_labels(
    points, colors, labels, yolo_model_names, output_path
):
    """
    Lưu PLY file có kèm label cho mỗi point.
    """
    unique_labels = np.unique(labels)
    label_to_name = {}
    for lbl in unique_labels:
        if lbl == NO_CLASS:
            label_to_name[lbl] = "background"
        else:
            label_to_name[lbl] = yolo_model_names.get(int(lbl), f"class_{lbl}")

    num_points = len(points)
    colors_uint8 = np.clip(colors, 0, 255).astype(np.uint8)

    with open(output_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property int label\n")
        f.write("end_header\n")

        for i in range(num_points):
            f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} ")
            f.write(f"{colors_uint8[i, 0]} {colors_uint8[i, 1]} {colors_uint8[i, 2]} ")
            f.write(f"{labels[i]}\n")

    print(f"   ✅ Semantic PLY (with labels): {output_path}")

    print("\n   📋 Label Legend:")
    for lbl in sorted(unique_labels):
        name = label_to_name[lbl]
        count = np.sum(labels == lbl)
        print(f"      Label {lbl} = {name}: {count:,} points")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLO + MapAnything Semantic 3D Map")
    parser.add_argument(
        "--input", type=str, required=True, help="Video file hoặc thư mục ảnh"
    )
    parser.add_argument(
        "--yolo_model", type=str, default="yolo11n.pt", help="YOLO model"
    )
    parser.add_argument(
        "--frame_interval", type=float, default=1.0, help="Lấy frame mỗi X giây"
    )
    parser.add_argument(
        "--conf", type=float, default=0.3, help="YOLO confidence threshold"
    )
    parser.add_argument(
        "--output_dir", type=str, default="semantic_3d_output", help="Thư mục output"
    )
    parser.add_argument("--apache", action="store_true", help="Dùng model Apache 2.0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ====== Step 1: Chuẩn bị ảnh ======
    orig_frame_sizes = []  # (H, W) của ảnh gốc
    if args.input.endswith((".mp4", ".avi", ".mov", ".mkv")):
        print(f"🎬 Extracting frames from video: {args.input}")
        frames_dir = os.path.join(args.output_dir, "frames")
        _, orig_frame_sizes = extract_frames(
            args.input, frames_dir, args.frame_interval
        )
    else:
        frames_dir = args.input
        print(f"📁 Using image folder: {frames_dir}")
        orig_frame_sizes = get_frame_sizes(frames_dir)

    print(f"   Frame sizes (first 3): {orig_frame_sizes[:3]}")

    # ====== Step 2: Chạy YOLO detection ======
    print(f"\n🔍 Running YOLO ({args.yolo_model}) on frames...")
    yolo = YOLO(args.yolo_model)
    yolo_model_names = yolo.model.names
    yolo_detections = run_yolo_on_frames(frames_dir, yolo, args.conf)
    total_dets = sum(len(d) for d in yolo_detections)
    print(f"   Detected {total_dets} objects across {len(yolo_detections)} frames")

    # ====== Step 3: Chạy MapAnything 3D reconstruction ======
    print("\n🗺️ Running MapAnything 3D reconstruction...")
    model_name = (
        "facebook/map-anything-apache" if args.apache else "facebook/map-anything"
    )
    ma_model = MapAnything.from_pretrained(model_name).to(device)

    views = load_images(frames_dir)
    print(f"   Loaded {len(views)} views")

    outputs = ma_model.infer(
        views,
        memory_efficient_inference=True,
        minibatch_size=1,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
    )
    print("   Inference complete!")

    # In model output size để debug
    out_H, out_W = outputs[0]["img_no_norm"][0].shape[:2]
    print(f"   Model output image size: {out_W}x{out_H}")
    print(f"   Original frame size: {orig_frame_sizes[0][1]}x{orig_frame_sizes[0][0]}")
    print(
        f"   Scale factor: x={out_W / orig_frame_sizes[0][1]:.3f}, y={out_H / orig_frame_sizes[0][0]:.3f}"
    )

    # ====== Step 4: Tô màu images theo YOLO ======
    print("\n🎨 Colorizing images with YOLO detections...")

    colored_images, label_maps = colorize_images_with_yolo(
        outputs, yolo_detections, yolo_model_names, orig_frame_sizes
    )

    # ====== Step 5: Lưu kết quả ======
    print(f"\n💾 Saving results to {args.output_dir}/")

    # 5a. Original map (màu gốc)
    orig_pts = []
    orig_imgs = []
    orig_masks = []
    for pred in outputs:
        depth = pred["depth_z"][0].squeeze(-1)
        K = pred["intrinsics"][0]
        pose = pred["camera_poses"][0]
        pts3d, valid = depthmap_to_world_frame(depth, K, pose)
        mask = (
            pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool) & valid.cpu().numpy()
        )
        orig_pts.append(pts3d.cpu().numpy())
        orig_imgs.append(pred["img_no_norm"][0].cpu().numpy())
        orig_masks.append(mask)

    predictions_orig = {
        "world_points": np.stack(orig_pts),
        "images": np.stack(orig_imgs),
        "final_masks": np.stack(orig_masks),
    }
    orig_glb = os.path.join(args.output_dir, "original_3d_map.glb")
    scene = predictions_to_glb(predictions_orig, as_mesh=True)
    scene.export(orig_glb)
    print(f"   ✅ Original map: {orig_glb}")

    # 5b. Semantic map (YOLO colors) - dùng predictions_to_glb
    sem_pts = []
    sem_imgs = []
    sem_masks = []
    for i, pred in enumerate(outputs):
        depth = pred["depth_z"][0].squeeze(-1)
        K = pred["intrinsics"][0]
        pose = pred["camera_poses"][0]
        pts3d, valid = depthmap_to_world_frame(depth, K, pose)
        mask = (
            pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool) & valid.cpu().numpy()
        )
        sem_pts.append(pts3d.cpu().numpy())
        sem_imgs.append(colored_images[i])
        sem_masks.append(mask)

    predictions_sem = {
        "world_points": np.stack(sem_pts),
        "images": np.stack(sem_imgs),
        "final_masks": np.stack(sem_masks),
    }
    sem_glb = os.path.join(args.output_dir, "semantic_3d_map.glb")
    scene_sem = predictions_to_glb(predictions_sem, as_mesh=True)
    scene_sem.export(sem_glb)
    print(f"   ✅ Semantic map (GLB): {sem_glb}")

    # 5c. Semantic PLY with labels
    sem_ply = os.path.join(args.output_dir, "semantic_3d_map.ply")
    all_pts = []
    all_labels = []
    all_colors = []
    for i in range(len(outputs)):
        mask = sem_masks[i]
        all_pts.append(sem_pts[i][mask])
        all_labels.append(label_maps[i][mask])
        all_colors.append(colored_images[i][mask])

    all_pts = np.concatenate(all_pts, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    save_semantic_ply_with_labels(
        all_pts, all_colors, all_labels, yolo_model_names, sem_ply
    )

    # 5d. Raw data
    det_path = os.path.join(args.output_dir, "detections.npz")
    np.savez(det_path, points=all_pts, labels=all_labels, colors=all_colors)
    print(f"   ✅ Detections data: {det_path}")

    # ====== Summary ======
    unique_classes = set()
    for dets in yolo_detections:
        for d in dets:
            unique_classes.add(d["class_name"])

    print(f"\n{'=' * 50}")
    print("📊 SUMMARY")
    print(f"{'=' * 50}")
    print(f"   Frames: {len(yolo_detections)}")
    print(f"   3D Points: {len(all_pts):,}")
    print(f"   Objects detected: {total_dets}")
    print(f"   Classes: {', '.join(sorted(unique_classes))}")
    print(f"   Output: {args.output_dir}/")
    print(f"{'=' * 50}")
    print("\n🌐 View GLB at: https://3dviewer.net/")
