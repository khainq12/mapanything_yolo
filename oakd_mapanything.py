"""
MapAnything + OAK-D (DepthAI v3.5)
Correct v3 flow: pipeline.build() + pipeline.start()
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time

import cv2
import depthai as dai
import numpy as np
import torch

from mapanything.models import MapAnything
from mapanything.utils.image import preprocess_inputs

# ============================================================
NUM_FRAMES = 3
CAPTURE_INTERVAL = 3
RGB_WIDTH = 640
RGB_HEIGHT = 480
OUTPUT_DIR = "oakd_output"


# ============================================================
# 1. CAPTURE TỪ OAK-D (DepthAI v3 flow)
# ============================================================
def capture_oakd():
    """Capture RGB + Depth + Intrinsics từ OAK-D."""
    print("🔧 Khởi tạo OAK-D pipeline (v3)...")

    pipeline = dai.Pipeline()

    # --- RGB Camera ---
    cam_rgb = pipeline.create(dai.node.Camera)
    cam_rgb.build(dai.CameraBoardSocket.CAM_A)
    rgb_out = cam_rgb.requestOutput(
        (RGB_WIDTH, RGB_HEIGHT), type=dai.ImgFrame.Type.RGB888i
    )
    q_rgb = rgb_out.createOutputQueue()

    # --- Left Mono ---
    cam_left = pipeline.create(dai.node.Camera)
    cam_left.build(dai.CameraBoardSocket.CAM_B)
    left_out = cam_left.requestOutput((640, 400))

    # --- Right Mono ---
    cam_right = pipeline.create(dai.node.Camera)
    cam_right.build(dai.CameraBoardSocket.CAM_C)
    right_out = cam_right.requestOutput((640, 400))

    # --- Stereo Depth ---
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    left_out.link(stereo.left)
    right_out.link(stereo.right)
    q_depth = stereo.depth.createOutputQueue()

    # Build + Start
    print("🏗️ Building pipeline...")
    pipeline.build()
    print("▶️ Starting pipeline...")
    pipeline.start()
    print("✅ Pipeline running!")

    # Lấy intrinsics
    intrinsics = get_intrinsics(pipeline)

    # Capture frames
    frames = []
    print(f"\n🎬 Capture {NUM_FRAMES} frames (cách nhau {CAPTURE_INTERVAL}s)")
    print("   ⚡ DI CHUYỂN camera giữa các frame!\n")

    # Warm up
    print("🔄 Warming up...")
    try:
        q_rgb.get()
        q_depth.get()
        time.sleep(0.5)
    except:
        pass

    for i in range(NUM_FRAMES):
        print(f"📸 Frame {i + 1}/{NUM_FRAMES}...")

        try:
            rgb_frame = q_rgb.get().getCvFrame()
            depth_frame = q_depth.get().getFrame()
        except Exception as e:
            print(f"   ❌ Lỗi: {e}")
            continue

        # Depth: mm → meters
        depth_m = depth_frame.astype(np.float32) / 1000.0
        depth_m[depth_m <= 0] = 0
        depth_m[depth_m > 20.0] = 0

        # Resize depth cho khớp với RGB
        if depth_m.shape[:2] != rgb_frame.shape[:2]:
            depth_m = cv2.resize(
                depth_m,
                (rgb_frame.shape[1], rgb_frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        valid = depth_m[depth_m > 0]
        if len(valid) > 0:
            print(
                f"   ✅ Depth: {valid.min():.2f}m - {valid.max():.2f}m ({len(valid)} px)"
            )
        else:
            print("   ⚠️ Không có valid depth")

        frames.append(
            {
                "rgb": rgb_frame.copy(),
                "depth_m": depth_m.copy(),
            }
        )

        # Preview
        try:
            cv2.imshow("OAK-D RGB", rgb_frame)
            d_disp = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX).astype(
                np.uint8
            )
            cv2.imshow("OAK-D Depth", cv2.applyColorMap(d_disp, cv2.COLORMAP_JET))
            cv2.waitKey(500)
        except:
            pass

        if i < NUM_FRAMES - 1:
            print(f"   ⏳ Di chuyển camera! Đợi {CAPTURE_INTERVAL}s...")
            time.sleep(CAPTURE_INTERVAL)

    # Dừng pipeline
    pipeline.stop()
    print("⏹️ Pipeline stopped")
    try:
        cv2.destroyAllWindows()
    except:
        pass

    return frames, intrinsics


# ============================================================
# 2. LẤY INTRINSICS
# ============================================================
def get_intrinsics(pipeline):
    """Lấy intrinsics từ pipeline calibration."""
    try:
        calib_obj = pipeline.getCalibrationData()

        # v3: thử nhiều cách tạo CalibrationHandler
        calib = None
        if isinstance(calib_obj, dai.CalibrationHandler):
            calib = calib_obj
        elif isinstance(calib_obj, dai.EepromData):
            calib = dai.CalibrationHandler(calib_obj)
        elif isinstance(calib_obj, bytes):
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                f.write(calib_obj)
                tmp = f.name
            calib = dai.CalibrationHandler(tmp)
            os.unlink(tmp)
        else:
            calib = calib_obj

        intrinsics = calib.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A, RGB_WIDTH, RGB_HEIGHT
        )
        K = np.array(intrinsics).reshape(3, 3).astype(np.float32)
        print(f"📷 Intrinsics từ calibration:\n{K}")
    except Exception as e:
        print(f"⚠️ Không đọc được intrinsics: {e}")
        print("   Dùng intrinsics mặc định OAK-D-Pro-W")
        K = np.array(
            [[448.0, 0.0, 320.0], [0.0, 448.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
    return K


# ============================================================
# 3. MAPANYTHING INFERENCE
# ============================================================
def run_mapanything(frames, intrinsics):
    """Chạy MapAnything multi-modal inference."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️ Device: {device}")

    print("📥 Đang tải model...")
    model = MapAnything.from_pretrained("facebook/map-anything").to(device)
    print("✅ Model OK!")

    views = []
    for i, f in enumerate(frames):
        rgb = f["rgb"]  # (H, W, 3)
        depth_m = f["depth_m"]  # (H, W) - đã resize khớp RGB

        # Bỏ frame không có depth
        valid_depth = depth_m[depth_m > 0]
        if len(valid_depth) == 0:
            print(f"   View {i}: ⚠️ Không có depth - bỏ qua")
            continue

        dr = f"{valid_depth.min():.2f}-{valid_depth.max():.2f}m"
        print(f"   View {i}: rgb={rgb.shape}, depth={depth_m.shape}, range={dr}")

        view = {
            "img": rgb,
            "intrinsics": torch.tensor(intrinsics, device=device),
            "depth_z": torch.tensor(depth_m, device=device),
            "is_metric_scale": torch.tensor([True], device=device),
        }
        views.append(view)

    if len(views) == 0:
        print("❌ Không có view hợp lệ!")
        return None

    print("🔄 Preprocessing...")
    processed = preprocess_inputs(views)

    print("🚀 Inference...")
    preds = model.infer(
        processed,
        memory_efficient_inference=True,
        minibatch_size=1,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
        apply_confidence_mask=True,
        confidence_percentile=10,
    )
    return preds


# ============================================================
# 4. XUẤT KẾT QUẢ
# ============================================================
def export_results(preds):
    """Xuất point cloud 3D."""
    import trimesh

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_pts, all_clr = [], []

    for i, p in enumerate(preds):
        pts = p["pts3d"][0].cpu().numpy()
        mask = p["mask"][0].cpu().numpy()
        img = p["img_no_norm"][0].cpu().numpy()
        conf = p["conf"][0].cpu().numpy()

        valid = (mask[..., 0] > 0) & (conf > np.percentile(conf, 10))
        pts_v = pts[valid]
        clr_v = img[valid] / 255.0

        if len(pts_v) > 0:
            d = np.linalg.norm(pts_v, axis=1)
            print(
                f"   View {i}: {len(pts_v):,} pts, dist={d.min():.2f}-{d.max():.2f}m, conf={conf[valid].mean():.4f}"
            )

        all_pts.append(pts_v)
        all_clr.append(clr_v)

    all_pts = np.vstack(all_pts)
    all_clr = np.vstack(all_clr)

    pc = trimesh.PointCloud(vertices=all_pts, colors=all_clr)

    ply_path = os.path.join(OUTPUT_DIR, "oakd_reconstruction.ply")
    pc.export(ply_path)
    print(f"\n✅ PLY: {ply_path} ({len(all_pts):,} points)")

    glb_path = os.path.join(OUTPUT_DIR, "oakd_reconstruction.glb")
    pc.export(glb_path)
    print(f"✅ GLB: {glb_path}")

    return ply_path, glb_path


# ============================================================
# 5. IN CHI TIẾT
# ============================================================
def print_details(preds):
    print("\n" + "=" * 60)
    print("📊 KẾT QUẢ MAPANYTHING + OAK-D")
    print("=" * 60)
    for i, p in enumerate(preds):
        print(f"\n📌 View {i}:")
        print(f"   pts3d:          {p['pts3d'].shape}")
        print(f"   depth_z:        {p['depth_z'].shape}")
        print(f"   camera_poses:   {p['camera_poses'].shape}")
        print(f"   intrinsics:     {p['intrinsics'].shape}")
        print(f"   avg confidence: {p['conf'][0].mean().item():.4f}")
        pose = p["camera_poses"][0].cpu().numpy()
        print(
            f"   translation:    [{pose[0, 3]:.3f}, {pose[1, 3]:.3f}, {pose[2, 3]:.3f}]"
        )


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🎥 MapAnything + OAK-D (DepthAI v3.5)")
    print("=" * 60)

    # Capture
    frames, intrinsics = capture_oakd()

    if len(frames) == 0:
        print("❌ Không capture được frame!")
        exit(1)

    # Inference
    preds = run_mapanything(frames, intrinsics)

    if preds is None:
        print("❌ Inference thất bại!")
        exit(1)

    # Chi tiết
    print_details(preds)

    # Xuất file
    ply_path, glb_path = export_results(preds)

    # Mở 3D viewer
    try:
        import open3d as o3d

        print("\n🖼️ Đang mở 3D viewer...")
        pcd = o3d.io.read_point_cloud(ply_path)
        o3d.visualization.draw_geometries([pcd], window_name="MapAnything + OAK-D")
    except:
        print("\n💡 Xem 3D tại: https://3dviewer.net/ (upload .glb)")

    print("\n🎉 Hoàn thành!")
