"""
predict.py — Brain Tumor Detection Inference Script
YOLOv12 | VIT Bhopal University | CSE Health Informatics 2025

Usage:
    python predict.py --source path/to/image.jpg
    python predict.py --source path/to/folder/
    python predict.py --source 0          (webcam)
    python predict.py --source video.mp4
"""

import argparse
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_MODEL  = 'brain_tumor_yolov12_best.pt'
CLASS_NAMES    = ['glioma', 'meningioma', 'pituitary', 'no_tumor']
CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45
IMG_SIZE       = 640

CLASS_COLORS = {
    'glioma':     (220, 50,  50),   # Red
    'meningioma': (50,  180, 50),   # Green
    'pituitary':  (50,  100, 220),  # Blue
    'no_tumor':   (200, 200, 50),   # Yellow
}

# ─── Core Inference ──────────────────────────────────────────────────────────

def load_model(model_path: str) -> YOLO:
    """Load YOLOv12 model from weights file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Train the model first using the Colab notebook, "
            "then place 'brain_tumor_yolov12_best.pt' in this directory."
        )
    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Running on: {device.upper()}")
    return model


def predict_single(model: YOLO, image_path: str, save_dir: str = 'predictions') -> dict:
    """
    Run inference on a single MRI image.
    Returns a dict with detections and confidence scores.
    """
    os.makedirs(save_dir, exist_ok=True)

    results = model(
        image_path,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMG_SIZE,
        verbose=False,
    )

    result    = results[0]
    detections = []

    for box in result.boxes:
        cls_id     = int(box.cls.item())
        confidence = float(box.conf.item())
        cls_name   = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f'class_{cls_id}'
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

        detections.append({
            'class':      cls_name,
            'confidence': round(confidence, 4),
            'bbox':       [x1, y1, x2, y2],
        })

    # Save annotated image
    annotated = result.plot()
    out_name  = Path(image_path).stem + '_prediction.jpg'
    out_path  = os.path.join(save_dir, out_name)
    cv2.imwrite(out_path, annotated)

    print(f"\n[RESULT] {Path(image_path).name}")
    print(f"  Detections : {len(detections)}")
    for det in detections:
        print(f"  ├─ {det['class']:<15} conf={det['confidence']:.2%}  bbox={det['bbox']}")
    print(f"  Saved to   : {out_path}")

    return {'image': image_path, 'detections': detections, 'saved': out_path}


def predict_batch(model: YOLO, source: str, save_dir: str = 'predictions') -> list:
    """Run inference on a folder of images."""
    os.makedirs(save_dir, exist_ok=True)
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    if os.path.isdir(source):
        image_paths = [
            os.path.join(source, f)
            for f in sorted(os.listdir(source))
            if f.lower().endswith(extensions)
        ]
    else:
        image_paths = [source]

    print(f"[INFO] Found {len(image_paths)} image(s) to process")
    all_results = []

    for i, img_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] Processing: {Path(img_path).name}")
        result = predict_single(model, img_path, save_dir)
        all_results.append(result)

    # Summary
    total_detections = sum(len(r['detections']) for r in all_results)
    class_summary    = {}
    for r in all_results:
        for det in r['detections']:
            class_summary[det['class']] = class_summary.get(det['class'], 0) + 1

    print("\n" + "="*50)
    print("  BATCH PREDICTION SUMMARY")
    print("="*50)
    print(f"  Images processed : {len(image_paths)}")
    print(f"  Total detections : {total_detections}")
    print(f"  Detections by class:")
    for cls, count in sorted(class_summary.items(), key=lambda x: -x[1]):
        print(f"    {cls:<15}: {count}")
    print(f"  Output directory : {save_dir}/")
    print("="*50)

    return all_results


def predict_video(model: YOLO, video_path: str, save_dir: str = 'predictions') -> str:
    """Run inference on a video file."""
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(0 if video_path == '0' else video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video source: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = os.path.join(save_dir, 'output_video.mp4')
    writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_count = 0
    print(f"[INFO] Processing video: {video_path}")
    print("[INFO] Press 'q' to stop (if showing live preview)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results   = model(frame, conf=CONF_THRESHOLD, verbose=False)
        annotated = results[0].plot()
        writer.write(annotated)
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"  Processed {frame_count} frames...")

    cap.release()
    writer.release()
    print(f"[INFO] Video saved to: {out_path}  ({frame_count} frames)")
    return out_path


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Brain Tumor Detection using YOLOv12',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--source',     type=str, required=True,
                        help='Image path, folder path, video path, or 0 for webcam')
    parser.add_argument('--model',      type=str, default=DEFAULT_MODEL,
                        help=f'Path to model weights (default: {DEFAULT_MODEL})')
    parser.add_argument('--conf',       type=float, default=CONF_THRESHOLD,
                        help=f'Confidence threshold (default: {CONF_THRESHOLD})')
    parser.add_argument('--iou',        type=float, default=IOU_THRESHOLD,
                        help=f'IoU threshold (default: {IOU_THRESHOLD})')
    parser.add_argument('--save-dir',   type=str, default='predictions',
                        help='Directory to save predictions (default: predictions/)')
    parser.add_argument('--video',      action='store_true',
                        help='Force video mode for the source')
    return parser.parse_args()


def main():
    args = parse_args()

    # Update global thresholds from args
    global CONF_THRESHOLD, IOU_THRESHOLD
    CONF_THRESHOLD = args.conf
    IOU_THRESHOLD  = args.iou

    model = load_model(args.model)

    # Determine source type
    source = args.source
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')

    if args.video or source == '0' or source.lower().endswith(video_extensions):
        predict_video(model, source, args.save_dir)
    elif os.path.isdir(source):
        predict_batch(model, source, args.save_dir)
    elif os.path.isfile(source):
        predict_single(model, source, args.save_dir)
    else:
        print(f"[ERROR] Source not found: {source}")


if __name__ == '__main__':
    main()
