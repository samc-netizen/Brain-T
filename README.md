# 🧠 Automated Brain Tumor Detection & Localization — YOLOv12

---

## Overview
Real-time detection and bounding-box localisation of brain tumours from MRI scans using **YOLOv12** — the latest YOLO architecture with attention-enhanced backbones (Area Attention + FPN-PAN feature fusion).

---

## Results (Table 6.1 — Project Report)

| Model | Precision | Recall | F1-Score | mAP@0.5 |
|-------|-----------|--------|----------|---------|
| YOLOv8 | 98.7 % | 98.5 % | 98.60 % | 99.4 % |
| YOLOv11 | 98.9 % | 98.8 % | 98.85 % | 99.5 % |
| **YOLOv12** | **99.2 %** | **99.3 %** | **99.25 %** | **99.8 %** |

---

## Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | `glioma` | Most common malignant primary brain tumour |
| 1 | `meningioma` | Arises from meninges, often benign |
| 2 | `pituitary` | Pituitary gland adenoma |
| 3 | `no_tumor` | Healthy / negative scan |

---

## Dataset
- **Source:** Kaggle / Roboflow — Brain Tumor MRI Dataset  
  `https://www.kaggle.com/datasets/pkdarabi/brain-tumor-image-dataset-object-detection`
- **Size:** ~7 000 annotated MRI images  
- **Format:** YOLO bounding-box `.txt` labels  
- **Split:** 70 % train · 20 % val · 10 % test

---

## Tech Stack
`Python 3.11` · `PyTorch 2.0` · `Ultralytics YOLOv12` · `OpenCV` · `Google Colab T4 GPU`

---

## Quick Start

### 1 — Install
```bash
pip install -r requirements.txt
```

### 2 — Train (Google Colab recommended)
Open **`brain_tumor_yolov12_training.ipynb`** in Colab,  
set `DATASET_ROOT` to your Drive path, and run all cells.

### 3 — Inference
```bash
# single image
python predict.py --source mri_scan.jpg

# folder of images
python predict.py --source /path/to/test_images/

# video
python predict.py --source video.mp4 --video
```

---

## File Structure
```
brain-tumor-yolov12/
├── brain_tumor_yolov12_training.ipynb   ← full Colab notebook
├── predict.py                           ← inference / deployment script
├── brain_tumor_data.yaml                ← dataset config
├── requirements.txt
└── README.md
```

---

<div align="center">

Developed as part of academic coursework  
VIT Bhopal University

</div>

