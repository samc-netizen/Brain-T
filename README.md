<div align="center">

# Automated Brain Tumor Detection & Localization

### YOLOv12-Based Real-Time MRI Analysis

![Python](https://img.shields.io/badge/Language-Python-blue?style=for-the-badge&logo=python&logoColor=white)
![YOLOv12](https://img.shields.io/badge/Model-YOLOv12-orange?style=for-the-badge&logo=pytorch&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-skyblue?style=for-the-badge&logo=googlecolab&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)

---

| Author |
|--------|
| Sumit Kumar Chandwani |

</div>

---

## About This Project

This project implements real-time detection and bounding-box localisation of brain tumours from MRI scans using **YOLOv12** — the latest YOLO architecture featuring attention-enhanced backbones (Area Attention + FPN-PAN feature fusion).

The system classifies scans into four categories and draws precise bounding boxes around detected tumour regions, enabling fast and accurate diagnostic support.

---

## Problem Statement

Manual analysis of brain MRI scans is time-consuming, requires specialist expertise, and is prone to human error. Early and accurate detection of brain tumours is critical for patient outcomes, yet existing automated tools often lack the speed or accuracy required for practical clinical assistance.

---

## Proposed Solution

This project provides a deep-learning pipeline that allows users to:

- Train a YOLOv12 model on annotated MRI datasets
- Detect and localise tumours with bounding boxes in real time
- Classify tumours into four clinical categories
- Run inference on single images, folders, or video streams
- Benchmark performance across YOLOv8, YOLOv11, and YOLOv12

---

## Key Features

- Real-time bounding-box detection on MRI images
- Four-class tumour classification (glioma, meningioma, pituitary, no tumour)
- State-of-the-art YOLOv12 backbone with Area Attention mechanism
- Google Colab training notebook with GPU acceleration
- Supports image, folder, and video inference modes
- Comparative benchmarking against YOLOv8 and YOLOv11

---

## Results (Table 6.1 — Project Report)

| Model | Precision | Recall | F1-Score | mAP@0.5 |
|-------|-----------|--------|----------|---------|
| YOLOv8 | 98.7 % | 98.5 % | 98.60 % | 99.4 % |
| YOLOv11 | 98.9 % | 98.8 % | 98.85 % | 99.5 % |
| **YOLOv12** | **99.2 %** | **99.3 %** | **99.25 %** | **99.8 %** |

YOLOv12 outperforms both prior versions across all metrics, achieving near-perfect mAP@0.5 of **99.8 %**.

---

## Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | `glioma` | Most common malignant primary brain tumour |
| 1 | `meningioma` | Arises from meninges; often benign |
| 2 | `pituitary` | Pituitary gland adenoma |
| 3 | `no_tumor` | Healthy / negative scan |

---

## Dataset

- **Source:** Kaggle — Brain Tumor MRI Dataset (Object Detection)
  `https://www.kaggle.com/datasets/pkdarabi/brain-tumor-image-dataset-object-detection`
- **Size:** ~7,000 annotated MRI images
- **Format:** YOLO bounding-box `.txt` labels
- **Split:** 70 % train · 20 % val · 10 % test

---

## Project Structure

```
brain-tumor-yolov12/
│
├── brain_tumor_yolov12_training.ipynb   → Full Colab training notebook
├── predict.py                           → Inference / deployment script
├── brain_tumor_data.yaml                → Dataset configuration
├── requirements.txt                     → Python dependencies
└── README.md                            → Project documentation
```

---

## System Design

### Architecture

- **Backbone** → YOLOv12 with Area Attention modules
- **Neck** → FPN-PAN feature pyramid fusion
- **Head** → Decoupled detection head for classification + localisation

### Flow of Execution

```
MRI Input → Preprocessing → YOLOv12 Inference → Bounding Box Output → Class Label
```

---

## Technologies Used

- Python 3.11
- PyTorch 2.0
- Ultralytics YOLOv12
- OpenCV
- Google Colab (T4 GPU)

---

## How to Run

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train (Google Colab recommended)

Open `brain_tumor_yolov12_training.ipynb` in Colab, set `DATASET_ROOT` to your Drive path, and run all cells.

### Step 3: Run inference

```bash
# Single image
python predict.py --source mri_scan.jpg

# Folder of images
python predict.py --source /path/to/test_images/

# Video
python predict.py --source video.mp4 --video
```

---

## Sample Output

```
Detected: glioma       — Confidence: 0.97  — BBox: [142, 88, 310, 265]
Detected: no_tumor     — Confidence: 0.99  — BBox: [  0,  0,   0,   0]
Detected: meningioma   — Confidence: 0.95  — BBox: [ 78, 55, 220, 198]
```

---

## Challenges Faced

- Sourcing and preprocessing a high-quality annotated MRI dataset
- Tuning anchor sizes and augmentation parameters for medical imagery
- Balancing class distribution across glioma, meningioma, and pituitary samples
- Validating model generalisation beyond the training distribution

---

## Learning Outcomes

- Practical implementation of state-of-the-art object detection (YOLOv12)
- Understanding of attention mechanisms and feature pyramid networks
- Experience with medical image preprocessing and annotation formats
- Model benchmarking and comparative evaluation methodology

---

## Future Enhancements

- Web-based inference interface using Gradio or Streamlit
- DICOM file format support for direct hospital system integration
- Grad-CAM visualisation for model explainability
- Multi-modal fusion with patient metadata for improved diagnosis

---

## Conclusion

This project demonstrates how YOLOv12's attention-enhanced architecture can achieve near-perfect accuracy on brain tumour MRI detection, surpassing both YOLOv8 and YOLOv11 benchmarks. It provides a scalable foundation for real-world clinical decision-support tools.

---

<div align="center">
Developed as part of academic coursework<br>
VIT Bhopal University
</div>
