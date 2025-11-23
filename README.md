# Waste Object Detection and Segmentation

**Deep Learning for Perception (CS4045) - Semester Project Part 1**

A comprehensive implementation of object detection (YOLOv8) and semantic segmentation (U-Net) for automated waste classification using the TACO dataset.

---

## 🎯 Project Overview

This project implements two complementary deep learning approaches for waste detection:

1. **YOLOv8 Object Detection** - Identifies and localizes waste items with bounding boxes
2. **Custom U-Net Segmentation** - Classifies each pixel as background or one of 5 waste categories

### Key Results

| Model | Task | Precision | Recall | mAP@50 | mAP@50-95 | Improvement |
|-------|------|-----------|--------|--------|-----------|-------------|
| YOLOv8 (No Aug) | Detection | 17.4% | 15.7% | **10.1%** | 6.7% | Baseline |
| YOLOv8 (With Aug) | Detection | 26.8% | 22.3% | **18.0%** | 12.3% | **+7.9% mAP@50** |
| Custom U-Net | Segmentation | - | - | - | - | See notebook |

**Key Achievement:** 78% relative improvement in mAP@50 with data augmentation (+7.9 percentage points)

---

## 📁 Project Structure

```
Waste-Detector_DeepLearningProject/
├── Final_Project_DL.ipynb                    # Main project notebook (137MB)
│                                             # ⚠️ Too large for GitHub (see Colab link below)
│
├── Deep Learning for Perception-Semester Project.pdf  # Assignment requirements
│
├── yolo_results/                             # Complete YOLO training outputs
│   └── detect/
│       ├── yolo_no_aug_v2/                  # Training without augmentation (49 epochs)
│       │   ├── weights/best.pt              # Best model checkpoint
│       │   ├── results.csv                  # Training metrics per epoch
│       │   ├── results.png                  # Training curves
│       │   ├── confusion_matrix.png         # Validation confusion matrix
│       │   ├── BoxP_curve.png              # Precision curve
│       │   ├── BoxR_curve.png              # Recall curve
│       │   ├── BoxF1_curve.png             # F1 score curve
│       │   ├── BoxPR_curve.png             # Precision-Recall curve
│       │   ├── labels.jpg                   # Dataset label distribution
│       │   ├── train_batch*.jpg            # Training batch samples
│       │   └── val_batch*.jpg              # Validation predictions
│       │
│       ├── yolo_with_aug_v2/               # Training WITH augmentation (99 epochs)
│       │   ├── weights/best.pt             # Best model checkpoint
│       │   ├── results.csv                 # Training metrics per epoch
│       │   ├── results.png                 # Training curves
│       │   ├── confusion_matrix.png        # Validation confusion matrix
│       │   ├── BoxP_curve.png             # Precision curve
│       │   ├── BoxR_curve.png             # Recall curve
│       │   ├── BoxF1_curve.png            # F1 score curve
│       │   ├── BoxPR_curve.png            # Precision-Recall curve
│       │   ├── labels.jpg                  # Dataset label distribution
│       │   ├── train_batch*.jpg           # Training batch samples
│       │   └── val_batch*.jpg             # Validation predictions
│       │
│       ├── val/                            # Additional validation visualizations
│       │   ├── confusion_matrix.png
│       │   └── val_batch*.jpg
│       │
│       └── val2/                           # Second validation run
│           ├── confusion_matrix.png
│           └── val_batch*.jpg
│
├── unet_taco_final.pth                      # Trained U-Net model weights
│
├── README.md                                # This file
├── .gitignore                               # Git ignore rules
└── .gitattributes                           # Git LFS configuration

```

---

## 📓 Access the Main Notebook

**⚠️ Important:** The main project notebook `Final_Project_DL.ipynb` is **137.5 MB** and exceeds GitHub's 100MB file size limit.

### Option 1: Google Colab (Recommended)
- **Open in Colab**: [🔗 Add your Colab link here after uploading]
- Upload the notebook to Google Drive and open with Google Colaboratory
- All code is ready to run in Colab environment

### Option 2: Local Execution
- Clone this repository
- The notebook is excluded from Git but available locally if you have the original file
- Requires CUDA-capable GPU with 11+ GB VRAM

---

## 🗂️ Dataset Information

### TACO (Trash Annotations in Context)

- **Source:** [Kaggle TACO Dataset](https://www.kaggle.com/datasets/kneroma/tacotrashdataset)
- **Original Size:** 1,500 images, 60 classes, 4,784 annotations
- **Filtered Size:** 861 images, 5 classes, 2,193 annotations
- **Format:** COCO JSON → Converted to YOLO format
- **Split:** 80/20 train/validation (688 train, 173 val)

### Top 5 Selected Classes

| Class ID | Class Name | Annotation Count |
|----------|------------|------------------|
| 59 | Cigarette | 667 |
| 58 | Unlabeled litter | 517 |
| 36 | Plastic film | 451 |
| 5 | Clear plastic bottle | 285 |
| 29 | Other plastic | 273 |

*Classes were selected based on annotation frequency to ensure sufficient training data*

---

## 🚀 Quick Start

### Prerequisites

```bash
# System Requirements
- Python 3.8+
- CUDA-capable GPU (Tesla T4 or better recommended)
- 10 GB free disk space
- 11+ GB VRAM for training
```

### Installation

```bash
# Clone the repository
git clone https://github.com/MoazzamHafeez1093/Waste-Detector_DeepLearningProject.git
cd Waste-Detector_DeepLearningProject

# Install dependencies
pip install ultralytics torch torchvision opencv-python numpy pandas matplotlib scikit-learn pycocotools albumentations jupyter

# Download TACO dataset (requires Kaggle API)
kaggle datasets download -d kneroma/tacotrashdataset
unzip tacotrashdataset.zip -d data/taco_data
```

### Running the Project

**On Google Colab:**
1. Upload `Final_Project_DL.ipynb` to Google Drive
2. Open with Google Colaboratory
3. Run all cells sequentially
4. Training takes ~2-3 hours on Tesla T4

---

## 🔬 Experimental Setup

### YOLOv8 Configuration

**Model:** YOLOv8-nano (3.01M parameters)

**Training Hyperparameters:**
- **Without Augmentation:**
  - Epochs: 50 (stopped at 49)
  - Batch size: 16
  - Image size: 640×640
  - Optimizer: AdamW
  - Learning rate: 0.001111 (initial)
  - Augmentation: `augment=False`

- **With Augmentation:**
  - Epochs: 100 (stopped at 99)
  - Batch size: 16
  - Image size: 640×640
  - Optimizer: AdamW
  - Learning rate: 0.001111 (initial)
  - Augmentation: `augment=True` (default YOLOv8 augmentations)
    - Mosaic augmentation
    - HSV color jittering
    - Horizontal flips
    - Translation
    - Scaling

**Evaluation Metrics:**
- Precision (P)
- Recall (R)
- mAP@50 (mean Average Precision at IoU=0.50)
- mAP@50-95 (mean Average Precision averaged over IoU=0.50:0.95)

### U-Net Configuration

**Architecture:** Custom U-Net with 4 encoder and 4 decoder blocks

**Training Hyperparameters:**
- Epochs: 20
- Batch size: 8
- Image size: 256×256
- Optimizer: Adam (lr=0.001)
- Loss: Weighted CrossEntropyLoss (class weights applied)

**Evaluation Metrics:**
- Mean IoU (Intersection over Union)
- Dice Score
- Pixel Accuracy

---

## 📊 Results Summary

### YOLOv8 Detection Performance

#### Final Model Comparison

| Metric | Without Augmentation (Epoch 49) | With Augmentation (Epoch 99) | Absolute Gain | Relative Improvement |
|--------|--------------------------------|------------------------------|---------------|---------------------|
| **Precision** | 0.174 (17.4%) | 0.268 (26.8%) | +9.4% | **+54%** |
| **Recall** | 0.157 (15.7%) | 0.223 (22.3%) | +6.6% | **+42%** |
| **mAP@50** | 0.101 (10.1%) | **0.180 (18.0%)** | **+7.9%** | **+78%** |
| **mAP@50-95** | 0.067 (6.7%) | 0.123 (12.3%) | +5.6% | **+84%** |

**Key Insight:** Data augmentation provided substantial improvements across all metrics, with mAP@50 nearly doubling.

#### Training Dynamics

- **Without Augmentation:** Converged in 49 epochs (~40 minutes)
- **With Augmentation:** Required 99 epochs (~90 minutes) but achieved significantly better performance
- Both models showed stable training with no signs of catastrophic forgetting

### Per-Class Performance Analysis

*Detailed per-class metrics available in training visualizations (`yolo_results/detect/*/confusion_matrix.png`)*

### U-Net Segmentation Performance

- Model weights saved in `unet_taco_final.pth`
- Detailed metrics and visualizations available in the main notebook
- Pixel-level classification for fine-grained waste boundary detection

---

## 🛠️ Technologies & Dependencies

### Core Libraries

```bash
ultralytics==8.3.229     # YOLOv8 implementation
torch==2.8.0             # PyTorch deep learning framework
torchvision              # Computer vision utilities
opencv-python            # Image processing
numpy                    # Numerical computing
pandas                   # Data manipulation
matplotlib               # Visualization
scikit-learn             # Train/val split
pycocotools              # COCO format handling
albumentations           # Advanced data augmentation
jupyter                  # Notebook environment
```

### Installation Command

```bash
pip install ultralytics torch torchvision opencv-python numpy pandas matplotlib scikit-learn pycocotools albumentations jupyter
```

---

## 🔍 Key Insights

### 1. Data Augmentation Impact

✅ **Highly Effective:** +7.9% absolute mAP@50 improvement (78% relative increase)

**Why it works for this dataset:**
- Small dataset (861 images) benefits greatly from synthetic data
- Waste items appear in various orientations → rotation/flipping helps
- Lighting conditions vary → HSV augmentation improves robustness
- Reduces overfitting by increasing effective dataset size
- Mosaic augmentation helps with object scale variation

**Trade-off:** Longer training time (99 vs 49 epochs) but worth the performance gain

### 2. Detection vs. Segmentation Trade-offs

| Aspect | YOLOv8 (Detection) | U-Net (Segmentation) |
|--------|-------------------|---------------------|
| **Speed** | ✅ Fast (real-time capable) | ❌ Slower (pixel-wise) |
| **Annotation Cost** | ✅ Simple bounding boxes | ❌ Polygon masks required |
| **Output** | Object location + class | Precise pixel boundaries |
| **Use Case** | Counting, localization | Shape analysis, area measurement |
| **Overlap Handling** | ✅ Better with NMS | ❌ Struggles with occlusion |
| **Deployment** | ✅ Edge-friendly | Requires more compute |

### 3. Challenges Encountered

1. **Class Imbalance:**
   - Top class (Cigarette: 667) vs. bottom class (Other plastic: 273)
   - Weighted loss helped but some classes still underperformed

2. **Small Objects:**
   - Cigarettes are tiny and often hard to detect
   - YOLOv8-nano may lack capacity for small object detection
   - Larger models (YOLOv8-small/medium) could improve this

3. **Dataset Size:**
   - 861 images is small for deep learning standards
   - Augmentation partially compensates but more data would help
   - Consider synthetic data generation or web scraping

4. **Annotation Quality:**
   - "Unlabeled litter" is ambiguous and affects model learning
   - Some images have crowded scenes with many overlapping objects

---

## 🌍 Real-World Applications

### Potential Deployment Scenarios

1. **Automated Waste Sorting Facilities**
   - Conveyor belt systems with overhead cameras
   - Real-time classification at 30+ FPS with YOLOv8
   - Integration with robotic arms for physical sorting

2. **Smart City Infrastructure**
   - IoT-enabled waste bins with edge AI (Jetson Nano, Raspberry Pi)
   - Real-time monitoring of contamination rates
   - Data-driven collection route optimization

3. **Environmental Monitoring**
   - Drone-based litter detection in parks, beaches, rivers
   - Mobile apps for citizen science waste reporting
   - Gamification to encourage community participation

4. **Recycling Optimization**
   - Quality control at material recovery facilities
   - Contamination detection in recycling streams
   - Data analytics for waste composition insights

---

## 🚧 Future Improvements

### Short-term Enhancements
- [ ] Experiment with YOLOv8-small/medium for better small object detection
- [ ] Implement class-weighted sampling during training
- [ ] Try advanced augmentation (CutMix, MixUp)
- [ ] Ensemble multiple model predictions

### Medium-term Goals
- [ ] Expand dataset to 5,000+ images via web scraping
- [ ] Balance class distribution with targeted data collection
- [ ] Explore transformer-based detectors (DETR, Swin Transformer)
- [ ] Implement active learning for efficient labeling

### Long-term Vision
- [ ] Model quantization (INT8/FP16) for edge deployment
- [ ] ONNX/TensorRT export for production inference
- [ ] Multi-camera 3D reconstruction and pose estimation
- [ ] Integration with robotic manipulation systems
- [ ] Temporal modeling with video sequences

---

## 📝 Course Information

- **Course:** Deep Learning for Perception (CS4045)
- **Institution:** [Your University]
- **Instructor:** Dr. Ahmad Raza Shahid
- **Project:** Part 1 - Waste Object Detection and Segmentation
- **Semester:** [Your Semester/Year]

---

## 👥 Contributors

- **Moazzam Hafeez** - Implementation, training, evaluation, report writing
- GitHub: [@MoazzamHafeez1093](https://github.com/MoazzamHafeez1093)

---

## 📄 License

This project is developed for academic purposes as part of the CS4045 course requirements. The code is available for educational use.

---

## 🙏 Acknowledgments

- **TACO Dataset:** Proença, P. F., & Simões, P. (2020). "TACO: Trash Annotations in Context for Litter Detection"
- **YOLOv8:** Ultralytics team (Glenn Jocher et al.) for the excellent object detection framework
- **U-Net:** Ronneberger, O., Fischer, P., & Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- **Instructor:** Dr. Ahmad Raza Shahid for course guidance and project supervision
- **Community:** Kaggle and PyTorch communities for resources and support

---

## 📚 References

1. Proença, P. F., & Simões, P. (2020). TACO: Trash Annotations in Context for Litter Detection. arXiv preprint arXiv:2003.06975.
2. Jocher, G., et al. (2023). Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics
3. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015.
4. Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement. arXiv preprint arXiv:1804.02767.

---

**⭐ If you found this project helpful, please consider starring this repository!**

---

*Last Updated: November 23, 2025*
