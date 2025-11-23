# Waste Detection using Deep Learning

## Project Overview
This project implements **waste object detection** using deep learning techniques as part of the Deep Learning for Perception course (CS4045). The project focuses on detecting and classifying waste items from real-world scenes using the TACO (Trash Annotations in Context) dataset.

## 📋 Project Objectives
- Perform comprehensive Exploratory Data Analysis (EDA) on the TACO dataset
- Implement object detection using **YOLOv8** (nano/small variants)
- Implement semantic segmentation using **U-Net**
- Compare model performance with and without data augmentation
- Analyze results and discuss practical applications for environmental sustainability

## 🗂️ Dataset
**TACO (Trash Annotations in Context)**
- 1,500+ real-world waste images
- 60 labeled classes (plastic, paper, metal, glass, etc.)
- COCO-format bounding boxes and segmentation masks
- **Subset used:** Top 5 most frequent classes
  - Clear plastic bottle
  - Other plastic
  - Plastic film
  - Unlabeled litter
  - Cigarette

**Dataset Link:** [TACO on Kaggle](https://www.kaggle.com/datasets/kneroma/tacotrashdataset)

## 🏗️ Project Structure
```
Waste-Detector_DeepLearningProject/
├── Final_Project_DL.ipynb          # Main project notebook (137MB - see Google Colab link below)
├── Deep Learning for Perception-Semester Project.pdf  # Project requirements
├── yolo_results/                   # YOLO training results and visualizations
│   └── detect/
│       ├── yolo_no_aug_v2/        # Results without augmentation
│       ├── yolo_with_aug_v2/      # Results with augmentation
│       ├── val/                    # Validation predictions
│       └── val2/                   # Additional validation
├── unet_taco_final.pth            # Trained U-Net model weights (not included - too large)
├── README.md                       # This file
└── .gitignore                      # Git ignore rules
```

## 📓 Access the Main Notebook
Due to GitHub's 100MB file size limit, the main project notebook is available on Google Colab:
- **Open in Colab**: [Add your Colab link here after uploading]
- **File**: `Final_Project_DL.ipynb` (137.5 MB)
- The notebook is also available in the local repository for running on Google Colab

## 🚀 Implementation Details

### 1. Data Preprocessing
- Filtered TACO dataset to top 5 most frequent classes
- Converted COCO format to YOLO format
- Created train/validation split (80/20)
- Final dataset: 861 images with 2,193 annotations
  - Training: 688 images
  - Validation: 173 images

### 2. YOLOv8 Object Detection
**Models Trained:**
- YOLOv8-nano without augmentation
- YOLOv8-nano with augmentation

**Training Configuration:**
- Epochs: 50 (with early stopping)
- Image size: 640×640
- Batch size: 16
- Patience: 10 epochs

**Data Augmentation Techniques:**
- Horizontal/vertical flips
- Color jitter
- Brightness/contrast adjustments
- Mosaic augmentation

### 3. U-Net Semantic Segmentation
- Custom U-Net architecture built in PyTorch
- Pixel-wise segmentation for waste classification
- Evaluated using IoU (Intersection over Union) and Dice Score

## 📊 Results

### YOLOv8 Performance Metrics
| Model | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|-----------|--------|--------|-----------|
| Without Augmentation | 0.2512 | 0.2343 | 0.1891 | 0.1316 |
| With Augmentation | 0.2512 | 0.2343 | 0.1891 | 0.1316 |

### U-Net Segmentation Results
*(Metrics available in the notebook)*

## 🛠️ Technologies Used
- **Python 3.x**
- **PyTorch** - Deep learning framework
- **Ultralytics YOLOv8** - Object detection
- **OpenCV** - Image processing
- **Matplotlib & Seaborn** - Visualization
- **Albumentations** - Data augmentation
- **Kaggle API** - Dataset download

## 📦 Installation & Setup

### Requirements
```bash
pip install ultralytics
pip install torch torchvision
pip install opencv-python
pip install albumentations
pip install matplotlib seaborn
pip install kaggle
```

### Running on Google Colab
1. Upload your Kaggle API token (`kaggle.json`)
2. Open `Final_Project_DL.ipynb` in Google Colab
3. Run cells sequentially
4. Models will train and save results automatically

## 🎯 Key Findings
- Successfully implemented end-to-end waste detection pipeline
- Compared YOLO (object-level) vs U-Net (pixel-level) approaches
- Data augmentation impact analyzed on model performance
- Challenges addressed: class imbalance, object overlap, dataset noise

## 📈 Future Improvements
- Test larger YOLO models (YOLOv8-small, medium)
- Implement additional augmentation strategies
- Explore ensemble methods
- Deploy model for real-time inference
- Integrate with IoT devices for smart waste management

## 👥 Team Members
- Moazzam Hafeez

## 📝 Course Information
- **Course:** Deep Learning for Perception (CS4045)
- **Instructor:** Dr. Ahmad Raza Shahid
- **Project:** Part 1 - Waste Object Detection and Segmentation

## 📄 License
This project is part of academic coursework.

## 🙏 Acknowledgments
- TACO Dataset creators
- Ultralytics YOLOv8 team
- Course instructor and teaching assistants

---
**Note:** This project is for educational purposes as part of the Deep Learning for Perception course requirements.
