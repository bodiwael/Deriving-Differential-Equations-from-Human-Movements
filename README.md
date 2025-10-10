# 🏋️ Human Exercise Classification System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Approach 1: 92.5%](https://img.shields.io/badge/Approach%201-92.5%25-success)](./Approach-1-IMU-Based/)
[![Approach 2: 100%](https://img.shields.io/badge/Approach%202-100%25-success)](./Approach-2-Video-Based/)

A comprehensive machine learning system for classifying human exercises using two distinct approaches: **IMU sensor-based** and **Video pose estimation-based** methods.

---

## 🎯 **Project Overview**

This project implements and compares two state-of-the-art approaches for automated exercise classification:

| Approach | Method | Accuracy | Use Case |
|----------|--------|----------|----------|
| **Approach 1** | IMU Sensors + SINDy | **92.5%** | Wearable devices, real-time tracking |
| **Approach 2** | Video + MediaPipe Pose | **100%** | Camera-based, gym monitoring |

### **Exercises Classified:**
- 🏋️ **Squats** (SQ)
- 🦵 **Lunges** (LU)
- 💪 **Push-ups** (PU)
- 🙌 **Shoulder Press** (SHP)

---

## 📊 **Results Summary**

### **Approach 1: IMU-Based (Wearable Sensors)**
- **Accuracy:** 92.5%
- **Method:** SINDy (Sparse Identification of Nonlinear Dynamics) + Random Forest
- **Sensors:** 9 IMU sensors (LThigh, RThigh, back, LUA, RUA, etc.)
- **Dataset:** 808 exercise repetitions from 30 subjects
- **Best for:** Real-time wearable applications, privacy-preserving scenarios

### **Approach 2: Video-Based (Computer Vision)**
- **Accuracy:** 100%
- **Method:** MediaPipe Pose + Statistical Features + Random Forest
- **Input:** RGB video (front/side views)
- **Dataset:** 390 videos from ~40 subjects
- **Best for:** Gym monitoring, form analysis, telemonitoring

---

## 🗂️ **Repository Structure**

```
exercise-classification/
│
├── Approach-1-IMU-Based/           # 🔴 IMU Sensor-Based Classification
│   ├── data/
│   │   └── README.md               # Data structure & format
│   ├── scripts/
│   │   ├── 1_main.py              # Data synchronization
│   │   ├── 2_process.py           # Feature computation
│   │   ├── 3_plot.py              # Visualization
│   │   ├── 4_repetitions.py       # Segmentation
│   │   └── 5_final_training.py    # Model training
│   ├── results/
│   │   ├── confusion_matrix.png
│   │   └── classification_report.png
│   ├── requirements.txt
│   └── README.md                   # Approach 1 detailed docs
│
├── Approach-2-Video-Based/         # 🔵 Video Pose-Based Classification
│   ├── data/
│   │   └── README.md               # Dataset organization
│   ├── scripts/
│   │   ├── 1_keypoint_extraction.py
│   │   ├── 2_visualization_analysis.py
│   │   ├── 3_model_training.py
│   │   ├── 4_fully_connected_nn.py
│   │   └── 5_inference.py
│   ├── results/
│   │   ├── paper_visualizations/
│   │   ├── model_results/
│   │   └── comparison_plots/
│   ├── requirements.txt
│   └── README.md                   # Approach 2 detailed docs
│
├── docs/
│   ├── comparison.md               # Approach 1 vs Approach 2
│   ├── deployment.md               # Production deployment guide
│   └── paper.md                    # Research paper draft
│
├── LICENSE
└── README.md                       # This file
```

---

## 🚀 **Quick Start**

### **Option 1: IMU Sensor-Based Classification**

```bash
cd Approach-1-IMU-Based/

# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python scripts/1_main.py          # Sync sensor data
python scripts/2_process.py       # Compute features
python scripts/4_repetitions.py   # Segment repetitions
python scripts/5_final_training.py # Train classifier

# Expected output: 92.5% accuracy
```

### **Option 2: Video Pose-Based Classification**

```bash
cd Approach-2-Video-Based/

# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python scripts/1_keypoint_extraction.py      # Extract pose keypoints
python scripts/2_visualization_analysis.py   # Generate visualizations
python scripts/3_model_training.py           # Train models

# Expected output: 100% accuracy
```

---

## 📈 **Performance Comparison**

### **Accuracy per Exercise:**

| Exercise | Approach 1 (IMU) | Approach 2 (Video) |
|----------|------------------|-------------------|
| Squats | 91.2% | 100% |
| Lunges | 88.7% | 100% |
| Push-ups | 95.6% | 100% |
| Shoulder Press | 94.5% | 100% |
| **Overall** | **92.5%** | **100%** |

### **Confusion Matrices:**

#### **Approach 1 (IMU)**
```
                Predicted
            PU   Lunges  SQ   SHP
Actual  PU  153    21     7    4
      Lung   14   210    11    3
        SQ   18    61    108   3
       SHP    6     3     0   186
```

#### **Approach 2 (Video)**
```
                Predicted
            Squat Lunges PU  SHP
Actual Squat  14    0    0   0
      Lunges   0   15    0   0
          PU   0    0   15   0
         SHP   0    0    0  15

Perfect diagonal!
```

---

## 🔬 **Methodology**

### **Approach 1: IMU-Based**

**Pipeline:**
1. **Data Collection:** 9 IMU sensors on body joints
2. **Synchronization:** Align timestamps across sensors
3. **Feature Engineering:** 
   - Angular velocity → displacement (integration)
   - Polynomial feature library (degree 3)
   - SINDy sparse regression for dynamics
4. **Classification:** Random Forest on learned coefficients

**Key Innovation:** Using SINDy to learn governing equations of motion from raw IMU data.

### **Approach 2: Video-Based**

**Pipeline:**
1. **Pose Estimation:** MediaPipe extracts 33 body landmarks
2. **Feature Engineering:**
   - 12 key landmarks (shoulders, elbows, wrists, hips, knees, ankles)
   - 8 joint angles (knee, hip, elbow, shoulder)
   - Statistical aggregation (mean, std, min, max, range)
   - Velocity features
3. **Classification:** Multiple models (RF, GB, SVM, NN)

**Key Innovation:** Person-based validation ensures generalization to unseen subjects.

---

## 📊 **Datasets**

### **Approach 1: EJUST-GYM IMU Dataset**
- **Size:** 808 exercise repetitions
- **Subjects:** 30 unique individuals
- **Sensors:** 9 × Xsens MTw IMU sensors
- **Sampling Rate:** ~60 Hz
- **Duration:** ~2.5 seconds per repetition

### **Approach 2: Custom Video Dataset**
- **Size:** 390 videos (97-98 per exercise)
- **Subjects:** ~40 unique individuals
- **Views:** Front + Side (synchronized)
- **Resolution:** 1920×1080, 30 fps
- **Duration:** 300-700 frames per video

---

## 🛠️ **Technologies Used**

### **Common:**
- Python 3.8+
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn

### **Approach 1 Specific:**
- SciPy (signal processing)
- Lasso regression (sparse identification)

### **Approach 2 Specific:**
- OpenCV (video processing)
- MediaPipe (pose estimation)
- TensorFlow/Keras (neural networks)

---

## 📝 **Installation**

### **System Requirements:**
- Python 3.8 or higher
- 8GB RAM (minimum)
- GPU (optional, for Approach 2 neural networks)

### **Install All Dependencies:**

```bash
# Clone repository
git clone https://github.com/yourusername/exercise-classification.git
cd exercise-classification

# Approach 1
cd Approach-1-IMU-Based
pip install -r requirements.txt

# Approach 2
cd ../Approach-2-Video-Based
pip install -r requirements.txt
```

---

## 🎓 **Citation**

If you use this code in your research, please cite:

```bibtex
@article{exercise_classification_2025,
  title={Comparative Study of IMU and Video-Based Exercise Classification: 
         SINDy vs MediaPipe Approaches},
  author={Your Name},
  journal={Your Journal},
  year={2025},
  note={GitHub: https://github.com/yourusername/exercise-classification}
}
```


---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📚 **Documentation Index**

- [Approach 1: IMU-Based Classification](./Approach-1-IMU-Based/README.md)
- [Approach 2: Video-Based Classification](./Approach-2-Video-Based/README.md)
- [Performance Comparison](./docs/comparison.md)
- [Deployment Guide](./docs/deployment.md)
- [Research Paper Draft](./docs/paper.md)

---

## 🔮 **Future Work**

- [ ] Real-time classification system
- [ ] Mobile app deployment
- [ ] Form quality assessment
- [ ] Repetition counting
- [ ] Multi-person tracking
- [ ] Additional exercise types
- [ ] Transfer learning across datasets
- [ ] Hybrid IMU + Video approach