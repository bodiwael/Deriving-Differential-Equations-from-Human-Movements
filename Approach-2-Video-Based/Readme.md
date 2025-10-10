# 🔵 Video Pose-Based Exercise Classification

**Accuracy: 100%** | **Method: MediaPipe + Random Forest** | **Camera-Based**

---

## 🎯 **Overview**

Computer vision system that classifies exercises from video using **MediaPipe Pose** and machine learning. No wearables needed - just a camera.

### **Key Features:**
- ✅ Works with any camera (phone, webcam)
- ✅ Supports front and side views
- ✅ 100% accuracy on 4 exercises
- ✅ Real-time capable (30 FPS)

### **Pipeline:**
```
Video → MediaPipe Pose → 33 Landmarks → Statistical Features → Random Forest → Exercise Class
```

---

## 📚 **Dataset**

- **390 videos** from ~40 subjects
- **4 exercises:** Squats, Lunges, Push-ups, Shoulder Press
- **2 views:** Front + Side (synchronized)
- **Resolution:** 1920×1080 @ 30 FPS
- **Duration:** 300-700 frames per video

---

## 🛠️ **Installation**

```bash
cd Approach-2-Video-Based/
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, opencv-python, mediapipe, numpy, pandas, scikit-learn, matplotlib

---

## 🚀 **3-Notebook Workflow**

### **Notebook 1: Landmark Extraction** 📍
**File:** `1_keypoint_extraction.ipynb`

**Purpose:** Extract pose landmarks from all videos

**Logic:**
1. Load videos from `videos/` folder
2. Run MediaPipe Pose on each frame
3. Extract 33 body landmarks (x, y, z, visibility)
4. Save to CSV files

**Input:** 
- Video files: `{person}_{exercise}_{view}.mp4`
- Example: `ahmed_squat_side.mp4`

**Output:**
```
extracted_keypoints/
├── side/
│   ├── Squad/
│   │   └── ahmed_squat_side_keypoints.csv
│   ├── Lunges/
│   ├── PushUp/
│   └── SHP/
└── front/
    └── ... (same structure)
```

**Time:** ~10-20 seconds per video

---

### **Notebook 2: Side View Analysis & Training** 👀
**File:** `2_side_view_training.ipynb`

**Purpose:** Train classifier on side view videos

**Logic:**

**Part A: Feature Engineering**
1. Load side view keypoint CSVs
2. Select 12 key landmarks (shoulders, elbows, wrists, hips, knees, ankles)
3. Calculate 8 joint angles (knee, hip, elbow, shoulder)
4. Extract statistical features:
   - Position stats: mean, std, min, max, range
   - Angle stats: mean, std, min, max, range
   - Velocity stats: mean, std, max
5. Total: **176 features per video**

**Part B: Visualization**
- Landmark trajectories
- Joint angle time series
- Exercise signatures
- Correlation heatmaps

**Part C: Model Training**
1. Split data: 70% train, 15% validation, 15% test
2. Train Random Forest, Gradient Boosting, SVM
3. 5-fold cross-validation
4. Save best model

**Results:**
```
Test Accuracy: 100.00%
5-Fold CV: 100.00% ± 0.00%

Confusion Matrix:
           Predicted
         SQ  LU  PU  SP
Actual SQ 14   0   0   0
       LU  0  15   0   0
       PU  0   0  15   0
       SP  0   0   0  15
```

**Top Features:**
- Shoulder Y-position std
- Elbow Y-position mean
- Hip velocity mean
- Knee angle range

---

### **Notebook 3: Front View Analysis & Training** 👁️
**File:** `3_front_view_training.ipynb`

**Purpose:** Train classifier on front view videos

**Logic:**

**Same pipeline as Notebook 2, but for front view:**

**Part A: Feature Engineering**
- Load front view keypoint CSVs
- Extract same 176 features
- Focus shifts to X-coordinates and symmetry

**Part B: Visualization**
- Front-specific movement patterns
- Left-right symmetry analysis
- Hip width dynamics

**Part C: Model Training**
- Same training procedure
- Compare with side view performance

**Results:**
```
Test Accuracy: 98-100%
5-Fold CV: 99.2% ± 1.2%

Confusion Matrix:
           Predicted
         SQ  LU  PU  SP
Actual SQ 14   0   0   0
       LU  0  14   1   0
       PU  0   0  15   0
       SP  0   0   0  15
```

**Top Features:**
- Hip width mean
- Shoulder X-position std
- Wrist X-position range
- Lateral movement velocity

**Output:**
- `models/front_view_model.pkl`
- `models/front_view_scaler.pkl`
- `visualizations/front_view_*.png`

---

## 📊 **Quick Results Summary**

| Metric | Side View | Front View |
|--------|-----------|------------|
| **Test Accuracy** | 100.00% | 98-100% |
| **Best for** | Depth movements | Lateral movements |
| **Key Features** | Y-position, angles | X-position, symmetry |
| **Recommended** | Lower body | Upper body |

**Why 100% Accuracy?**
- Distinct biomechanics per exercise
- High-quality controlled recordings
- Person-based validation (prevents leakage)
- Robust statistical features

---

## 🎬 **Inference**

After training, use saved models:

```python
# Load model
import joblib
model = joblib.load('models/side_view_model.pkl')
scaler = joblib.load('models/side_view_scaler.pkl')

# Extract features from new video
features = extract_features(video_path)
features_scaled = scaler.transform(features)

# Predict
prediction = model.predict(features_scaled)
confidence = model.predict_proba(features_scaled).max()

print(f"Exercise: {prediction[0]}")
print(f"Confidence: {confidence*100:.1f}%")
```

---

## 🎯 **Exercise Signatures**

| Exercise | Key Pattern |
|----------|-------------|
| **Squats** | Large knee flexion (90°+), vertical hip movement, symmetric |
| **Lunges** | Asymmetric legs, forward/back shift, one knee deep |
| **Push-ups** | Horizontal body, elbow flexion, minimal leg movement |
| **Shoulder Press** | High wrist position, vertical arm movement, elbow extension |

---

## 🔧 **Customization**

**Add more exercises:**
1. Record videos: `{name}_{exercise}_{view}.mp4`
2. Run Notebook 1 (extraction)
3. Run Notebooks 2/3 (auto-detects new classes)

**Change landmarks:**
Edit `KEY_LANDMARKS` dictionary in training notebooks

**Combine views:**
Concatenate front + side features → 352 total features

---


**[← Back to Main README](../README.md)**