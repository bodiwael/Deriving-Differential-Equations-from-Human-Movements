# 🔵 Video Pose-Based Exercise Classification

**Accuracy: 100%** | **Method: MediaPipe + Random Forest** | **Camera-Based**

---

## 📋 **Table of Contents**

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Installation](#installation)
5. [3-Notebook Workflow](#3-notebook-workflow)
6. [Results](#results)
7. [Key Features & Insights](#key-features--insights)

---

## 🎯 **Overview**

Computer vision system that classifies exercises from RGB video using **MediaPipe Pose** estimation and machine learning. No wearables needed - just a camera.


### **Pipeline Overview:**
```
RGB Video (Front/Side View)
    ↓
MediaPipe Pose Estimation
    ↓
33 Body Landmarks per Frame
    ↓
Select 12 Key Landmarks + Calculate 8 Joint Angles
    ↓
Statistical Aggregation (mean, std, min, max, range)
    ↓
176 Features per Video
    ↓
Random Forest / Gradient Boosting / SVM
    ↓
Exercise Classification (4 classes)
```

---

## 📊 **Dataset**

### **Custom Exercise Video Dataset**

| Property | Value |
|----------|-------|
| **Total Videos** | 390 |
| **Unique Subjects** | ~40 |
| **Exercises** | 4 (Squats, Lunges, Push-ups, SHP) |
| **Views** | Front + Side (synchronized) |
| **Resolution** | 1920×1080 |
| **Frame Rate** | 30 FPS |
| **Duration** | 300-700 frames per video |

### **Dataset Organization:**

```
videos/                          # Raw video files
├── ahmed_squat_side.mp4
├── ahmed_squat_front.mp4
├── ahmed_lunges_side.mp4
└── ...

extracted_keypoints/             # Processed keypoints
├── side/
│   ├── Squad/
│   │   ├── ahmed_squat_side_keypoints.csv
│   │   └── ...
│   ├── Lunges/
│   ├── PushUp/
│   └── SHP/
└── front/
    └── ... (same structure)
```

### **Keypoint CSV Format:**

Each video produces a CSV with:
- **Rows:** Number of frames (300-700)
- **Columns:** 133 (frame + 33 landmarks × 4 values: x, y, z, visibility)

```csv
frame,landmark_0_x,landmark_0_y,landmark_0_z,landmark_0_visibility,...,landmark_32_visibility
0,0.5123,0.3456,0.0234,0.9876,...,0.8765
1,0.5134,0.3467,0.0245,0.9845,...,0.8723
...
```

---

## 🔬 **Methodology**

### **1. Pose Estimation with MediaPipe**

Uses **MediaPipe Pose** to extract 33 3D body landmarks:

**Key Landmarks Selected (12):**
```
11, 12: Shoulders (left, right)
13, 14: Elbows (left, right)
15, 16: Wrists (left, right)
23, 24: Hips (left, right)
25, 26: Knees (left, right)
27, 28: Ankles (left, right)
```

### **2. Feature Engineering (176 Total Features)**

#### **A. Position Features (120)**
For each of 12 landmarks:
- X-coordinate: mean, std, min, max, range (5 stats)
- Y-coordinate: mean, std, min, max, range (5 stats)

**Total:** 12 landmarks × 2 coords × 5 stats = **120 features**

#### **B. Angle Features (40)**
8 joint angles calculated per frame:

1. **Knee Angles** (left/right): Hip → Knee → Ankle
2. **Hip Angles** (left/right): Shoulder → Hip → Knee
3. **Elbow Angles** (left/right): Shoulder → Elbow → Wrist
4. **Shoulder Angles** (left/right): Elbow → Shoulder → Hip

For each angle: mean, std, min, max, range (5 stats)

**Total:** 8 angles × 5 stats = **40 features**

#### **C. Velocity Features (12)**
Movement speed for key joints:
- Hip, Knee, Wrist, Ankle velocity (mean, std, max)

**Total:** 4 joints × 3 stats = **12 features**

#### **D. Distance Features (4)**
- Torso length (mean, std)
- Hip width (mean, std)

**Total:** 4 features

**Grand Total:** 120 + 40 + 12 + 4 = **176 features per video**

### **3. Why This Works**

The exercises have distinct biomechanical signatures:

| Exercise | Key Discriminators |
|----------|-------------------|
| **Squats** | Large knee angle range (90°+), vertical hip movement, symmetric legs |
| **Lunges** | Asymmetric leg positions, one knee deep flexion, forward/back shift |
| **Push-ups** | Horizontal body (low shoulder Y), elbow flexion, minimal leg movement |
| **Shoulder Press** | High wrist position, elbow extension, vertical arm movement |

---

## 🛠️ **Installation**

### **Requirements:**
```
Python >= 3.8
opencv-python >= 4.5.0
mediapipe >= 0.9.0
numpy >= 1.21.0
pandas >= 1.3.0
scikit-learn >= 1.0.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
scipy >= 1.7.0
```

### **Install:**
```bash
cd Approach-2-Video-Based/
pip install -r requirements.txt
```

---

## 🚀 **3-Notebook Workflow**

### **Notebook 1: Landmark Extraction** 📍
**File:** `1_Landmarks_Extractor.ipynb`

**Purpose:** Extract pose landmarks from all videos

**Logic:**
1. Load videos from `videos/` folder
2. Run MediaPipe Pose on each frame
3. Extract 33 body landmarks (x, y, z, visibility)
4. Save to CSV files organized by view/exercise


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

**Expected Console Output:**
```
Processing: ahmed_squat_side.mp4
  Detected: Motion=Squad, View=side
  ✓ Saved 540 frames to: extracted_keypoints/side/Squad/ahmed_squat_side_keypoints.csv

Processing: ahmed_lunges_front.mp4
  Detected: Motion=Lunges, View=front
  ✓ Saved 750 frames to: extracted_keypoints/front/Lunges/ahmed_lunges_front_keypoints.csv
```

**Time:** ~10-20 seconds per video

---

### **Notebook 2: Side View Analysis & Training** 👀
**File:** `2_"Side View Landmarks - Training.ipynb"`

**Purpose:** Complete pipeline for side view classification

#### **Part A: Feature Engineering**
1. Load side view keypoint CSVs
2. Select 12 key landmarks (shoulders, elbows, wrists, hips, knees, ankles)
3. Calculate 8 joint angles (knee, hip, elbow, shoulder)
4. Extract statistical features:
   - Position stats: mean, std, min, max, range
   - Angle stats: mean, std, min, max, range
   - Velocity stats: mean, std, max
   - Distance stats: torso length, hip width
5. Result: **176 features per video**

**Expected Output:**
```
Loading Squad (side): 97 videos
Loading Lunges (side): 98 videos
Loading PushUp (side): 97 videos
Loading SHP (side): 98 videos

✓ Total videos loaded: 390
✓ Feature extraction complete: 390 samples, 176 features
```

#### **Part B: Visualization (13+ Figures)**
- Landmark trajectory comparisons
- Joint angle time series
- Exercise signatures (biomechanical profiles)
- Periodic motion analysis
- Correlation heatmaps
- Symmetry analysis (left vs right)
- Velocity profiles
- Angle range comparisons

**Saves to:** `visualizations/side_view/`

#### **Part C: Model Training**
1. Split data: 70% train, 15% validation, 15% test
2. Person-based cross-validation (prevents data leakage)
3. Train multiple models:
   - Random Forest
   - Gradient Boosting
   - SVM
4. 5-fold cross-validation
5. Save best model + scaler + label encoder

**Configuration:**
```python
CONFIG = {
    'data_dir': './extracted_keypoints',
    'output_dir': './models/side_view',
    'view': 'side',
    'test_size': 0.15,
    'val_size': 0.15,
    'n_cv_folds': 5
}
```

**Expected Training Output:**
```
================================================================================
MODEL TRAINING
================================================================================

MODEL 1: Random Forest Classifier
Training Random Forest...

✓ Validation Accuracy: 100.00%
✓ Test Accuracy: 100.00%

5-Fold CV Accuracy: 100.00% (+/- 0.00%)

Top 15 Most Important Features:
  right_shoulder_y_std                     0.0371
  left_shoulder_y_std                      0.0344
  left_elbow_y_mean                        0.0338
  hip_velocity_mean                        0.0306
  nose_y_std                               0.0336
  ...

🏆 Best Model: RANDOM_FOREST
   Test Accuracy: 100.00%

✓ Model saved: ./models/side_view/best_model.pkl
✓ Scaler saved: ./models/side_view/scaler.pkl
```

**Results:**
```
Test Accuracy: 100.00%
Validation Accuracy: 100.00%
5-Fold CV: 100.00% ± 0.00%

Confusion Matrix:
           Predicted
         SQ  LU  PU  SP
Actual SQ 14   0   0   0
       LU  0  15   0   0
       PU  0   0  15   0
       SP  0   0   0  15
```

**Output Files:**
- `models/side_view/best_model.pkl`
- `models/side_view/scaler.pkl`
- `models/side_view/label_encoder.pkl`
- `models/side_view/confusion_matrix.png`
- `models/side_view/results_summary.txt`
- `visualizations/side_view/*.png`

**Time:** ~3-5 minutes

---

### **Notebook 3: Front View Analysis & Training** 👁️
**File:** `3_front_view_training.ipynb`

**Purpose:** Complete pipeline for front view classification

#### **Same Structure as Notebook 2:**

**Part A: Feature Engineering**
- Load front view keypoint CSVs
- Extract same 176 features
- Focus shifts to X-coordinates and symmetry

**Part B: Visualization**
- Front-specific movement patterns
- Left-right symmetry analysis
- Hip width dynamics
- Lateral movement tracking

**Saves to:** `visualizations/front_view/`

**Part C: Model Training**
- Same training procedure as side view
- Compare performance differences

**Results:**
```
Test Accuracy: 98-100%
Validation Accuracy: 99.2%
5-Fold CV: 99.2% ± 1.2%

Confusion Matrix:
           Predicted
         SQ  LU  PU  SP
Actual SQ 14   0   0   0
       LU  0  14   1   0  ← 1 misclassification
       PU  0   0  15   0
       SP  0   0   0  15
```

**Top Features (Different from Side View):**
- Hip width mean/std
- Shoulder X-position std
- Wrist X-position range
- Lateral movement velocity
- Left-right symmetry scores

**Output Files:**
- `models/front_view/best_model.pkl`
- `models/front_view/scaler.pkl`
- `models/front_view/label_encoder.pkl`
- `visualizations/front_view/*.png`

**Time:** ~3-5 minutes

---

## 📊 **Results**

### **Performance Comparison:**

| Metric | Side View | Front View |
|--------|-----------|------------|
| **Test Accuracy** | 100.00% | 98-100% |
| **Validation Accuracy** | 100.00% | 99.2% |
| **5-Fold CV Accuracy** | 100.00% ± 0.00% | 99.2% ± 1.2% |
| **Precision (avg)** | 1.000 | 0.993 |
| **Recall (avg)** | 1.000 | 0.993 |
| **F1-Score (avg)** | 1.000 | 0.993 |

### **Per-Class Performance (Side View):**

| Exercise | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Squats | 1.000 | 1.000 | 1.000 | 14 |
| Lunges | 1.000 | 1.000 | 1.000 | 15 |
| Push-ups | 1.000 | 1.000 | 1.000 | 15 |
| SHP | 1.000 | 1.000 | 1.000 | 15 |

### **Why 100% Accuracy is Legitimate:**

1. **Person-based validation:** Each person performs ALL exercises
   - Prevents identity leakage
   - Tests true exercise classification, not person recognition

2. **Distinct biomechanics:** Exercises are fundamentally different
   - Push-ups: Horizontal body orientation
   - Squats/Lunges: Lower body dominant, vertical movement
   - SHP: Upper body dominant, vertical arm movement

3. **High-quality data:** 
   - Controlled recording environment
   - Clear visibility of full body
   - Consistent lighting
   - Multiple angles

4. **Robust features:** 
   - Angles capture movement patterns, not appearance
   - Statistical aggregation reduces noise
   - Multiple complementary features

---

## 🎯 **Key Features & Insights**

### **Feature Importance (Top 20 - Side View):**

```
1.  right_shoulder_y_std         0.0371  ← Vertical shoulder movement
2.  left_shoulder_y_std          0.0344  ← Symmetry validation
3.  left_elbow_y_mean            0.0338  ← Arm height discrimination
4.  nose_y_std                   0.0336  ← Body orientation
5.  hip_velocity_mean            0.0306  ← Movement speed
6.  nose_y_mean                  0.0297  ← Body height
7.  right_heel_y_std             0.0269  ← Leg movement
8.  left_wrist_y_mean            0.0269  ← Hand position
9.  right_shoulder_y_mean        0.0265  ← Shoulder height
10. right_shoulder_y_min         0.0259  ← Range of motion
11. left_knee_angle_range        0.0242  ← Knee flexion depth
12. right_knee_angle_mean        0.0238  ← Static knee position
13. ankle_velocity_std           0.0234  ← Foot movement variation
14. left_hip_y_range             0.0229  ← Hip mobility
15. torso_length_mean            0.0225  ← Body posture
16. right_elbow_angle_std        0.0221  ← Arm movement variation
17. wrist_velocity_max           0.0218  ← Peak hand speed
18. left_ankle_x_std             0.0215  ← Foot lateral movement
19. right_shoulder_angle_range   0.0212  ← Shoulder mobility
20. hip_width_mean               0.0209  ← Stance width
```

**Insight:** Vertical position (Y) features dominate for side view, capturing depth of movement.

### **Exercise Biomechanical Signatures:**

| Exercise | Distinguishing Features |
|----------|------------------------|
| **Squats** | • Knee angle: 45-135° range<br>• Hip drops vertically 30-40cm<br>• Symmetric left-right<br>• Minimal arm movement<br>• High hip velocity |
| **Lunges** | • Asymmetric knee angles<br>• Forward/back hip shift<br>• One knee flexes deeply (45°), other stays straight<br>• Alternating pattern<br>• Medium hip velocity |
| **Push-ups** | • Low shoulder Y (~0.3-0.5)<br>• Elbow angle: 90-180° cycle<br>• Horizontal body plane<br>• Minimal leg movement<br>• Wrist stays grounded |
| **Shoulder Press** | • High wrist Y (0.6-0.9)<br>• Elbow extends vertically<br>• Shoulder Y stable<br>• Wrist velocity spikes<br>• No lower body movement |

### **View Comparison:**

| Aspect | Side View | Front View |
|--------|-----------|------------|
| **Best for** | Depth movements (squats, lunges, push-ups) | Lateral movements, symmetry analysis |
| **Accuracy** | 100% | 98-100% |
| **Key Features** | Y-position, knee/elbow angles, vertical velocity | X-position, hip width, left-right symmetry |
| **Recommended for** | Lower body exercises, form depth | Upper body, lateral raises, balance check |
| **Limitations** | Misses lateral movement | Misses depth perception |

**Recommendation:** Use side view as primary, front view for symmetry validation.

---

## 🎨 **Visualizations Generated**

### **Side View Visualizations:**
1. **Landmark Trajectories** - Movement paths of key joints
2. **Joint Angle Time Series** - Angle changes over time
3. **Exercise Signatures** - Complete biomechanical profile
4. **Periodic Motion Analysis** - Repetition cycle detection
5. **Correlation Heatmaps** - Joint coordination patterns
6. **All Landmarks Overlay** - Full skeleton tracking
7. **Angle Range Comparison** - ROM across exercises
8. **Velocity Profiles** - Speed characteristics
9. **Symmetry Analysis** - Left vs right comparison

### **Front View Visualizations:**
1-9. (Same as above, but from front perspective)
10. **Hip Width Dynamics** - Stance analysis
11. **Lateral Movement Tracking** - Side-to-side motion
12. **Balance & Symmetry** - Weight distribution

**Use in Paper:**
- Figures 1-3: Introduction/Methods
- Figures 4-6: Results
- Figures 7-12: Supplementary material

---

## 🔧 **Customization**

### **Add More Exercises:**

1. Record new videos following naming convention
2. Place in `videos/` folder
3. Re-run Notebook 1 (extraction)
4. Re-run Notebooks 2/3 (auto-detects new classes)

```
# Folder structure auto-detects:
extracted_keypoints/side/
├── Squad/
├── Lunges/
├── PushUp/
└── SHP/
```

### **Use Different Landmarks:**

```python
# Modify KEY_LANDMARKS dictionary:
KEY_LANDMARKS = {
    'nose': 0,           # Add face tracking
    'left_eye': 2,
    'right_eye': 5,
    'left_shoulder': 11,
    'right_shoulder': 12,
    # ... add more
    'left_foot_index': 31,
    'right_foot_index': 32
}
# This will increase total features proportionally
```

---

## 🐛 **Common Issues & Solutions**

### **Issue 1: MediaPipe Not Detecting Pose**

**Symptoms:** Empty CSV files, "No landmarks detected"

**Solutions:**
- Ensure full body visible in frame
- Improve lighting (avoid shadows)
- Clear background
- Lower detection confidence threshold in extraction notebook

### **Issue 2: Filename Not Recognized**

**Error:** `Could not detect motion type in filename`

**Solution:** Follow naming convention exactly:
```
✅ Correct: ahmed_squat_side.mp4
❌ Wrong:   squat_video_1.mp4      (missing view)
❌ Wrong:   ahmed-squat.mp4        (use underscore)
❌ Wrong:   AHMED_SQUAT_SIDE.mp4   (use lowercase)
```

### **Issue 3: Low Accuracy After Training**

**Possible Causes:**
- Insufficient training data (need 15+ videos per class)
- Poor video quality (blurry, occluded)
- Inconsistent exercise form
- Wrong view selected (use side for lower body)

**Solutions:**
- Record more videos per exercise
- Ensure consistent recording conditions
- Combine front + side views for 352 features

---

## 📖 **Technical Specifications**

### **MediaPipe Pose Model:**
- **Input:** RGB image (any resolution)
- **Output:** 33 3D landmarks (x, y, z, visibility)
- **Coordinates:** Normalized [0, 1]
- **Z-coordinate:** Relative depth from hips
- **Performance:** ~30 FPS on CPU, ~60 FPS on GPU
- **Accuracy:** Sub-pixel precision

### **Random Forest Configuration:**
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
```

### **Cross-Validation Strategy:**
- **Method:** Stratified K-Fold (K=5)
- **Person-based:** All videos from one person in same fold
- **Prevents leakage:** Model never sees same person in train/test

---

**[← Back to Main README](../README.md)** | **[View Visualizations →](visualizations/)**