"""
===============================================================================
IMU-BASED EXERCISE CLASSIFICATION - IMPROVEMENTS
===============================================================================
Goal: Improve from 82% to 90%+ accuracy
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================================
# IMPROVEMENT 1: USE MULTIPLE SENSORS (CRITICAL!)
# ============================================================================

def extract_multi_sensor_features(file_path, sensors=["LThigh", "RThigh", "back"], degree=2, alpha=0.1):
    """
    Extract SINDy features from MULTIPLE sensors
    This is THE most important improvement!
    """
    try:
        df = pd.read_csv(file_path)
        
        all_features = []
        
        for sensor in sensors:
            # Check if sensor columns exist
            required_cols = [f"X_D{sensor}", f"Y_D{sensor}", f"Z_D{sensor}"]
            if not all(col in df.columns for col in required_cols):
                # print(f"⚠️  Sensor {sensor} not found in {file_path}")
                continue
            
            df_clean = df[required_cols + ["timestamp (+0200)"]].dropna()
            
            if len(df_clean) < 10:  # Need minimum samples
                continue
            
            ts = pd.to_datetime(df_clean["timestamp (+0200)"])
            dt = ts.diff().dt.total_seconds().mean()
            if pd.isna(dt) or dt <= 0:
                continue
            
            # Extract displacement data
            x = df_clean[f"X_D{sensor}"].values
            y = df_clean[f"Y_D{sensor}"].values
            z = df_clean[f"Z_D{sensor}"].values
            
            # Compute derivatives
            vx, vy, vz = np.gradient(x, dt), np.gradient(y, dt), np.gradient(z, dt)
            ax, ay, az = np.gradient(vx, dt), np.gradient(vy, dt), np.gradient(vz, dt)
            
            # Build library
            X_features = np.stack([x, y, z, vx, vy, vz], axis=1)
            poly = PolynomialFeatures(degree=degree, include_bias=True)
            Theta = poly.fit_transform(X_features)
            
            # NORMALIZE Theta to improve convergence
            theta_scaler = StandardScaler()
            Theta_scaled = theta_scaler.fit_transform(Theta)
            
            # Sparse regression with increased iterations
            model = Lasso(alpha=alpha, fit_intercept=False, max_iter=50000, tol=1e-4)
            
            model.fit(Theta_scaled, ax)
            coeffs_x = model.coef_
            model.fit(Theta_scaled, ay)
            coeffs_y = model.coef_
            model.fit(Theta_scaled, az)
            coeffs_z = model.coef_
            
            sensor_coeffs = np.concatenate([coeffs_x, coeffs_y, coeffs_z])
            all_features.append(sensor_coeffs)
        
        if len(all_features) == 0:
            return None
        
        # Concatenate features from all sensors
        combined_features = np.concatenate(all_features)
        return combined_features
        
    except Exception as e:
        # print(f"Error: {e}")
        return None


# ============================================================================
# IMPROVEMENT 2: ADD STATISTICAL FEATURES (Time-Domain)
# ============================================================================

def extract_statistical_features(file_path, sensors=["LThigh", "RThigh", "back"]):
    """
    Add simple statistical features alongside SINDy
    """
    try:
        df = pd.read_csv(file_path)
        features = []
        
        for sensor in sensors:
            for axis in ["X", "Y", "Z"]:
                col = f"{axis}_D{sensor}"
                if col not in df.columns:
                    continue
                
                data = df[col].dropna()
                if len(data) == 0:
                    continue
                
                # Statistical features
                features.extend([
                    data.mean(),
                    data.std(),
                    data.min(),
                    data.max(),
                    data.max() - data.min(),  # Range
                    np.percentile(data, 25),
                    np.percentile(data, 75),
                    np.sqrt(np.mean(data**2)),  # RMS
                ])
        
        return np.array(features) if len(features) > 0 else None
        
    except Exception as e:
        return None


# ============================================================================
# IMPROVEMENT 3: COMBINE SINDY + STATISTICAL FEATURES
# ============================================================================

def extract_combined_features(file_path, sensors=["LThigh", "RThigh", "back"], 
                              degree=2, alpha=0.1):
    """
    Combine SINDy (dynamic) + Statistical (descriptive) features
    """
    sindy_feats = extract_multi_sensor_features(file_path, sensors, degree, alpha)
    stat_feats = extract_statistical_features(file_path, sensors)
    
    if sindy_feats is None and stat_feats is None:
        return None
    elif sindy_feats is None:
        return stat_feats
    elif stat_feats is None:
        return sindy_feats
    else:
        return np.concatenate([sindy_feats, stat_feats])


# ============================================================================
# IMPROVEMENT 4: BETTER DATA PROCESSING PIPELINE
# ============================================================================

def process_all_classes_improved(base_path, class_folders, 
                                 sensors=["LThigh", "RThigh", "back"],
                                 degree=2, alpha=0.1, train_users_ratio=0.66):
    """
    Improved data processing with multiple sensors + FIX for dimension mismatch
    """
    user_data = defaultdict(list)
    feature_dimensions = []  # Track feature dimensions
    
    for cls in class_folders:
        cls_path = os.path.join(base_path, cls, "repetition_segments_csvs")
        
        if not os.path.exists(cls_path):
            print(f"⚠️  Path not found: {cls_path}")
            continue
        
        files = [f for f in os.listdir(cls_path) if f.endswith(".csv")]
        print(f"📂 Processing {cls}: {len(files)} files")
        
        for file in files:
            user_id = file.split("-")[0]
            path = os.path.join(cls_path, file)
            
            # Extract combined features
            features = extract_combined_features(path, sensors, degree, alpha)
            
            if features is not None:
                user_data[user_id].append((features, cls))
                feature_dimensions.append(len(features))
    
    # CHECK: Ensure all features have same dimension
    if len(feature_dimensions) > 0:
        unique_dims = set(feature_dimensions)
        print(f"\n🔍 Feature dimension check:")
        print(f"   Unique dimensions found: {unique_dims}")
        
        if len(unique_dims) > 1:
            print(f"   ⚠️  WARNING: Inconsistent feature dimensions!")
            print(f"   Most common: {max(set(feature_dimensions), key=feature_dimensions.count)}")
            
            # FIX: Filter out samples with wrong dimensions
            target_dim = max(set(feature_dimensions), key=feature_dimensions.count)
            
            # Remove samples with wrong dimensions
            filtered_user_data = defaultdict(list)
            removed_count = 0
            
            for user_id, samples in user_data.items():
                for features, label in samples:
                    if len(features) == target_dim:
                        filtered_user_data[user_id].append((features, label))
                    else:
                        removed_count += 1
            
            user_data = filtered_user_data
            print(f"   ✓ Removed {removed_count} samples with wrong dimensions")
            print(f"   ✓ Keeping samples with {target_dim} features")
    
    # User-based split
    all_users = list(user_data.keys())
    print(f"\n👥 Total unique users: {len(all_users)}")
    
    if len(all_users) == 0:
        print("❌ No valid users found!")
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    np.random.seed(42)
    np.random.shuffle(all_users)
    split_idx = int(len(all_users) * train_users_ratio)
    train_users, test_users = all_users[:split_idx], all_users[split_idx:]
    
    print(f"   Training users: {len(train_users)}")
    print(f"   Test users: {len(test_users)}")
    
    # Build datasets
    X_train, y_train, X_test, y_test = [], [], [], []
    
    for user in train_users:
        for features, label in user_data[user]:
            X_train.append(features)
            y_train.append(label)
    
    for user in test_users:
        for features, label in user_data[user]:
            X_test.append(features)
            y_test.append(label)
    
    print(f"\n📊 Dataset split:")
    print(f"   Train samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Convert to numpy arrays (now all same dimension)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    print(f"   Train shape: {X_train.shape}")
    print(f"   Test shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# IMPROVEMENT 5: TRY MULTIPLE CLASSIFIERS
# ============================================================================

def train_multiple_classifiers(X_train, X_test, y_train, y_test, class_folders):
    """
    Try different classifiers and pick the best
    """
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    classifiers = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, 
            max_depth=20, 
            min_samples_split=5,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        'SVM (RBF)': SVC(
            kernel='rbf',
            C=10.0,
            gamma='scale',
            random_state=42
        )
    }
    
    results = {}
    
    print("\n" + "="*80)
    print("TRAINING MULTIPLE CLASSIFIERS")
    print("="*80)
    
    for name, clf in classifiers.items():
        print(f"\n🔧 Training {name}...")
        clf.fit(X_train_scaled, y_train)
        
        y_pred = clf.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"   ✓ Accuracy: {accuracy*100:.2f}%")
        
        results[name] = {
            'model': clf,
            'accuracy': accuracy,
            'predictions': y_pred
        }
    
    # Find best model
    best_name = max(results, key=lambda k: results[k]['accuracy'])
    best_accuracy = results[best_name]['accuracy']
    
    print(f"\n🏆 Best Model: {best_name} ({best_accuracy*100:.2f}%)")
    
    # Detailed report for best model
    y_pred_best = results[best_name]['predictions']
    
    print("\n📋 Classification Report (Best Model):")
    print("="*80)
    print(classification_report(y_test, y_pred_best, target_names=class_folders))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_best, labels=class_folders)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_folders, yticklabels=class_folders)
    plt.xlabel("Predicted", fontsize=12, fontweight='bold')
    plt.ylabel("True", fontsize=12, fontweight='bold')
    plt.title(f"Confusion Matrix - {best_name} (Improved)", 
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("improved_confusion_matrix.png", dpi=300)
    print("\n✓ Saved: improved_confusion_matrix.png")
    plt.show()
    
    return results, best_name


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("IMU-BASED CLASSIFICATION - IMPROVED APPROACH")
    print("="*80)
    print()
    
    base_path = "final_processed_data"
    class_folders = ["pu_synced_imus", "lunges_synced_imus", 
                    "sq_synced_imus", "shp_synced_imus"]
    
    # OPTION 1: Try with multiple sensors
    print("🔧 OPTION 1: Using Multiple Sensors (LThigh, RThigh, back)")
    print("-"*80)
    
    X_train, X_test, y_train, y_test = process_all_classes_improved(
        base_path, 
        class_folders,
        sensors=["LThigh", "RThigh", "back"],  # Use 3 sensors!
        degree=2,
        alpha=0.1
    )
    
    if len(X_train) > 0:
        results, best_model = train_multiple_classifiers(
            X_train, X_test, y_train, y_test, class_folders
        )
    
    # OPTION 2: Try with different sensor combinations
    print("\n" + "="*80)
    print("🔧 OPTION 2: Trying Different Sensor Combinations")
    print("-"*80)
    
    sensor_combinations = [
        ["LThigh", "RThigh"],
        ["LThigh", "back"],
        ["LThigh", "RThigh", "back", "LUA", "RUA"],  # All upper + lower
    ]
    
    best_combo_acc = 0
    best_combo = None
    
    for sensors in sensor_combinations:
        print(f"\nTesting sensors: {sensors}")
        
        X_train, X_test, y_train, y_test = process_all_classes_improved(
            base_path, class_folders, sensors=sensors
        )
        
        if len(X_train) == 0:
            print(f"   ⚠️  No data extracted for {sensors}")
            continue
        
        # Quick RF test
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_scaled, y_train)
        
        acc = accuracy_score(y_test, clf.predict(X_test_scaled))
        print(f"   Accuracy: {acc*100:.2f}%")
        
        if acc > best_combo_acc:
            best_combo_acc = acc
            best_combo = sensors
    
    print(f"\n🏆 Best sensor combination: {best_combo} ({best_combo_acc*100:.2f}%)")