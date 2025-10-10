import pandas as pd
import matplotlib.pyplot as plt
import os

# === SETTINGS ===
CLASS_FOLDER = "sq_synced_imus"
BASE_DIR = "final_processed_data"
SENSOR = "LThigh"
AXIS_COL = f"X_D{SENSOR}"
WINDOW_SIZE = 130
REPETITION_DURATION_SEC = 2.5

# === PROCESS FUNCTION ===
def process_file_manual_reps(file_path, class_folder):
    filename = os.path.basename(file_path).replace(".csv", "")
    df = pd.read_csv(file_path)
    df["timestamp (+0200)"] = pd.to_datetime(df["timestamp (+0200)"])
    df.sort_values("timestamp (+0200)", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Smooth displacement
    df["displacement_smoothed"] = df[AXIS_COL].rolling(window=WINDOW_SIZE, center=True).mean()

    # Determine sampling rate
    time_diffs = df["timestamp (+0200)"].diff().dt.total_seconds().dropna()
    avg_sampling_rate = 1 / time_diffs.mean()
    samples_per_rep = int(REPETITION_DURATION_SEC * avg_sampling_rate)

    # Manual segmentation
    segments = []
    num_rows = len(df)
    for i in range(0, num_rows, samples_per_rep):
        segment = df.iloc[i:i + samples_per_rep]
        if len(segment) > 1:
            segments.append(segment)

    # === OUTPUT FOLDERS ===
    class_path = os.path.join(BASE_DIR, class_folder)
    output_csv_dir = os.path.join(class_path, "repetition_segments_csvs")
    output_plot_dir = os.path.join(class_path, "plots")
    os.makedirs(output_csv_dir, exist_ok=True)
    os.makedirs(output_plot_dir, exist_ok=True)

    # === SAVE SEGMENTS ===
    for i, segment in enumerate(segments):
        rep_filename = f"{filename}_rep_{i+1}.csv"
        segment.to_csv(os.path.join(output_csv_dir, rep_filename), index=False)

    # === PLOTTING ===
    # Smoothed plot
    plt.figure(figsize=(12, 5))
    plt.plot(df["timestamp (+0200)"], df[AXIS_COL], alpha=0.5, label="Original")
    plt.plot(df["timestamp (+0200)"], df["displacement_smoothed"], color='red', label="Smoothed")
    plt.title(f"Smoothed Angular Displacement - {SENSOR} ({filename})")
    plt.xlabel("Time")
    plt.ylabel("Angular Displacement (deg)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_plot_dir, f"{filename}_smoothed.png"), dpi=300)
    plt.close()

    # Plot first 5 repetitions
    plt.figure(figsize=(12, 5))
    for i, segment in enumerate(segments[:5]):
        plt.plot(segment["timestamp (+0200)"], segment["displacement_smoothed"], label=f"Repetition {i+1}")
    plt.title(f"First 5 Repetitions - {filename} (Manual 2.5s)")
    plt.xlabel("Time")
    plt.ylabel("Angular Displacement (deg)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_plot_dir, f"{filename}_sample_reps.png"), dpi=300)
    plt.close()

    print(f"✅ Processed {filename}: {len(segments)} manual reps (every 2.5s)")

# === MAIN LOOP ===
folder_path = os.path.join(BASE_DIR, CLASS_FOLDER)
csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

print(f"\n📁 Processing 'walk' class: Found {len(csv_files)} CSV files.")
for file in csv_files:
    try:
        process_file_manual_reps(os.path.join(folder_path, file), CLASS_FOLDER)
    except Exception as e:
        print(f"❌ Error processing {file}: {e}")
