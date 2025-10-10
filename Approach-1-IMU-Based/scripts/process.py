import os
import pandas as pd
import numpy as np

# Input and output directories
input_dir = "processed_data"  # Folder containing movement folders
output_dir = "final_processed_data"  # Folder to save processed data
os.makedirs(output_dir, exist_ok=True)

# Function to process gyroscope data
def process_gyroscope_data(filepath):
    print(f"📂 Processing file: {filepath}")

    # Load CSV
    df = pd.read_csv(filepath)

    # Convert timestamp to datetime and sort
    df["timestamp (+0200)"] = pd.to_datetime(df["timestamp (+0200)"], format="%Y-%m-%dT%H.%M.%S.%f")
    df = df.sort_values("timestamp (+0200)")

    # Compute time delta in seconds
    df["dt"] = df["timestamp (+0200)"].diff().dt.total_seconds().fillna(0)

    # Get all gyroscope columns (deg/s)
    gyro_cols = [col for col in df.columns if col.startswith(("X_", "Y_", "Z_"))]

    for col in gyro_cols:
        # Angular acceleration (deg/s²) = diff(angular velocity) / dt
        a_col = col.replace("_", "_A")  # Example: X_LThigh -> X_A_LThigh
        df[a_col] = df[col].diff() / df["dt"]
        df[a_col] = df[a_col].fillna(0)

        # Angular displacement (deg) = ∫ angular velocity * dt
        d_col = col.replace("_", "_D")  # Example: X_LThigh -> X_D_LThigh
        df[d_col] = np.cumsum(df[col] * df["dt"])

    # Select useful columns
    keep_cols = ["timestamp (+0200)", "dt"] + gyro_cols + \
                [col.replace("_", "_A") for col in gyro_cols] + \
                [col.replace("_", "_D") for col in gyro_cols]
    df = df[keep_cols]

    return df

# Loop through all movement folders
for movement_folder in os.listdir(input_dir):
    movement_path = os.path.join(input_dir, movement_folder)

    if os.path.isdir(movement_path):  # Check if it's a folder
        print(f"📌 Processing Movement: {movement_folder}")

        # Create corresponding folder in output directory
        movement_output_dir = os.path.join(output_dir, movement_folder)
        os.makedirs(movement_output_dir, exist_ok=True)

        # Process each CSV file inside this movement folder
        for file in os.listdir(movement_path):
            if file.endswith(".csv"):
                file_path = os.path.join(movement_path, file)
                processed_df = process_gyroscope_data(file_path)

                # Save processed file in the corresponding movement folder
                output_path = os.path.join(movement_output_dir, file)
                processed_df.to_csv(output_path, index=False)
                print(f"✅ Saved processed data: {output_path}")

print("🚀 All movements and trials processed successfully!")
