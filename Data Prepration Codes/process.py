import os
import pandas as pd
import numpy as np

# Input and output directories
input_dir = "processed_data"  # Folder containing movement folders
output_dir = "final_processed_data"  # Folder to save processed data
os.makedirs(output_dir, exist_ok=True)

# Function to process accelerometer data for all sensors
def process_accelerometer_data(filepath):
    print(f"📂 Processing file: {filepath}")

    # Load CSV
    df = pd.read_csv(filepath)

    # Convert timestamp to datetime and sort data
    df["timestamp (+0200)"] = pd.to_datetime(df["timestamp (+0200)"], format="%Y-%m-%dT%H.%M.%S.%f")
    df = df.sort_values("timestamp (+0200)")

    # Compute time differences (dt) in seconds
    df["dt"] = df["timestamp (+0200)"].diff().dt.total_seconds().fillna(0)

    # Convert all acceleration columns from g to m/s²
    acceleration_cols = [col for col in df.columns if col.startswith(("X_", "Y_", "Z_"))]
    for col in acceleration_cols:
        df[col] = df[col] * 9.81  # Convert g to m/s²

    # Compute velocity using trapezoidal integration
    for col in acceleration_cols:
        v_col = col.replace("_", "_V")  # Example: X_LC -> X_V_LC
        df[v_col] = np.cumsum(df[col] * df["dt"])  # Integrate acceleration to get velocity

    # Compute displacement using trapezoidal integration
    for col in acceleration_cols:
        d_col = col.replace("_", "_D")  # Example: X_LC -> X_D_LC
        v_col = col.replace("_", "_V")  # Corresponding velocity column
        df[d_col] = np.cumsum(df[v_col] * df["dt"])  # Integrate velocity to get displacement

    # Drop unnecessary columns
    keep_cols = ["timestamp (+0200)", "dt"] + acceleration_cols + [col.replace("_", "_V") for col in acceleration_cols] + [col.replace("_", "_D") for col in acceleration_cols]
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
                processed_df = process_accelerometer_data(file_path)

                # Save processed file in the corresponding movement folder
                output_path = os.path.join(movement_output_dir, file)
                processed_df.to_csv(output_path, index=False)
                print(f"✅ Saved processed data: {output_path}")

print("🚀 All movements and trials processed successfully!")
