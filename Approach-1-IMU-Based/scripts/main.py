import os
import pandas as pd
import re

# Define your data directory
data_dir = "synced_imus"  # Change this to the actual path

# Output directory for processed data
output_dir = "processed_data"
os.makedirs(output_dir, exist_ok=True)

# List of valid sensor points
sensor_points = ["LC", "LThigh", "RUA", "RWrist", "LUA", "back", "RThigh", "RC", "LWrist"]

# Regular expression to extract movement name from folder
movement_regex = re.compile(r"([a-zA-Z]+)-([a-zA-Z]+)-([a-zA-Z]+)-imu")

# Regular expression to extract timestamp from filename
timestamp_regex = re.compile(r"_C\d+-(\d{4}-\d{2}-\d{2}T\d{2}\.\d{2}\.\d{2}\.\d+)_")

# Function to safely read CSV files
def safe_read_csv(filepath):
    try:
        df = pd.read_csv(filepath, usecols=["timestamp (+0200)", "x-axis (deg/s)", "y-axis (deg/s)", "z-axis (deg/s)"], encoding="utf-8", engine="python")
        if df.empty:
            print(f"⚠️ Warning: Empty CSV file - {filepath}")
        return df
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return None

# Loop through each movement folder
for folder in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder)

    if os.path.isdir(folder_path):
        match = movement_regex.search(folder)
        if match:
            movement = match.group(3)  # Extract movement name (e.g., 'shp', 'sq', etc.)
            print(f"📌 Processing Movement: {movement} in {folder}")

            if os.path.isdir(folder_path):  # Ensure it's a folder
                print(f"🔄 Processing Trial: {folder_path} for {movement}")

                trial_data = None  # Placeholder for merged data

                # Print detected files
                print(f"📂 Checking folder: {folder_path}")
                print(f"🔍 Files in trial folder: {os.listdir(folder_path)}")

                # Loop through sensor files inside the trial folder
                for file in os.listdir(folder_path):
                    if file.endswith(".csv") and "Gyroscope" in file:
                        print(f"📄 Processing File: {file}")

                        # Extract sensor point name from filename
                        point_name = next((p for p in sensor_points if p in file), None)

                        # Extract timestamp
                        trial_match = timestamp_regex.search(file)
                        timestamp = trial_match.group(1) if trial_match else "UnknownTimestamp"

                        if point_name:
                            file_path = os.path.join(folder_path, file)

                            # Load sensor data
                            df = safe_read_csv(file_path)
                            if df is None or df.empty:
                                continue  # Skip if reading failed

                            print(f"📊 Data preview for {file}:\n{df.head()}")

                            # Rename columns to include sensor point
                            df = df.rename(columns={"x-axis (deg/s)": f"X_{point_name}", "y-axis (deg/s)": f"Y_{point_name}", "z-axis (deg/s)": f"Z_{point_name}"})

                            # Merge based on timestamp
                            if trial_data is None:
                                trial_data = df
                            else:
                                trial_data = pd.merge(trial_data, df, on="timestamp (+0200)", how="outer")

                # Sort by timestamp and interpolate missing values
                if trial_data is not None and not trial_data.empty:
                    trial_data = trial_data.sort_values("timestamp (+0200)").interpolate()
                    print(f"🔗 Merged Data Sample ({movement} - {folder_path}):\n{trial_data.head()}")

                    # Save processed trial
                    output_file = os.path.join(output_dir, f"{movement}_{folder_path}.csv")
                    print(f"✅ Saving processed data: {output_file}")
                    trial_data.to_csv(output_file, index=False)
                else:
                    print(f"⚠️ No data to save for {folder_path}. Check input files.")

print("🚀 All movements and trials processed successfully!")
