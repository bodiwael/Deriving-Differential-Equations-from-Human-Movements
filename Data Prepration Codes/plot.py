import os
import pandas as pd
import matplotlib.pyplot as plt

# Function to plot and save the data
def plot_sensor_data(df, sensor, output_folder):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Acceleration Plot
    axes[0].plot(df["timestamp (+0200)"], df[f"X_{sensor}"], label="X Acceleration", color="blue")
    axes[0].set_ylabel("Acceleration (m/s²)")
    axes[0].legend()

    # Velocity Plot
    axes[1].plot(df["timestamp (+0200)"], df[f"X_V{sensor}"], label="X Velocity", color="red")
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].legend()

    # Displacement Plot
    axes[2].plot(df["timestamp (+0200)"], df[f"X_D{sensor}"], label="X Displacement", color="green")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Displacement (m)")
    axes[2].legend()

    # Save plot as PNG
    os.makedirs(output_folder, exist_ok=True)  # Ensure folder exists
    plot_path = os.path.join(output_folder, f"{sensor}_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")  # Save with high resolution
    plt.close()  # Close the figure to prevent it from displaying

    print(f"📊 Plot saved: {plot_path}")

# Load processed data
file_path = "final_processed_data/lunges_synced_imus/ahmed-tarek-lunges-imu.csv"  # Change to actual path
df = pd.read_csv(file_path)

# Convert timestamp to datetime for proper plotting
df["timestamp (+0200)"] = pd.to_datetime(df["timestamp (+0200)"])

# Call the function for a specific sensor
plot_sensor_data(df, "LThigh", "plots")  # Change "LThigh" to other sensors if needed
