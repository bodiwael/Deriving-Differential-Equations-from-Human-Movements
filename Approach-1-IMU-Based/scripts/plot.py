import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Constants ---
RADIUS_M = 0.4  # radius from joint to sensor in meters

# --- Function to plot angular data ---
def plot_angular_sensor_data(df, sensor, output_folder):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(df["timestamp (+0200)"], df[f"X_{sensor}"], label="X Angular Velocity (deg/s)", color="blue")
    axes[0].set_ylabel("Angular Velocity (deg/s)")
    axes[0].legend()

    axes[1].plot(df["timestamp (+0200)"], df[f"X_A{sensor}"], label="X Angular Acceleration (deg/s²)", color="red")
    axes[1].set_ylabel("Angular Acceleration (deg/s²)")
    axes[1].legend()

    axes[2].plot(df["timestamp (+0200)"], df[f"X_D{sensor}"], label="X Angular Displacement (deg)", color="green")
    axes[2].set_ylabel("Angular Displacement (deg)")
    axes[2].set_xlabel("Time")
    axes[2].legend()

    os.makedirs(output_folder, exist_ok=True)
    plot_path = os.path.join(output_folder, f"{sensor}_angular_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Angular plot saved: {plot_path}")

# --- Function to convert angular to linear ---
def add_linear_columns(df, sensor):
    for prefix in ["", "A", "D"]:
        deg_col = f"X_{prefix}{sensor}" if prefix else f"X_{sensor}"
        linear_col = f"X_{prefix}Linear_{sensor}" if prefix else f"X_Linear_{sensor}"
        df[linear_col] = df[deg_col] * (np.pi / 180) * RADIUS_M
    return df

# --- Function to plot linear data ---
def plot_linear_sensor_data(df, sensor, output_folder):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(df["timestamp (+0200)"], df[f"X_Linear_{sensor}"], label="X Linear Velocity (m/s)", color="blue")
    axes[0].set_ylabel("Linear Velocity (m/s)")
    axes[0].legend()

    axes[1].plot(df["timestamp (+0200)"], df[f"X_ALinear_{sensor}"], label="X Linear Acceleration (m/s²)", color="red")
    axes[1].set_ylabel("Linear Acceleration (m/s²)")
    axes[1].legend()

    axes[2].plot(df["timestamp (+0200)"], df[f"X_DLinear_{sensor}"], label="X Linear Displacement (m)", color="green")
    axes[2].set_ylabel("Linear Displacement (m)")
    axes[2].set_xlabel("Time")
    axes[2].legend()

    os.makedirs(output_folder, exist_ok=True)
    plot_path = os.path.join(output_folder, f"{sensor}_linear_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📈 Linear plot saved: {plot_path}")

# --- Load and preprocess data ---
file_path = "final_processed_data/lunges_synced_imus/ahmed-tarek-lunges-imu.csv"
df = pd.read_csv(file_path)
df["timestamp (+0200)"] = pd.to_datetime(df["timestamp (+0200)"])

# --- Choose sensor and run ---
sensor = "LThigh"
df = add_linear_columns(df, sensor)
plot_angular_sensor_data(df, sensor, "plots")
plot_linear_sensor_data(df, sensor, "plots")
