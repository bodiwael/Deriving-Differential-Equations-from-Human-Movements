# **Deriving Differential Equations from Human Movements using Computer Vision & PINNs**  

🚀 **This project aims to derive biomechanical differential equations from human movement data using Computer Vision techniques and Physics-Informed Neural Networks (PINNs).**  

## 🔍 **Overview**  
Understanding human motion mathematically is a key challenge in biomechanics, robotics, and AI-driven sports analytics. This project explores:  
✅ **Extracting human movement features** efficiently using **MediaPipe** and **motion magnification** techniques.  
✅ **Reducing dimensionality** of keypoints using **Autoencoders** to retain the most meaningful movement descriptors.  
✅ **Using PINNs to infer differential equations** governing movement dynamics.  

## 📌 **Key Components**  
- **Feature Extraction**: Using **3D keypoints** from **MediaPipe** and alternative representations.  
- **Dimensionality Reduction**: Comparing **Autoencoders, PCA, and other methods**.  
- **Motion Magnification**: Enhancing imperceptible movements using **Eulerian Video Magnification (EVM) and Phase-Based Magnification (PBMM)**.  
- **PINNs Implementation**: Training models to learn the governing equations of human motion.  

## 📂 **Project Structure**  
```plaintext
📦 Deriving-Differential-Equations-from-Human-Movements
├── 📁 Notebooks/           # Jupyter notebooks for visualization
├── 📁 Meeting Notes/       # Abstracting the notes for each meeiting + Action Plans
├── 📁 Data/                # Model Dataset
├── 📁 Papers/              # Testing different methods
├── 📁 notebooks/           # Collected Useful Papers
└── README.md               # You are here 🚀
```

## 🛠️ **Setup & Installation**  
1️⃣ Clone the repository:  
```bash
git clone https://github.com/bodiwael/Deriving-Differential-Equations-from-Human-Movements.git
cd Deriving-Differential-Equations-from-Human-Movements
```
2️⃣ Install dependencies:  
```bash
pip install -r requirements.txt
```
