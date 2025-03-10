[![GitHub release](https://img.shields.io/github/v/release/KevGildea/KinePose.svg?color=white)](https://github.com/KevGildea/KinePose/releases/latest)
![Windows](https://img.shields.io/badge/platform-Windows-white?logo=windows&logoColor=black)

[![DOI](https://zenodo.org/badge/DOI/10.56541/QTUV2945.svg)](https://doi.org/10.56541/QTUV2945)

![Python Version](https://img.shields.io/badge/python-3.11%20|%203.10%20|%203.9%20|%203.8%20|%203.7-green)
![License](https://img.shields.io/github/license/KevGildea/KinePose)

![Last Commit](https://img.shields.io/github/last-commit/KevGildea/KinePose)
![GitHub stars](https://img.shields.io/github/stars/KevGildea/KinePose?style=social)


# KinePose

The KinePose framework employs a modular suite of computer vision (CV) and inverse kinematics (IK) techniques to track human body motions from image/video data.

<img src="images/IRCOBI2024_Fig1.png" alt="KinePose Framework" width="1000"/>

### Computer vision techniques
The initial step is 2D human pose estimation (2DHPE), employing a deep learning (DL) model trained on annotated image data, which localises body keypoints in pixel-space; [YOLOv8](https://github.com/ultralytics/ultralytics) trained on the MS COCO dataset is currently implemented in the pipeline. Optional manual refinement tools are also provided for enhanced accuracy. 

For monocular data, 3D human pose estimation (3DHPE) can subsequently be achieved through a DL model trained on motion capture data; [MotionBert](https://github.com/Walter0807/MotionBERT) trained on the H3.6M dataset is currently implemented in the pipeline. For multi-camera data, we have devised a weighted midpoint method (wMP) to triangulate 2D coordinates into 3D while weighting the solution by individual 2D keypoint confidence scores.

### Inverse kinematics algorithms
The IK system allows for a fully user-defined kinematic chain, accommodating various joint numbers, placements, configurations, degrees of freedom (DOFs), and ranges of motion (ROMs) as Euler/Cardan angle ranges from the rest pose for HBM-specific results.

Furthermore, specific pose and temporal loss terms, weights, and temporal patch lengths can be tailored to the needs of the analysis. See reference [10.56541/QTUV2945](https://arxiv.org/pdf/2207.12841) ([Supp. Mat.](https://kevgildea.github.io/assets/docs/KinePose/SuppMat.pdf)) for details on the algorithms. The outputs are joint orientations in both global and local coordinate systems, and angular velocities, with optional Savitzky-Golay filtering to reduce noise while preserving key motion features.

<img src="images/IRCOBI2024_Fig2.png" alt="KinePose Applications" width="1000"/>


## Installation instructions

### For users (no development required)

If you would like to use KinePose without modifying the source code, follow these steps:

#### 1. Download the executables
- Go to the [Releases page](https://github.com/KevGildea/KinePose/releases).
- Download the necessary release assets `KinePose_vX.X.X_Calibration.zip`, `KinePose_vX.X.X_2D.zip`, `KinePose_vX.X.X_3D.zip`, and `KinePose_vX.X.X_6DOF.zip`.

#### 2. Running the executables
Once the `.exe` files are downloaded, you can run them directly on your Windows machine as administrator.


### For developers (source code and customisation)

If you'd like to modify the KinePose framework, run it from source, or contribute to development, follow these steps:

#### 1. Prerequisites
Before cloning and running the project, ensure you have the following:

- **Python**: Version 3.7 to 3.11 supported.
- **Git**: For cloning the repository.
- **Git LFS (Large File Storage)**: Required for handling large `.exe` and `.bin` files. Install it from [Git LFS](https://git-lfs.github.com/).

#### 2. Cloning the repository

To get the latest version of the source code and retrieve large files tracked by Git LFS:

```bash
git lfs install
git clone https://github.com/KevGildea/KinePose.git
cd KinePose
```




## Flowchart for usage

```mermaid
graph LR;
    Calibration --> Knowable[Knowable Intrinsics: Access to camera];
    Knowable --> Intrinsics[Intrinsics.exe: Tool for intrinsic camera calibration];
    Knowable --> PnP[PnP.exe: Tool for determining camera extrinsics using known intrinsics];
    Calibration --> Unknown[Unknown Intrinsics: Unknown camera parameters];
    Unknown --> Tsai[Tsai Method: External tool for Tsai calibration method];
    
    2D --> Auto[Auto.exe: Automatic 2D pose estimation];
    2D --> SemiAuto[Semi-auto.exe: Semi-automatic 2D pose estimation];
    2D --> Manual[Manual.exe: Manual 2D pose refinement];
    
    3D --> Monocular[Monocular: Single camera 3D pose estimation];
    Monocular --> 3DMonocular[3DMonocular.exe: 3D keypoints from monocular data];
    3D --> Multicamera[Multicamera: Multi-camera 3D pose estimation];
    Multicamera --> wMP[wMP.exe: Weighted midpoint triangulation for 3D];
    
    6DOF --> KinePose[KinePose.exe: 6 degrees of freedom inverse kinematics tool];
```



## Methods papers
```
@inproceedings{Gildea22IMVIP,
    author    = {Gildea, Kevin and Mercadal-Baudart, Clara and Blythman, Richard and Smolic, Aljosa and Simms, Ciaran},
    title     = {KinePose: A temporally optimized inverse kinematics technique for 6DOF human pose estimation with biomechanical constraints},
    booktitle = {Irish Machine Vision and Image Processing Conference (IMVIP)},
    year      = {2022},
    doi       = {10.56541/QTUV2945}
}

@inproceedings{Gildea24IRCOBI,
    author    = {Gildea, Kevin and Simms, Ciaran},
    title     = {KinePose Framework for Computer Vision‐Aided Reconstruction of Pose and Motion in Human Body Models},
    booktitle = {International Research Council on Biomechanics of Injury (IRCOBI)},
    year      = {2024}
}

```

## Example applications
```
@Article{Gildea24JBiomech,
    author    = {Gildea, Kevin and Hall, Daniel and Simms, Ciaran},
    title     = {Forward dynamics computational modelling of a cyclist fall with the inclusion of protective response using deep learning-based human pose estimation},
    journal   = {Journal of Biomechanics},
    volume    = {163},
    pages     = {111959},
    year      = {2024},
    issn      = {0021-9290},
    doi       = {10.1016/j.jbiomech.2024.111959}
}

@Article{Hall24AppSci,
    author    = {Hall, Daniel and Gildea, Kevin and Simms, Ciaran},
    title     = {Drainage Troughs as a Protective Measure in Subway–Pedestrian Collisions: A Multibody Model Evaluation},
    journal   = {Applied Sciences},
    volume    = {14},
    year      = {2024},
    number    = {22},
    issn      = {2076-3417},
    doi       = {10.3390/app142210738}
}
```


## Funding

<a href="https://www.saferresearch.com/projects/advancements-kinepose-framework">
  <img src="images/SAFER.svg" alt="Funded by Organization" width="200"/>
</a>

<br>

<a href="https://portal.research.lu.se/en/projects/surrogate-measures-of-safety-for-single-bicycle-crashes">
  <img src="images/TRAFIKVERKET.png" alt="Funded by Organization" width="200"/>
</a>


## Contributing
Feel free to submit pull requests or open issues for improvements or bug fixes!

To-do lists with individual tasks are includeed in each folder, but here are few 'big'/'broad' ideas:
- [ ] Add a pose annotator tool similar to [KevGildea/yolo-pose-annotation](https://github.com/KevGildea/yolo-pose-annotation/tree/main) repo, with options for augmented/semi-automatic annotation of standard 17-keypoint COCO data, extended human pose representations, or fully custom keypoint sets, ensuring compatibility with common deep learning frameworks for pose estimation, including Ultralytics YOLOv8, MMPose, Detectron2, and MediaPipe.
- [ ] Work on adaptations for a multi-person/object tool.
- [ ] Reformat into a web app.
