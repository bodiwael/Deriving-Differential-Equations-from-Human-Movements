## [KinePose.exe](Dist/KinePose.exe)

This tool uses inverse kinematics (IK) to map 3D human pose data onto a user-defined kinematic chain. Run on a Windows machine as administrator.

### Key features:

- **Load kinematic chain**: Load the kinematic chain (in Excel format) for the human body model (HBM), defining joint orientations and positions, and joint ranges of motion (ROMs).
- **Load 3D keypoints**: Load 3D pose keypoints (in `.txt` format).
- **Inverse kinematics**: Perform frame-based or temporal IK optimisation to align the kinematic chain with the loaded 3D pose keypoints.
- **Motion smoothing**: Apply the Savitzky-Golay filter to smooth IK results across frames.
- **Visualisation**: Visualise both the kinematic chain and the pose data, with options to display animations using a slider.
- **Save results**: Save the transformation data (local/global orientations, positions, and angular velocities) to Excel files for further analysis.

Video demo available here: [VideoDemo_x175.mp4](Demo/VideoDemo_x175.mp4).

### Outputs:
- **Transformation data**: Saves local and global joint orientations and positions in `.xlsx` files.
- **Angular velocities**: Saves angular velocities for each joint in `.xlsx` format.

### GUI

<p align="center">
  <img src="https://github.com/KevGildea/KinePose/blob/main/images/KinePose1.PNG" alt="KinePose.exe" width="400">
  <br>
  <i>The GUI of the KinePose.exe application.</i>
</p>

<p align="center">
  <img src="https://github.com/KevGildea/KinePose/blob/main/images/KinePose2.PNG" alt="KinePose.exe" width="400">
  <br>
  <i>Visualisation of the user-defined kinematic chain in the KinePose.exe application.</i>
</p>

<p align="center">
  <img src="https://github.com/KevGildea/KinePose/blob/main/images/KinePose3.PNG" alt="KinePose.exe" width="500">
  <br>
  <i>Visualisation of the 3D pose in the KinePose.exe application.</i>
</p>

<p align="center">
  <img src="https://github.com/KevGildea/KinePose/blob/main/images/KinePose4.PNG" alt="KinePose.exe" width="400">
  <br>
  <i>Visualisation of pose loss vectors and keypoint indices, and optional setting of initial guess for optimisation in the KinePose.exe application.</i>
</p>

<p align="center">
  <img src="https://github.com/KevGildea/KinePose/blob/main/images/KinePose5.PNG" alt="KinePose.exe" width="500">
  <br>
  <i>Visualisation of the optimised 6DOF pose in the KinePose.exe application.</i>
</p>


<p align="center">
  <img src="https://github.com/KevGildea/KinePose/blob/main/images/chains.png" alt="KinePose chains" width="700">
  <br>
  <i>Examples of the variety of kinematic chain configurations that may be specified.</i>
</p>



## Contributing
Feel free to submit pull requests or open issues for improvements or bug fixes!

### To Do:
- [ ] Document the formatting of user inputs, and outputs.
- [ ] Document and add to common HBM models.
- [ ] Add user options for smoothing (user-defined, and informed by function value).
- [ ] Fix threading issue.
- [ ] Optional input of a custom pose.
- [ ] Add linear positions and velocities to output.
- [ ] Add live visualisation of the loss terms, and a final visualisation of convergence (plotting of guesses at various iterations for each frame accross the motion sequence)
- [ ] Add synthetic data for assessing the algorithm for speed and accuracy (overall and in boundary cases)
- [ ] Investigate Natural and Fully Cartesian Coordinates as an alternative to Relative (recursive) DH formulation for handling closed-loop kinematics. Compare to DH solution with proximity constraints. Resources: [Roupa2022](https://arcade.inesc-id.pt/papers/Roupa2022_Article_OnTheModelingOfBiomechanicalSy.pdf), [Andersen2007](https://elearning.uniroma1.it/pluginfile.php/71316/mod_resource/content/1/2009%20ANDERSEN%20Kinematic%20analysis%20of%20over-determinate%20biomechanical%20system.pdf)).
- [ ] Allow for setting different temporal weights between joints.
- [ ] Try implementing a KF approach.
- [ ] Add functionality for custom loss terms to the IK tool, and allow for special physical settings, e.g., nonholonomic constraints for rolling disks.
- [ ] Add options for translational joints.
- [ ] Add options to set ROMs about specific axes not corresponding to the coordinate system axes.
- [ ] Add options for rotation order.
- [ ] Implement reduced precision (float32) in optimisation to improve speed.
- [ ] Explore parallelisation or alternative optimizers (L-BFGS-B, CMA-ES) for efficiency.
