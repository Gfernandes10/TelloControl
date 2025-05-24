# TelloControl ROS2 Workspace

This repository contains the ROS2 workspace for simulation and control of the Tello drone using Gazebo and ROS2 Foxy.

## How to use this workspace

### 1. Install all system and ROS2 dependencies
Run the installation script (only once, or whenever setting up on a new machine):

```bash
bash install.sh
```

### 2. Build the workspace
Use the build script to compile all packages and set up the environment:

```bash
bash 01_Builds/build_tello_gazebo.sh
```

---

> **Note:**
> - Do not commit external packages cloned in `src/` (they are already in `.gitignore`).
> - To add or remove packages, edit the `01_Builds/repos.yaml` file.
> - To update packages, run `vcs pull src`.

---

If you have any questions, check the README files of the packages in `src/` or open an issue in this repository.
