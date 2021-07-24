# Open3D
## 1. Main GUI
### 1.1 Multi-view Camera Streaming

### 1.2 Scene load

### 1.3 Multi-view Camera Registration
#### a. Initial Registration
- When start the app
- Use 2.2
#### b. Adaptive Registration while Streaming
- Refinement only
- Fragment registration if the camera setting changed

### 1.4 Usage
```bash
python3 main_gui.py
```

## 2. 3D Scene Reconstruction and Multi-view Mapping

### 2.1 3D Scene Reconstruction
[notion doc](https://www.notion.so/choiw/Scene-Reconstruction-01379bf6e91440f7a3760ca51e92a098)

### 2.2 Multi-view Mapping
- Generate a fragment for each view
- Apply fragment registration and refinement of 3D Scene Reconstruction

### 2.3 Usage
```bash
# multiview simulation data generation
python3 multiview_simulation_dataset.py config_o3d/stanford/lounge.json 
# run all the steps for 3D reconstruction and multi-view mapping
python3 run_system.py --make --register --refine --integrate --register_multiview config_o3d/stanford/lounge_multiview_scene.json
```
