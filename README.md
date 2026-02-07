<!-- PROJECT SHIELDS -->
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![stability-experimental](https://img.shields.io/badge/stability-experimental-orange.svg)](https://github.com/emersion/stability-badges#experimental)

<!-- PROJECT LOGO -->
<p align="center">
  <img src="https://github.com/tiagomfmadeira/Meshtrics/blob/main/meshtrics_logo.png" width="550"><br>
  <b style="bold">Objective Quality Assessment of Textured 3D Meshes for 3D Reconstruction</b>
</p>

---

### [`test_photometric_full_reference_metrics.py`](https://github.com/tiagomfmadeira/Meshtrics/blob/main/tests/test_photometric_full_reference_metrics.py)

Evaluates the visual quality of a 3D mesh using **image-based full-reference metrics**, comparing renders against ground-truth photographs.

- Loads a mesh and camera intrinsics (from `.e57` or text file)
- User-assisted correspondence selection between photos and 3D model
- Estimates and refines camera pose
- Generates simulated viewpoints and compares them with reference photos

**Usage**
```bash
python test_photometric_full_reference_metrics.py \
  -photos ./ground_truth_photos \
  -pext .jpg \
  -mesh ./models/mesh.ply \
  -K ./camera_intrinsics.txt \
  -o ./output \
  -show
```

**Arguments**
- `-photos`, `--input_photos_directory` (required): directory containing ground-truth photos  
- `-pext`, `--photo_extension` (required): photo extension (e.g. `.jpg`)  
- `-mesh`, `--input_mesh` (required): mesh file (`.ply`, `.obj`, etc.)  
- `-e57`, `--input_e57` (optional): `.e57` file containing camera info  
- `-K`, `--intrinsics` (optional): camera intrinsics file (required if `-e57` not provided)  
- `-o`, `--output_path` (optional): output folder (default: current directory)  
- `-show`, `--show_visual_feedback` (optional): enables visualization windows  

---

### [`test_topological_metrics.py`](https://github.com/tiagomfmadeira/Meshtrics/blob/main/tests/test_topological_metrics.py)

Computes **topological and geometric quality metrics** for a 3D mesh.

- Vertex/face statistics and mesh area
- Smoothness estimation from adjacent face ratios
- Aspect ratio and skewness analysis
- Skewness histogram export
- Hole detection and outline perimeter analysis

**Usage**
```bash
python test_topological_metrics.py   -mesh ./models/mesh.ply   -o ./output
```

**Arguments**
- `-mesh`, `--input_mesh` (required): mesh file (`.ply`, `.obj`, etc.)  
- `-o`, `--output_path` (optional): output folder (default: current directory)  

---

### [`test_photometric_no_reference_metrics.py`](https://github.com/tiagomfmadeira/Meshtrics/blob/main/tests/test_photometric_no_reference_metrics.py)

Evaluates visual quality using **no-reference photometric metrics** (entropy-based), allowing comparison of the same region across two meshes.

- Renders both meshes using consistent camera parameters
- Interactive ROI selection
- Entropy metric comparison between the two models

**Usage**
```bash
python test_photometric_no_reference_metrics.py   -mesh1 ./models/mesh1.ply   -mesh2 ./models/mesh2.ply   -o ./output   -show
```

**Arguments**
- `-mesh1`, `--input_mesh1` (required): first mesh  
- `-mesh2`, `--input_mesh2` (required): second mesh  
- `-o`, `--output_path` (optional): output folder (default: current directory)  
- `-show`, `--show_visual_feedback` (optional): enables visualization windows  

---

## Citation

If you use **Meshtrics** in your work, please cite:

```bibtex
@inproceedings{10.2312:stag.20241351,
  booktitle = {Smart Tools and Applications in Graphics - Eurographics Italian Chapter Conference},
  title     = {{Meshtrics: Objective Quality Assessment of Textured 3D Meshes for 3D Reconstruction}},
  author    = {Madeira, Tiago and Oliveira, Miguel and Dias, Paulo},
  year      = {2024},
  publisher = {The Eurographics Association},
  doi       = {10.2312/stag.20241351}
}
```

---

## License

Distributed under the **GPL-3.0 License**. See [`LICENSE`](LICENSE) for more information.
