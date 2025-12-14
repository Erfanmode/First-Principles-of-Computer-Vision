<center>
<img src="./HKUST.png" alt="HKUST LOGO" width="300">

**Department of Electronic and Computer Engineering**

**Hong Kong University of Science and Technology**
</center>

#  

### ELEC 5630 - First Principles of Cumputer Vision
*  Assignment **5**
* Professor: **TAN, Ping**

* Developed by: **Erfan RADFAR**
* **Fall 2025**

#  3D Generation: Point Cloud Diffusion Model


*A PyTorch implementation of a denoising diffusion model for 3D point cloud generation.*

This project implements a **3D point cloud generative diffusion model** using principles from DDPMs and simplified PointNet-style neural architectures. The system supports:

* Diffusion noise scheduling (cosine/linear)
* Forward & reverse diffusion processes
* A simplified PointNet-inspired noise prediction network
* High-quality 3D visualization (Open3D + Matplotlib)
* MMD/CD evaluation metrics
* ShapeNet data loading with resampling & normalization
* Training, sampling, checkpointing, and visualization tools

---

## Features

### ✔ Diffusion Engine

* Cosine or linear beta schedules
* Forward sampling (adds noise)
* Reverse denoising loop & generation

(Implemented in **DiffusionScheduler** inside *main.py* )

### ✔ PointNet-Like Noise Predictor

* Point-level shared MLPs
* Global max pooling
* Time-step conditioning
* Per-point noise prediction

(Implemented in **pointnet_model.py** )

### ✔ Data Loader (ShapeNet)

Automatically resamples any point cloud to a fixed size (2048), normalizes each cloud, and stacks into tensors.

(Function: `load_shapenet_split()` in *main.py* )

### ✔ Evaluation Metrics

Provided in **evaluation_metrics.py**:

* Chamfer Distance
* Minimum Matching Distance (MMD)
* Nearest-neighbor squared distance

### ✔ Visualization Utilities

Provided in **visualization.py**:

* Training loss curve plot
* High-quality 3D point cloud rendering using Open3D
* Color-coding by height
* Screenshot saving & interactive viewer

---

## Project Structure

```
python/
 ├── point_cloud_diffusion.py # Training, generation, scheduler, pipeline
 ├── pointnet_model.py        # PointNet noise predictor
 ├── evaluation.py            # CD, NN distance, MMD
 └── visualization.py         # Loss plots + 3D visualization with Open3D
data/
 └── 03001627/           # ShapeNet category (train/val/test)
results/
 ├── model_epoch_final.pth
 ├── model_step_xxxxxx.pth
 ├── generated_points.npy
 ├── generated_points.png
 ├── training_loss.png
 └── denoising_process.png
```

---

## Installation

```bash
pip install torch numpy matplotlib open3d
```

---

## Dataset Setup

Place ShapeNet `.npy` point clouds in:

```
../data/03001627/train/
../data/03001627/val/
../data/03001627/test/
```

Each file must have shape:

```
(N_points, 3)
```

The loader automatically:

* downsamples/upsamples to 2048 points
* normalizes mean/std
* loads into GPU tensors

(Handled in load_shapenet_split() in main.py )

---

## Training

Run:

```bash
python point_cloud_diffusion.py
```


Or edit:

```python
if __name__ == '__main__':
    main(do_train=True)
```

Outputs:

* checkpoints under `/results/`
* training loss plot (`training_loss.png`)
* final weights (`model_epoch_final.pth`)

---

## Generating New Point Clouds

To generate samples without training using saved models:

```python
main(do_train=False)
```

Produces:

```
results/generated_points.npy     # (4, 2048, 3)
results/generated_points.png     # Open3D render
results/denoising_process.png    # Visualization grid
```

---

## Evaluation Metrics

Located in **evaluation.py**:

### Chamfer Distance (CD)

Computes bidirectional nearest-neighbor squared distances.

```python
cd = chamfer_distance(pc1, pc2)
```

### Minimum Matching Distance (MMD)

Compares generated set vs. ground-truth set.

```python
mmd = minimum_matching_distance(gen, gt)
```
The final value for linear schedule and existing parameters:
```
MMD = 0.1457
```
---

## Model Overview

### PointCloudNoisePredictor (PointNet-like)

Input shape:

```
(B, N, 3)
```

Pipeline:

1. Shared MLP via Conv1D (Permutation Invariance)
2. BatchNorm + GELU
3. Global max pooling → global feature
4. Time embedding
5. Concatenation and noise-prediction MLP
6. Output:

```
(B, N, 3)
```

Details in *pointnet_model.py* .

---

## Diffusion Process

### Forward (q)

```
x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 − ᾱ_t) * ε
```

(Implemented in forward_sample() in point_cloud_diffusion.py )

### Reverse (p)

One-step denoising using ε̂ predicted by the model.
As described in https://doi.org/10.48550/arXiv.2006.11239

(Implemented in reverse_step(), generate() in point_cloud_diffusion.py)


---

## Visualization Tools (visualization.py)

### ✔ Training Loss Curve

```python
plot_training_loss(losses)
```

Saves `/results/training_loss.png`.
With `final loss = 0.254`

![](/results/training_loss.png)

### ✔ High-Quality 3D Rendering (Open3D)

(From visualization.py )

Features:

* Height-based color mapping
* Lighting, camera controls
* Interactively inspect results
* Screenshot saving

Usage:

```python
plot_point_cloud_3d(pc_array, num_samples=4)
```

Outputs in `/results/generated_points.png` for 4 samples.

![](/results/generated_points.png)

---

### Example: Visualizing a Single Point Cloud

```python
import numpy as np
from visualization import plot_point_cloud_3d

pc = np.load('../data/03001627/train/example.npy')
plot_point_cloud_3d(pc[None, :, :], num_samples=1)
```


### Visualization of Denoising process
```python
visualize_denoising_process(model, scheduler)
```
Outputs in `/results/denoising_process.png`

![](/results/denoising_process.png)


---

## File-by-File Summary

| File                      | Purpose                                                   |   |
| ------------------------- | --------------------------------------------------------- | - |
| **point_cloud_diffusion.py** | Training loop, scheduler, generation, visualization       |   |
| **pointnet_model.py**              | PointNet noise prediction network                         |   |
| **evaluation.py** | Chamfer Distance, NN distance, MMD                        |   |
| **visualization.py**      | Open3D static/interactive visualization + training curves |   |

---


## References

1 - Jonathan Ho, Ajay Jain, and Pieter Abbeel. *Denoising Diffusion Probabilistic Models.*
2020. arXiv: 2006.11239 [cs.LG]. url: https://arxiv.org/abs/2006.11239.

2 - Charles R. Qi et al. *PointNet: Deep Learning on Point Sets for 3D Classification and
Segmentation.* 2017. arXiv: 1612.00593 [cs.CV]. url: https://arxiv.org/abs/
1612.00593

3 - Charles R. Qi et al. *PointNet++: Deep Hierarchical Feature Learning on Point Sets in
a Metric Space.* 2017. arXiv: 1706.02413 [cs.CV]. url: https://arxiv.org/abs/
1706.02413

4 - Merlin Nimier-David et al. *“Mitsuba 2: A Retargetable Forward and Inverse Renderer”.*
In: Transactions on Graphics (Proceedings of SIGGRAPH Asia) 38.6 (Dec. 2019). doi:
10.1145/3355089.3356498.

