# Data for ELEC5630 Assignment5 - ShapeNet 03001627 (Chair)

## Overview

This directory contains point cloud data for the **03001627** category (Chair) from the ShapeNet dataset. The data is organized into three splits: **train**, **val**, and **test**.
## Directory Structure

```
03001627/
├── train/          # Training samples
├── val/            # Validation samples
└── test/           # Testing samples
```

## Data Format

### Object Variations in Chairs
The dataset contains various chair types and poses:
- Different chair designs (office, dining, rocking, etc.)
- Various orientations and rotations
- Different chair sizes and proportions
- Includes chairs with and without armrests

### Data Structure

Each `.npy` file contains a single point cloud represented as a 2D NumPy array:

- **Shape**: `(N, 3)` where:
  - `N`: Number of points in the point cloud (typically 1024 or 2048 points)
  - `3`: XYZ coordinates (x, y, z)

- **Data Type**: `float32` (32-bit floating point)

### Example

```python
import numpy as np

# Load a point cloud
data = np.load('./03001627/test/103b75dfd146976563ed57e35c972b4b.npy')

print(f"Shape: {data.shape}")        # Output: (1024, 3) or similar
print(f"Type: {data.dtype}")         # Output: float32
print(f"Min: {data.min()}")          # Output: ~-0.5
print(f"Max: {data.max()}")          # Output: ~0.5

# Access coordinates
x_coords = data[:, 0]  # X coordinates
y_coords = data[:, 1]  # Y coordinates
z_coords = data[:, 2]  # Z coordinates
```


## Usage Examples

### Basic Loading

```python
import numpy as np

# Load single point cloud
point_cloud = np.load('./03001627/test/sample.npy')
print(point_cloud.shape)  # (N, 3)
```

### Visualization

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.load('./03001627/test/sample.npy')

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', s=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
```

### Batch Loading

```python
import numpy as np
import os
from pathlib import Path

def load_split(split_name='train'):
    split_dir = f'./03001627/{split_name}'
    point_clouds = []
    
    for file in sorted(os.listdir(split_dir)):
        if file.endswith('.npy'):
            path = os.path.join(split_dir, file)
            data = np.load(path)
            point_clouds.append(data)
    
    return point_clouds

# Load all training samples
train_data = load_split('train')
print(f"Loaded {len(train_data)} training samples")
```

## Visualization Tool

A visualization script is provided in `visulization.py`:

```bash
python visulization.py
```

This script provides:
- Dataset statistics
- Single point cloud visualization
- Batch visualization of multiple samples
- Detailed point cloud analysis

## Notes

- All coordinates are in the same scale (typically normalized to unit sphere)
- No color or additional features are provided; only XYZ coordinates
- Point order is arbitrary and should not be relied upon

## References

- **ShapeNet Dataset**: https://shapenet.org/
- **Category 03001627**: Chair

```
@techreport{shapenet2015,
  title       = {{ShapeNet: An Information-Rich 3D Model Repository}},
  author      = {Chang, Angel X. and Funkhouser, Thomas and Guibas, Leonidas and Hanrahan, Pat and Huang, Qixing and Li, Zimo and Savarese, Silvio and Savva, Manolis and Song, Shuran and Su, Hao and Xiao, Jianxiong and Yi, Li and Yu, Fisher},
  number      = {arXiv:1512.03012 [cs.GR]},
  institution = {Stanford University --- Princeton University --- Toyota Technological Institute at Chicago},
  year        = {2015}
}
```