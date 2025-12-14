"""
Point Cloud Data Visualization Script
Visualizes NPY files containing point cloud data from the ShapeNet dataset
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


class PointCloudVisualizer:
    """Visualize point cloud data from NPY files"""
    
    def __init__(self, data_dir="data/03001627"):
        """Initialize visualizer with data directory"""
        self.data_dir = data_dir
        self.splits = ["train", "val", "test"]
        
    def load_point_cloud(self, npy_path):
        """Load point cloud from NPY file"""
        return np.load(npy_path)
    
    def get_dataset_stats(self):
        """Print statistics about the dataset"""
        print("=" * 60)
        print("Dataset Statistics")
        print("=" * 60)
        
        for split in self.splits:
            split_dir = os.path.join(self.data_dir, split)
            if os.path.exists(split_dir):
                files = os.listdir(split_dir)
                num_files = len(files)
                print(f"\n{split.upper()}:")
                print(f"  Number of samples: {num_files}")
                
                if files:
                    # Check first file to get shape info
                    first_file = os.path.join(split_dir, files[0])
                    data = self.load_point_cloud(first_file)
                    print(f"  Point cloud shape: {data.shape}")
                    print(f"  Data type: {data.dtype}")
                    print(f"  Value range: [{data.min():.4f}, {data.max():.4f}]")
    
    def visualize_single(self, npy_path, title=None):
        """Visualize a single point cloud"""
        data = self.load_point_cloud(npy_path)
        
        if len(data.shape) == 2 and data.shape[1] == 3:
            # 3D point cloud (N, 3)
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot points
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', s=1, alpha=0.6)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            if title:
                ax.set_title(title)
            else:
                ax.set_title(os.path.basename(npy_path))
            
            plt.tight_layout()
            plt.show()
        else:
            print(f"Unexpected data shape: {data.shape}")
    
    def visualize_batch(self, split="test", num_samples=4):
        """Visualize multiple point clouds from a split"""
        split_dir = os.path.join(self.data_dir, split)
        
        if not os.path.exists(split_dir):
            print(f"Split directory not found: {split_dir}")
            return
        
        files = sorted(os.listdir(split_dir))[:num_samples]
        
        rows = (num_samples + 1) // 2
        cols = 2 if num_samples > 1 else 1
        
        fig = plt.figure(figsize=(15, 5 * rows))
        
        for idx, fname in enumerate(files, 1):
            fpath = os.path.join(split_dir, fname)
            data = self.load_point_cloud(fpath)
            
            ax = fig.add_subplot(rows, cols, idx, projection='3d')
            
            # Plot points with color gradient
            if len(data.shape) == 2 and data.shape[1] == 3:
                ax.scatter(data[:, 0], data[:, 1], data[:, 2], 
                          c=data[:, 2], cmap='viridis', s=1, alpha=0.6)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(fname[:16] + "...")
        
        plt.tight_layout()
        plt.show()
    
    def analyze_point_cloud(self, npy_path):
        """Analyze and print detailed statistics about a point cloud"""
        data = self.load_point_cloud(npy_path)
        
        print("=" * 60)
        print(f"Point Cloud Analysis: {os.path.basename(npy_path)}")
        print("=" * 60)
        print(f"Shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        print(f"Memory size: {data.nbytes / 1024:.2f} KB")
        
        if len(data.shape) == 2:
            print(f"\nStatistics by dimension:")
            for i in range(data.shape[1]):
                print(f"  Dimension {i}:")
                print(f"    Min: {data[:, i].min():.6f}")
                print(f"    Max: {data[:, i].max():.6f}")
                print(f"    Mean: {data[:, i].mean():.6f}")
                print(f"    Std: {data[:, i].std():.6f}")


def main():
    """Main function - demonstration"""
    visualizer = PointCloudVisualizer("./03001627")
    
    # Print dataset statistics
    visualizer.get_dataset_stats()
    
    # Analyze a specific file
    test_file = "./03001627/test/103b75dfd146976563ed57e35c972b4b.npy"
    if os.path.exists(test_file):
        print("\n")
        visualizer.analyze_point_cloud(test_file)
        
        # Visualize single point cloud
        print("\nVisualizing single point cloud...")
        visualizer.visualize_single(test_file)
        
        # Visualize batch
        print("\nVisualizing batch of point clouds...")
        visualizer.visualize_batch("test", num_samples=4)


if __name__ == "__main__":
    main()
