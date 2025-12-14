# python/visualization.py

import matplotlib.pyplot as plt
import open3d as o3d
from matplotlib import cm
import numpy as np

def plot_training_loss(losses, save_path='../results/training_loss.png'):
    """
    Plots and saves the training loss curve.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-', color='b')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average MSE Loss')
    plt.grid(True)
    plt.savefig(save_path)
    print(f"\n[INFO] Training loss plot saved to {save_path}")
    plt.close()


def plot_point_cloud_3d(pc_array, num_samples, save_path='../results/generated_points.png', 
                               interactive=True, window_width=1200, window_height=800):
    """
    Visualizes the generated point cloud results in 3D using Open3D.
    Provides high-quality rendering with lighting and materials.
    
    Args:
        pc_array (np.ndarray): Point cloud array to visualize (N_samples, N_POINTS, 3).
        num_samples (int): The number of samples to plot.
        save_path (str): Path to save the image.
        interactive (bool): If True, keeps window open for interaction.
        window_width (int): Width of the visualization window.
        window_height (int): Height of the visualization window.
    """
    # TODO: Implement the visualization logic here, you can use matplotlib/open3d/mistuba for 3D plotting or any other library of your choice.
    # If using mitsuba, please refer https://github.com/stevenygd/PointFlow?tab=readme-ov-file
    # Limit to requested number of samples
    pc_array = pc_array[:num_samples]
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=window_width, height=window_height, visible=True)
    
    # Add each point cloud
    geometries = []
    for i, points in enumerate(pc_array):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Color points by height (y-coordinate)
        colors = cm.viridis((points[:, 1] - points[:, 1].min()) / 
                           (points[:, 1].max() - points[:, 1].min()))[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Translate each sample for better visibility
        translation = [(-num_samples/2+i) * 6.0, 0, 0]  # Shift along x-axis
        pcd.translate(translation)
        
        vis.add_geometry(pcd)
        geometries.append(pcd)
    
    # Set camera parameters for better view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([1, 0, 0.1])
    ctr.set_lookat([num_samples, 0, 0])
    ctr.set_up([0, 1, 0])
    
    # Set render options for better visualization
    render_option = vis.get_render_option()
    render_option.point_size = 4.0
    render_option.background_color = np.asarray([1, 1, 1])  # White background
    render_option.light_on = True

    if interactive:
        # Keep window open until user closes it
        print("Open3D window opened. Press 'Q' or close window to exit.")
        print("Controls:")
        print("  - Mouse drag: Rotate")
        print("  - Ctrl + Mouse drag: Pan")
        print("  - Mouse wheel: Zoom")
        print("  - R: Reset view")
        print("  - Q: Exit")
        
        vis.run()  # This blocks until window is closed

    if save_path:
        # Capture screenshot
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(save_path)
        print(f"Saved Open3D visualization to {save_path}")
    
    vis.destroy_window()
    
    return geometries  # Return geometries for further processing if needed

if __name__ == "__main__":
    # file_path = '../data/03001627/train/1ab4c6ef68073113cf004563556ddb36.npy'
    # file_path = '../data/03001627/train/1b6c268811e1724ead75d368738e0b47.npy'
    file_path = '../data/03001627/train/364a43c9f94f315496db593b49da23e5.npy'
    pc_array = np.load(file_path)
    plot_point_cloud_3d(pc_array[None,:,:], 1)