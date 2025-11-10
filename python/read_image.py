import cv2
import numpy as np
import os

def read_images(folder_path):
    """
    Reads all image files from a specified folder and returns them as a list of OpenCV images.

    Args:
        folder_path (str): The path to the folder containing the images.

    Returns:
        images: A list of OpenCV image matrices (numpy arrays).
        camera_matrix: Camera specifications

    """
    images = []
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp') 

    current_directory = os.getcwd()
    folder = os.path.join(current_directory, folder_path)
    folder = os.path.normpath(folder)

    # Read images
    for filename in os.listdir(folder):
        if filename.lower().endswith(supported_extensions):
            file_path = os.path.join(folder_path, filename)
            img = cv2.imread(file_path)
            img = cv2.resize(img, (640,480))
            if img is not None:  # Check if the image was loaded successfully
                images.append(img)
            else:
                print(f"Warning: Could not load image {filename}. Skipping.")
    
    # Read camera specifications
    try: 
        camera_path = os.path.join(folder, "./camera.txt")
        camera_path = os.path.normpath(camera_path)
        camera_matrix = np.loadtxt(camera_path)
    except:
        pass

    try: 
        camera_path = os.path.join(folder, "../poses_bounds.npy")
        camera_path = os.path.normpath(camera_path)
        pose_bounds = np.load(camera_path)
        # The original 3x5 matrix format is [R|T|hwf]
        poses = pose_bounds[:, :-2].reshape([-1, 3, 5])
        # Extract the last 2 elements (near and far depth bounds)
        bounds = pose_bounds[:, -2:]
        h, w = poses[0,0:2,4]
        f = poses[0,2,4] # Focal length in pixels
        camera_matrix = np.array([[f, 0, w/2],
                                  [0, f, h/2],
                                  [0, 0,  1 ]])
    except: 
        pass

    return images, camera_matrix

