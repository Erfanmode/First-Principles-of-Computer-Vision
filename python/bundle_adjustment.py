import numpy as np
import cv2
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def build_visibility_matrix(n_cameras, n_points_per_cam):
    V = np.zeros((n_cameras, n_points_per_cam*(n_cameras-1)))
    for i in range(n_cameras-1):
        if i == 0:
            V[i, i:n_points_per_cam*(i+2)-1] = 1
        else:
            V[i, n_points_per_cam*(i-1):n_points_per_cam*(i+1)] = 1

    return V

class BundleAdjustment:
    def __init__(self, camera_matrix, dist_coeffs=None):
        self.K = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)
        
    def project_points(self, points_3d, rvec, tvec):
        """
        Project 3D points to 2D image coordinates
        """
        points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, self.K, self.dist_coeffs)
        return points_2d.reshape(-1, 2)
    
    def reprojection_error(self, params, n_cameras, n_points, visibility_matrix, observations_2d):
        """
        Compute reprojection error 
        
        Args:
            params: Flattened array of [camera_params, points_3d]
            n_cameras: Number of cameras
            n_points: Number of 3D points
            visibility_matrix: Boolean matrix (n_cameras, n_points) indicating visibility
            observations_2d: Matrix (n_points, 2) containing 2D observations
            
        Returns:
            errors: Flattened array of reprojection errors for all visible points
        """
        # Extract camera parameters (each camera has 6 parameters: rvec(3) + tvec(3))
        camera_params = params[:n_cameras * 6].reshape(n_cameras, 6)
        
        # Extract 3D points
        points_3d = params[n_cameras * 6:].reshape(n_points, 3)
        
        errors = []
        
        # Iterate through all camera-point pairs
        for cam_idx in range(n_cameras):
            # Get camera parameters
            rvec = camera_params[cam_idx, :3]
            tvec = camera_params[cam_idx, 3:6]
            
            # Find points visible in this camera
            visible_points_idx = np.where(visibility_matrix[cam_idx])[0]
            
            if len(visible_points_idx) == 0:
                continue
            
            # Get 3D points visible in this camera
            points_visible = points_3d[visible_points_idx]
            
            # Project 3D points to 2D
            points_2d_proj, _ = cv2.projectPoints(
                points_visible, rvec, tvec, self.K, self.dist_coeffs
            )
            points_2d_proj = points_2d_proj.reshape(-1, 2)
            
            # Get observed 2D points
            points_2d_obs = observations_2d[visible_points_idx]
            
            # Compute reprojection errors
            error = points_2d_proj - points_2d_obs
            errors.extend(error.flatten())
        
        return np.array(errors)
    
    def bundle_adjust(self, points_3d, camera_poses, observations_2d, visibility_matrix):
        """
        Perform bundle adjustment using visibility matrix approach
        """
        print("Bundle adjustment initiated...")
        n_cameras = len(camera_poses)
        n_points = len(points_3d)
        
        # Prepare initial parameters
        initial_params = []
        
        # Add camera parameters
        for rvec, tvec in camera_poses:
            initial_params.extend(rvec.flatten())
            initial_params.extend(tvec.flatten())
        
        # Add 3D points
        initial_params.extend(points_3d.flatten())
        
        initial_params = np.array(initial_params)
        
        # Run optimization
        result = least_squares(
            fun=self.reprojection_error,
            x0=initial_params,
            args=(n_cameras, n_points, visibility_matrix, observations_2d),
            method='lm',
            verbose=2,
            ftol=1e-1,
            xtol=1e-1,
            max_nfev=10
        )
        
        # Extract optimized parameters
        optimized_params = result.x
        camera_params_optim = optimized_params[:n_cameras * 6].reshape(n_cameras, 6)
        points_3d_optim = optimized_params[n_cameras * 6:].reshape(n_points, 3)
        
        # Convert back to camera poses format
        optimized_poses = []
        for i in range(n_cameras):
            rvec = camera_params_optim[i, :3]
            tvec = camera_params_optim[i, 3:6]
            optimized_poses.append((rvec, tvec))

        print("Bundle adjustment finished")
        return optimized_poses, points_3d_optim



