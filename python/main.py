import cv2
import numpy as np
import scipy
import matplotlib.pyplot as plt
from read_image import read_images
from find_correspondence import find_correspondence
from recover_pose import recover_camera_pose_with_E, triangulate, PnP
from bundle_adjustment import build_visibility_matrix, BundleAdjustment

data_name = "templeRing" # only 'templeRing', 'fern', and 'trex'



## images folder path
image_truncate = -1 # Use all images
match data_name:
    case "templeRing":
        images_path = "../data/templeRing"
        results_address = "../results/templeRing"
        N_matches = 10
    case "fern":
        images_path = "../data/fern/images"
        results_address = "../results/fern"
        poses_path = "../data/fern/poses_bounds.npy"
        N_matches = 15
    case "trex":
        images_path = "../data/trex/images"
        results_address = "../results/trex"
        poses_path = "../data/trex/poses_bounds.npy"
        N_matches = 8
        image_truncate = 18
    case _:
        raise(KeyError(f"data name: {data_name} is invalid"))

N_feature = 100 # number of features points extracted per image
            
adjust_step = 100 # The step for bundle adjustment (in addition to final adjustment)
                  # For each adjust_step of cameras observations,
                  #  one bundle adjustment happens.
# Read images
images, K = read_images(images_path) # K with shape: (3,3)
images = images[:image_truncate]
N = len(images) # Number of images


## 2.2 Matching
# 2.2.1 Correspondence
(x0, x1), (des0, des1), _ = find_correspondence(images[0], images[1], N_feature)

#2.2.2 Fundamental Matrix Estimation
F_rank3, mask = cv2.findFundamentalMat(x0, x1, method=cv2.FM_RANSAC,
                 ransacReprojThreshold=1.0, confidence=0.999, mask=None)
U, S, Vh = np.linalg.svd(F_rank3)
S[-1] = 0
F = U @ np.diag(S) @ Vh # Enforce rank 2 and correct F

## 2.3 Relative Pose Estimation
# 2.3.1 Essential Matrix Estimation
E = K.T @ F @ K
E = E / np.linalg.norm(E)

# 2.3.2 Camera Pose Extraction
(R, t), max_valid_points = recover_camera_pose_with_E(x0, x1, K, E, mask)

## 2.4 Triangulation
X_all = []
X = triangulate(x0, x1, np.eye(3), R, np.zeros(3), t, K) # shape: (N, 3)

# initialize Xj, X_all, camera_poses
Xj = X
desj = des1.copy()
X_all = np.vstack((X,X))
rvecj, _ = cv2.Rodrigues(R)
tvecj = t
desck = des1.copy()
Rj = R.copy()
camera_poses = [(np.zeros(3), np.zeros(3)),
                (rvecj.flatten(), tvecj.flatten())]
observations_2d = np.vstack((x0,x1))

# initialize bundle adjuster
BA = BundleAdjustment(K)


for i in range(2,N):
    # j is equal to i-1, but for easier identification
    # we use separate variables.

    (xj, xi), (_, desi), _= find_correspondence(images[i-1], images[i], N_feature)
    rveci, tveci, Ri, _  = PnP(Xj, xi, desj, desi, N_matches, K)
    Xi = triangulate(xj, xi, Rj, Ri, tvecj, tveci, K)
    
    # Stack data
    observations_2d = np.vstack((observations_2d, xi))
    camera_poses.append((rveci.flatten(), tveci.flatten()))
    X_all = np.vstack((X_all, Xi))
    Rj = Ri.copy()
    Xj = Xi.copy()
    tvecj = tveci.copy()
    desj = desi.copy()

    # Build Visibility Matrix
    V = build_visibility_matrix(i+1, N_feature)
    # Bundle adjustment based on observed data till now
    # Optimized poses and 3D points are computed in each iteration
    # Adjustment happens for every adjust_step observations and once at the end
    if i % adjust_step == 0 or i==N-1 and adjust_step<N:
        camera_poses, X_all = \
            BA.bundle_adjust(X_all, camera_poses, observations_2d, V)
    print(f"for observation {i}, mean x is : {np.mean(Xj, axis=0)}")
    







# camera poses: groundtruth vs estimated
camera_poses_np = np.zeros((len(camera_poses),6))
for i, (rvec,tvec) in enumerate(camera_poses):
    camera_poses_np[i,:3] = rvec.flatten()
    camera_poses_np[i,3:] = tvec.flatten()

# Groundtruth camera poses
if data_name == "templeRing":
    g_camera_poses = np.zeros((N,6))
else:
    poses_bounds = np.load(poses_path)
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # Shape: (N, 3, 5)
    bounds = poses_bounds[:, -2:]                   # Shape: (N, 2)
    # Extract individual components
    R = poses[:, :3, :3]   # Rotation matrix (3x3)
    rvecs = []
    for i in range(N):
        rvec, _ = cv2.Rodrigues(R[i]) 
        rvecs.append(rvec.flatten())
    rvecs = np.array(rvecs)# Rotation vector (Nx3x1)
    T = poses[:, :3, 3]    # Translation vector (Nx3x1)
    g_camera_poses = np.hstack((np.squeeze(rvecs), np.squeeze(T[:image_truncate])))

rot_RMSE = np.mean((camera_poses_np[:,:3] - g_camera_poses[:,:3])**2, axis = 0)
trans_RMSE = np.mean((camera_poses_np[:,3:] - g_camera_poses[:,3:])**2, axis = 0)

# Plot rotation vectors
fig_rot, axs_rot = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
components = ['X', 'Y', 'Z']
for i, comp in enumerate(components):
    axs_rot[i].plot(g_camera_poses[:, i], label='Ground Truth', linewidth=2)
    axs_rot[i].plot(camera_poses_np[:, i], label='Estimation', linestyle='--', linewidth=2)
    axs_rot[i].set_ylabel(f'Rotation {comp} (rad)')
    axs_rot[i].legend()
    axs_rot[i].grid(True)
axs_rot[2].set_xlabel('Sample index')
fig_rot.suptitle(f'Rotation Vector Comparison of {data_name},'
+f'RMSE = x:{rot_RMSE[0]:.2}, y:{rot_RMSE[1]:.2}, z:{rot_RMSE[2]:.2}', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(results_address + f"/RotComparison_{data_name}.jpg")
plt.show()


# Plot translation vectors
fig_trans, axs_trans = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for i, comp in enumerate(components):
    axs_trans[i].plot(g_camera_poses[:, i+3], label='Ground Truth', linewidth=2)
    axs_trans[i].plot(camera_poses_np[:, i+3], label='Estimation', linestyle='--', linewidth=2)
    axs_trans[i].set_ylabel(f'Translation {comp} (pixel)')
    axs_trans[i].legend()
    axs_trans[i].grid(True)
axs_trans[2].set_xlabel('Sample index')
fig_trans.suptitle(f'Translation Vector Comparison of {data_name},'
+f'RMSE = x:{trans_RMSE[0]:.2}, y:{trans_RMSE[1]:.2}, z:{trans_RMSE[2]:.2}', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(results_address + f"/TransComparison_{data_name}.jpg")
plt.show()








# Plot 3D points
fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(111, projection='3d')
X_mean = np.mean(X_all, axis=0)
X_std = np.std(X_all, axis=0)
ax1.scatter(X_all[:, 0], X_all[:, 1], X_all[:, 2], 
            c='blue', s=1, alpha=0.6, label='3D Points')
ax1.set_title('Before Bundle Adjustment')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
limit = np.mean(X_std)*4
ax1.set_xlim(-limit + X_mean[0], limit + X_mean[0])
ax1.set_ylim(-limit + X_mean[1], limit + X_mean[1])
ax1.set_zlim(-limit + X_mean[2], limit + X_mean[2])
ax1.legend()
plt.tight_layout()

# Record Image Sequence
for i, elev in enumerate(range(-180,180,5)):
    ax1.view_init(elev=elev, azim=-90)
    plt.savefig(results_address + f"/{data_name}_{i}.jpg")

ax1.view_init(elev=90, azim=-90)
plt.show()


# 2.7 Non-linear Optimization (Optional)
"""
For non-linear optimization, SQPNP was used based on the
`George Terzakis and Manolis Lourakis. A consistently fast and globally optimal
solution to the perspective-n-point problem. In European Conference on Computer Vision, pages 478–494.
Springer International Publishing, 2020.` work.
They formulated the PnP problem as a non-linear quadratic program and then used
a specific approach to guarantee finding the global minimum,
unlike many other methods that get stuck in local minima. This algorithm was used
in this program for PnP computation.
"""

