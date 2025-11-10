import cv2
import numpy as np




def triangulate(points1, points2, R1, R2, t1, t2, K):
    """
    Returns 3D coordinates of feature points in
    the format of [x,y,z,1].T
    """
    # Create projection matrices
    P1 = K @ np.hstack((R1, t1.reshape(3,1)))
    P2 = K @ np.hstack((R2, t2.reshape(3,1)))
    # Triangulate points
    points4D = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
    points3D = points4D / points4D[3]
    return points3D[:-1].T



def check_cheirality(R, t, points1, points2, K):
    """
    Check which points are in front of both cameras (positive depth)
    Returns: number of points with positive depth in both views
    """
    points3D = triangulate(points1, points2, np.eye(3), R, np.zeros(3), t, K)
    
    # Check depth in first camera
    depths_cam1 = points3D[2]
    
    # Transform to second camera coordinates and check depth
    points3D_cam2 = R @ points3D[:3] + t.reshape(3,1)
    depths_cam2 = points3D_cam2[2]
    
    # Count points with positive depth in both cameras
    valid_points = np.sum((depths_cam1 > 0) & (depths_cam2 > 0))
    
    return valid_points




def recover_camera_pose_with_E(points1, points2, K, E, mask):
    """
    Complete pipeline to recover the correct camera pose
    """
    
    # 1. Get inlier points using the mask
    points1 = points1[mask.ravel() == 1]
    points2 = points2[mask.ravel() == 1]
    
    # 2. Recover all possible poses
    R1, R2, t_vec = cv2.decomposeEssentialMat(E)
    
    poses = [
        (R1,  t_vec), 
        (R1, -t_vec),
        (R2,  t_vec), 
        (R2, -t_vec)
    ]
    
    # 3. Find the correct pose using cheirality check
    best_pose = (np.diag([1,1,1]), t_vec)
    max_valid = 0
    
    for R, t in poses:        
        valid_points = check_cheirality(R, t, points1, points2, K)
        
        if valid_points > max_valid:
            max_valid = valid_points
            best_pose = (R, t)
    
    return best_pose, max_valid




def PnP(object_points, image_points, descriptor_obj, descriptor_img, N_matches, K):
    """
    Computes and returns transformation matrix of 4x4 using Perspective-n-Point algorithm
    using SQPNP. In this function, we first match object points to image points.
    ### args:
    `objectPoints`: An array of 3D points in the object's coordinate space. This is a Nx3 matrix,
    where N is the number of points.\n
    `imagePoints`: An array of corresponding 2D points in the image plane (in pixels).
    This is an Nx2 matrix.\n
    `descriptor_obj`: Object points descriptor\n
    `descriptor_img`: Image points descriptor\n
    `N_matches`: Number of top matches used for pose estimation\n
    `K`: The camera intrinsic matrix (3x3),
    which contains the focal length (fx, fy) and the optical center (cx, cy)

    
    ### Returns:
    Rotation vector, Translation vector,
    Rotation matrix, Transformation matrix

    """

    # Match corresponding object points to image points
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) 
    matches = bf.match(descriptor_obj, descriptor_img)
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    if N_matches < len(matches):
        matches = matches[:N_matches]
    # Else all matches are used

    matched_object_points = []
    matched_image_points = []
    
    for match in matches:
        obj_idx = match.queryIdx
        img_idx = match.trainIdx
        matched_object_points.append(object_points[obj_idx])
        matched_image_points.append(image_points[img_idx])

    matched_object_points = np.array(matched_object_points)
    matched_image_points = np.array(matched_image_points)
        
    success, rvec, tvec = cv2.solvePnP(matched_object_points, matched_image_points,
            K, distCoeffs=None, flags=cv2.SOLVEPNP_SQPNP)

    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.flatten()
    # Create 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return rvec, tvec, R, T

