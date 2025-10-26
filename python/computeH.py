# Q2.3

import numpy as np
import cv2 

def computeH(x1: np.array, x2: np.array):
    """
    Computes H transform from image 2 to image 1 without normalization.
    ## Parameters:
        x1: feature coordinates of image 1, shape(N,2)
        x2: feature coordinates of image 2, shape(N,2)
    ## Returns:
        H transform matrix (3,3) from coordinates 2 to 1
    """
    assert x1.shape[0] == x2.shape[0] and x1.shape[1] == 2 and x2.shape[1] == 2
    n = x1.shape[0]
    # Making coordinates homogeneous
    x1h = np.pad(x1, ((0,0),(0,1)), constant_values=1) # shape(n,3)
    x2h = np.pad(x2, ((0,0),(0,1)), constant_values=1) # shape(n,3)

    # Arranging needed entries in A
    y = x1h[:, 1:2] # shape(n,1)
    x = x1h[:, 0:1] # shape(n,1)
    p = x2h.T # shape(3,n)
    zeros = np.zeros((3,n))

    A1 = np.concatenate([  zeros.T,      -p.T,       y * p.T  ], axis=1)
    A2 = np.concatenate([    p.T,       zeros.T,    -x * p.T  ], axis=1)
    A3 = np.concatenate([ -y * p.T,     x * p.T,      zeros.T ], axis=1)
    A = np.concatenate([A1,A2,A3], axis=0)
    A /= np.max(A) # normalizing 

    # Choosing the last right-singular vector as soultion for H
    U, S, VT = np.linalg.svd(A)
    H21 = np.reshape(VT[-1,:],(3,3)) 
    return H21





def get_matched_coordinates(kp1, kp2, matches):
    """
    Extract corresponding coordinates from matched keypoints
    
    Args:
        kp1: Keypoints from image1
        kp2: Keypoints from image2  
        matches: List of DMatch objects
    
    Returns:
        pts1: Coordinates from image1 (N, 2)
        pts2: Coordinates from image2 (N, 2)
    """
    points1 = []
    points2 = []
    
    for match in matches:
        # Get the keypoint coordinates
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        
        # (x, y) coordinates from first image
        x1, y1 = kp1[img1_idx].pt
        points1.append([x1, y1])
        
        # (x, y) coordinates from second image
        x2, y2 = kp2[img2_idx].pt
        points2.append([x2, y2])
    
    return np.array(points1), np.array(points2)







def transform_image_to_plane(I1: np.array, I2: np.array, computeH_func,
                              num_match = 10):
    """
    Transform images image I2 to image I1 coordinates based on the matched features.
    Feature detection and extraction is done using ORB (Oriented FAST and Rotated BRIEF)
    algorithm.

    ## Parameters
        I1: Image 1 (projected to)
        I2: Image 2 (projected from)
        computeH_func: 
            The function used for computing H (transfrom matrix)
        num_match: Number of matched points used for calculation of H
            As this number increases, there will be a better chance that correct
            matches can be detected with sampling algorithms like RANSAC

    ## returns:
        The match_image which is groundtruth image concatenated by transformed image
    """
    rows, cols, _ = I1.shape
    output_size = (cols, rows)
    # Convert images to grayscale
    I1_G = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    I2_G = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
   
    # Initiate ORB detector and extractor
    orb = cv2.ORB_create()
    
    # Detect features in both images
    kp1 = orb.detect(I1_G,None)
    kp2 = orb.detect(I2_G,None)
    # Compute the descriptors with ORB
    kp1, des1 = orb.compute(I1_G, kp1)
    kp2, des2 = orb.compute(I2_G, kp2)

    # Compute matches for two images by using the Brute Force Matcher object
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors
    matches = bf_matcher.match(des1,des2)
    # Also sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # Select top matches and compute H
    matches = matches[:num_match]
    pts1, pts2 = get_matched_coordinates(kp1,kp2,matches)
    H21 = computeH_func(pts1, pts2)

    # Project image and keypoints using the H transform       
    warped_image = cv2.warpPerspective(I2, H21, output_size )
    for kp in kp2:
        x, y = kp.pt
        kph = np.array([x,y,1]).T
        kph_new = H21 @ kph
        x_new, y_new, w = kph_new
        kp.pt = (x_new/w, y_new/w)

    # Draw matches between transformed image and its groundtruth
    match_image = cv2.drawMatches(I1, kp1, warped_image, kp2, matches, None,
                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return match_image
   







if __name__ == "__main__":
    I1 = cv2.imread('../data/hp_cover.jpg')
    I2 = cv2.imread('../data/hp_desk.png')
    # I1 = cv2.imread('../data/cv_cover.jpg')
    # I2 = cv2.imread('../data/cv_desk.png')
    match_image = transform_image_to_plane(I1, I2, computeH, 10)
    cv2.imshow("Transformed Image Match with Groundtruth", match_image)
    cv2.imwrite("../results/2_3_Match_between_Groundtruth_and_Transformed.jpg",
                match_image, )
    cv2.waitKey()

    