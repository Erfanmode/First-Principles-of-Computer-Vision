# Q2.5

import numpy as np
import cv2
from computeH import get_matched_coordinates
from computeH_norm import computeH_norm
from scipy.special import comb
np.random.seed(14)





def computeH_ransac(x1: np.array, x2: np.array):
    """
    Computes H transform from image 2 to image 1 applying RANSAC.
    The best-case is when the highest number of inliers are detected.
    ## Parameters:
        x1: (np.array)
          feature coordinates of image 1, shape(N,2)
        x2: (np.array)
          feature coordinates of image 2, shape(N,2)
    ## Returns:
        H21_best:
          'Best-case H transform matrix (3,3) from coordinates 2 to 1'
        fitting_idx_best:
          'Best-case 4 sampled pairs used for inliers detection and H fitting'
        best_case_inliers:
          'Best-case detected inliers'
    """
    N = x1.shape[0]
    max_trial = max(int(comb(N,4)/20), 100)
    error_threshold = 3 # error threshold for identifying inliers
    x1h = np.pad(x1, ((0,0),(0,1)), constant_values=1) # shape(N,3)
    x2h = np.pad(x2, ((0,0),(0,1)), constant_values=1) # shape(N,3) 

    max_recorded_inliers = -1 
    fitting_idx_best = None
    H21_best = None
    best_case_inliers = None

    for trial in range(max_trial):
        try:
            fitting_idx = np.random.choice(N, 4, replace=False)
            x1_sampled = x1[fitting_idx]
            x2_sampled = x2[fitting_idx]

            H21 = computeH_norm(x1_sampled, x2_sampled)
            x1h_p = H21 @ x2h.T #(3,N)
            w = x1h_p[2,:]
            errors = np.linalg.norm(x1h_p/w - x1h.T, axis=0) #(1,N)

            # Identify inliers
            inliers = errors < error_threshold
            n_inliers = np.sum(inliers)
            
            if max_recorded_inliers < n_inliers:
                H21_best = H21
                fitting_idx_best = fitting_idx
                max_recorded_inliers = n_inliers
                best_case_inliers = np.where(inliers > 0)
        except:
            continue

    return H21_best, fitting_idx_best, best_case_inliers[0]








def draw_matches_custom(img1, kp1, img2, kp2, matches, colors, **kwargs):
    """
    Draw matches with individual colors for each match
    
    Args:
        img1, img2: Input images
        kp1, kp2: Keypoints
        matches: List of DMatch objects
        colors: List of BGR colors for each match
    """
    
    # Create output image
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    
    # Draw each match with its specific color
    for match, color in zip(matches, colors):
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        
        if img1_idx < len(kp1) and img2_idx < len(kp2):
            # Get keypoint coordinates
            (x1, y1) = kp1[img1_idx].pt
            (x2, y2) = kp2[img2_idx].pt
            
            # Convert to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw circles and line
            cv2.circle(vis, (x1, y1), 4, color, 1)
            cv2.circle(vis, (x2 + w1, y2), 4, color, 1)
            cv2.line(vis, (x1, y1), (x2 + w1, y2), color, 1)
    
    return vis









def transform_image_to_plane_RANSAC(I1: np.array, I2: np.array, num_match = 10):
    """
    Transform images image I2 to image I1 coordinates based on the matched features.
    Feature detection and extraction is done using ORB (Oriented FAST and Rotated BRIEF)
    algorithm.
    The only difference between this version and 'transform_image_to_plane' is that
    it demonstrates inliers and sampled matches (which were used to detect inliers)
    separately in the final match_image

    ## Parameters
        I1: Image 1 (projected to)
        I2: Image 2 (projected from)
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
    H21, fitting_idx_best, best_case_inliers  = computeH_ransac(pts1, pts2)

    # Define specific color for best-case detected inliers and 4 sampled pairs of points
    matches_inliers = [matches[i] for i in best_case_inliers.tolist()]
    sampled_fitting_pairs = [matches[i] for i in fitting_idx_best.tolist()]
    matches_correct = [*sampled_fitting_pairs, *matches_inliers]
    colors1 = [(100,0,0) for i in range(len(matches_inliers))] # inliers matches (blue)
    colors2 = [(0,200,0) for i in range(len(sampled_fitting_pairs))] # fitting samples (green)
    colors = [*colors1, *colors2]

    # Project image and keypoints using the H transform       
    warped_image = cv2.warpPerspective(I2, H21, output_size )
    for kp in kp2:
        x, y = kp.pt
        kph = np.array([x,y,1]).T
        kph_new = H21 @ kph
        x_new, y_new, w = kph_new
        kp.pt = (x_new/w, y_new/w)

    # Draw matches between transformed image and its groundtruth
    match_image = draw_matches_custom(I1, kp1, warped_image, kp2, matches_correct, colors)
    
    return match_image







if __name__ == "__main__":
    I1 = cv2.imread('../data/hp_cover.jpg')
    I2 = cv2.imread('../data/hp_desk.png')
    # I1 = cv2.imread('../data/cv_cover.jpg')
    # I2 = cv2.imread('../data/cv_desk.png')
    match_image = transform_image_to_plane_RANSAC(I1, I2, 50)
    cv2.imshow("Transformed Image Match using RANSAC", match_image)
    cv2.imwrite("../results/2_5_Match_using_RANSAC.jpg",
                match_image, )
    cv2.waitKey()