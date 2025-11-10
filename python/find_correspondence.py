import cv2
import numpy as np




def find_correspondence(img1, img2, N):
    """Finds correspondence and matching points using SIFT
    ### args:
        img1: image 1
        img2: image 2
        N: number of top matches to keep
    ### returns:
        (x1, x2): Correponding points in each image
        (des1_corr, des2_corr): Corresponding descriptors of images matched points
        (kp1, kp2, matches): Keypoints 1, Keypoints2, matches
         """
    
    # Initialize ORB detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Initialize Brute-Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) # crossCheck=True for better matches

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:N]
    x1, x2, des1_corr, des2_corr = get_matched_coordinates(kp1, kp2, des1, des2, matches) # matching points

    return (x1, x2), (des1_corr, des2_corr), (kp1, kp2, matches)




    


def get_matched_coordinates(kp1, kp2, des1, des2, matches):
    """
    Extract corresponding coordinates from matched keypoints
    
    Args:
        kp1: Keypoints from image1
        kp2: Keypoints from image2
        des1: Descriptor from corresponding keypoints of image 1  
        des2: Descriptor from corresponding keypoints of image 2 
        matches: List of DMatch objects
    
    Returns:
        pts1: Coordinates from image1 (N, 2)
        pts2: Coordinates from image2 (N, 2)
        Descriptor1: Descriptor of image 1 (N, descriptor size)
        Descriptor2: Descriptor of image 2 (N, descriptor size)
    """
    points1 = []
    points2 = []
    Descriptor1 = []
    Descriptor2 = []
    
    for match in matches:
        # Get the keypoint coordinates
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        
        # (x, y) coordinates from first image
        x1, y1 = kp1[img1_idx].pt
        points1.append([x1, y1])
        Descriptor1.append(des1[img1_idx])
        
        # (x, y) coordinates from second image
        x2, y2 = kp2[img2_idx].pt
        points2.append([x2, y2])
        Descriptor2.append(des2[img2_idx])
    
    return np.array(points1), np.array(points2),\
        np.array(Descriptor1), np.array(Descriptor2)





if __name__ == "__main__":
    # Load images
    img1 = cv2.imread('../data/templeRing/04.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('../data/templeRing/05.png', cv2.IMREAD_GRAYSCALE)
    # img1 = cv2.imread('../data/trex/images/DJI_20200223_163548_810.jpg', cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread('../data/trex/images/DJI_20200223_163551_210.jpg', cv2.IMREAD_GRAYSCALE)
    # img1 = cv2.imread('../data/fern/images/IMG_4029.JPG', cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread('../data/fern/images/IMG_4030.JPG', cv2.IMREAD_GRAYSCALE)
    img1 = cv2.resize(img1,(640,480))
    img2 = cv2.resize(img2,(640,480))
    # Draw top matches
    _, _, (kp1, kp2, matches) = find_correspondence(img1, img2, 50)
    output_image = cv2.drawMatches(img1, kp1, img2, kp2, matches,
                                    None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
 
    cv2.imshow("Matches", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()