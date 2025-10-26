import cv2
import numpy as np
def matchPics(I1, I2):
    """_summary_

    Args:
        I1: Image 1 (projected to)
        I2: Image 2 (projected from)

    Returns:
        kp1: Keypoints from image1
        kp2: Keypoints from image2  
        matches: List of DMatch objects
    """
    #MATCHPICS Extract features, obtain their descriptors, and match them!

    # Convert images to grayscale
    I1_G = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    I2_G = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

    # Initiate FAST object with default values
    fast = cv2.xfeatures2d.StarDetector_create()
    # Initiate BRIEF extractor
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    
    # Detect features in both images
    kp1 = fast.detect(I1_G,None)
    kp2 = fast.detect(I2_G,None)
    # Compute the descriptors with BRIEF
    kp1, des1 = brief.compute(I1_G, kp1)
    kp2, des2 = brief.compute(I2_G, kp2)

    # Compute matches for two images by using the Brute Force Matcher object
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors
    matches = bf_matcher.match(des1,des2)
    # Also sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    ## Match features using the descriptors
    return kp1, kp2, matches
