import numpy as np

def myPMS(data, resize:tuple):
    """
    This function implements the Robust Photometric Stereo algorithm.

    ## Parameters:
    - data:
        Data instance which includes pixel values, lights direction,
        and lights intensities.

    - resize:
        Resizing shape of output normal mapping.
        
    ## Returns:
    - Normal mapping for each pixel.
    """
    num_pixels, num_lights = data.imgs.shape
    height, width  = resize

    if (num_pixels != width * height): raise ValueError("data inputs and 'resize' value " \
    "are incompatible")

    # TO find the normal map, we use pseudo inverse of Light Matrix
    normal_map = data.imgs @ np.linalg.pinv(data.L_direction).T
    normal = np.reshape(normal_map, (*resize,3))

    # Additionally, because the pseudo-inverse does not return an exact unit vector,
    # the vectors are normalized and masked out by the outline of the object  
    normal /= (np.linalg.norm(normal,axis=2,keepdims=True) + 1e-6)
    normal *= data.mask[:,:,None]/255.0
    
    return normal
