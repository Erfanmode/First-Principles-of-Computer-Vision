import numpy as np

def myRobustPMS(data, resize:tuple, discard_ratio:float): 
    """
    This function implements the Robust Photometric Stereo algorithm.

    ## Parameters:
    - data:
        Data instance which includes pixel values, lights direction,
        and lights intensities.

    - resize:
        Resizing shape of output normal mapping.

    - discard_ratio: 
        Amount of extreme light conditioning for each pixel
        which would be omitted as highlights or shadows. Half of this value 
        would be discarded from top and another half from bottom values of each pixel.
        The value should be between 0 and 1.

    ## Returns:
    - Normal mapping for each pixel.
    """
    if discard_ratio <0 or discard_ratio>1:
        raise(ValueError(f"The ratio {discard_ratio} is not in range of [0,1]"))
    
    num_pixels, num_lights = data.imgs.shape
    height, width  = resize

    if (num_pixels != width * height): raise ValueError("data inputs and 'resize' value " \
    "are incompatible")

    # Firstly, we discard the darkest and brightest pixels among different
    # lighting condition based on the discard ratio
    sort_idx = np.argsort(data.imgs,axis=1)
    start_idx = max(int((discard_ratio/2)*num_lights), 0)
    end_idx = min(int((1 - discard_ratio/2)*num_lights), num_lights) 
    accepeted_idx = sort_idx[:,start_idx:end_idx]

    # Then, we mask only the pixels with unacceptable values
    masked_imgs = np.take_along_axis(data.imgs, accepeted_idx, axis=1)
    masked_imgs = np.expand_dims(masked_imgs, axis=1)

    # Also, mask the corresponding light direction of images
    expanded_L_direction = np.expand_dims(data.L_direction, axis=0)
    expanded_L_direction = np.repeat(expanded_L_direction, repeats=num_pixels, axis=0)
    accepeted_idx = np.expand_dims(accepeted_idx, axis=-1)
    masked_L_direction = np.take_along_axis(expanded_L_direction, accepeted_idx, axis=1)  

    # TO find the normal map, we use pseudo inverse of Light Matrix
    pinv_t = np.transpose(np.linalg.pinv(masked_L_direction), axes=(0,2,1))
    normal_map = masked_imgs @ pinv_t
    normal_map = np.squeeze(normal_map)
    normal = np.reshape(normal_map, (*resize,3))

    # Additionally, because the pseudo-inverse does not return an exact unit vector,
    # the vectors are normalized and masked out by the outline of the object  
    normal /= (np.linalg.norm(normal,axis=2,keepdims=True) + 1e-6)
    normal *= data.mask[:,:,None]/255.0
    
    return normal
