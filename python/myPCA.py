import numpy as np
from sklearn.decomposition import PCA

def myPCA(data, resize:tuple, rank:int = 3):
    """
    This function implements the PCA based Photometric Stereo

    ## Parameters:
    - data:
        Data instance which includes pixel values, lights direction,
        and lights intensities.

    - resize:
        Resizing shape of output normal mapping.
        
    - rank:
        The rank that information is factored by.
        If the rank is equal or higher than 3, most information regarding surface
        of the object is saved.
        
    ## Returns:
    - Normal mapping for each pixel.
    """
    num_pixels, num_lights = data.imgs.shape
    height, width  = resize

    if (num_pixels != width * height): raise ValueError("data inputs and 'resize' value " \
    "are incompatible")

    # mask images
    mask = np.reshape(data.mask,(-1,1)) / 255.0
    masked_imgs = data.imgs * mask

    # Perform PCA on the image data
    # pca = PCA(n_components = rank)
    # principal_components = pca.fit_transform(masked_imgs) 
    
    # Now, estimate the lighting directions
    U, s, Vt = np.linalg.svd(masked_imgs.T, full_matrices=False)
    est_imgs = U[:,:rank] @ np.diag(s[:rank]) @ Vt[:rank,:]
    
    # Finally, we get normals using estimations
    normal = est_imgs.T @ np.linalg.pinv(data.L_direction).T
    normal = np.reshape(normal, (*resize,3))

    # Additionally, because the pseudo-inverse does not return an exact unit vector,
    # the vectors are normalized and masked out by the outline of the object  
    normal /= (np.linalg.norm(normal,axis=2,keepdims=True) + 1e-6)
    normal *= data.mask[:,:,None]/255.0

    return normal