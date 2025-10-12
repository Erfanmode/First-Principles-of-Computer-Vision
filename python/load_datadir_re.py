import numpy as np
from PIL import Image
import matplotlib.pyplot as  plt
import os

def load_datadir_re(datadir, resize, gamma:float, load_imgs=True,
                     load_mask=True, white_balance=[1,1,1]):
    """
    Load photometric stereo data from the given directory.

    ## Parameters:
    -   datadir:
            relative data directory of dataset
    -   gamma:
            mapping coefficient
    -   load_imgs:
            True, if images should be loaded
    -   load_mask:
            True, if mask should be loaded
    -   white_balance:
            White balance weights

    ## Returns 
    -   data:
            Data class
    """
    height, width = resize
    # If white_balance is a vector, convert it to a diagonal matrix
    if isinstance(white_balance, (list, tuple, np.ndarray)) and len(white_balance) == 3:
        white_balance = np.diag(white_balance)
    
    # Build data structure
    data = type('data', (object,), {})()  # Create an empty class instance
    data.L_direction = np.loadtxt(os.path.join(datadir, 'light_directions.txt'))
    data.L = np.loadtxt(os.path.join(datadir, 'light_intensities.txt'))
    
    # Ensure data.L is a 2D array so we can perform matrix multiplication correctly
    if data.L.ndim == 1:
        data.L = data.L.reshape(-1, 3)
    
    # Perform matrix multiplication to apply white balance
    data.L = np.dot(data.L, white_balance)
    
    # Read filename list
    with open(os.path.join(datadir, 'filenames.txt'), 'r') as f:
        data.filenames = [os.path.join(datadir, line.strip()) for line in f.readlines()]
    
    # Load mask image if needed
    if load_mask and not hasattr(data, 'mask'):
        data.mask = Image.open(os.path.join(datadir, 'mask.png'))
        data.mask = data.mask.resize((width,height), Image.NEAREST)
        data.mask = np.array(data.mask)
        data.mask = data.mask[:,:,None]
        data.mask = data.mask.reshape((-1, 1))
        data.foreground_ind = np.where(data.mask != 0)[0]
        data.background_ind = np.where(data.mask == 0)[0]
        data.mask = data.mask.reshape(*resize)
    # Load images if needed
    if load_imgs and not hasattr(data, 'imgs'):
        data.imgs = None
        for i, filename in enumerate(data.filenames):
            img = Image.open(filename)
            img = img.resize((width,height), Image.NEAREST)
            img = np.array(img)
        
            # Apply gamma correction
            img = (img / 255.0) ** gamma
            # Normalize images by dividing the channels by the light intensity on that channel
            img = np.dot(img, np.diag(1 / data.L[i]))
            # Transform image ginto gray scale
            img = img.mean(axis=2)
           
            if data.imgs is None:
                img = img.reshape((-1,1))
                data.imgs = img.reshape((-1, 1))
            else:
                data.imgs = np.append(data.imgs, img.reshape((-1, 1)), axis=1)

    
    return data





