import cv2
import numpy as np
from warpH import warpH

def compositeH(H1to2, template, img): 
    """
    Projects and combine template image (template) to another image (img)
    ## Parameters
        H1to2:
            'Homography matrix used for transformation'
        template:
            'The image which will be tranformed onto img'
        img:
            'The image that will be transformed on'

    ## Returns:
        'Composite image after combining template and img'
    """

    # Create mask of same size as template
    height, width  = template.shape[:2]
    mask = np.ones((height, width, 3), dtype=np.uint8)*255

    # Warp mask by appropriate homography
    warped_mask = warpH(mask, H1to2, img.shape)
    warped_mask_inv = cv2.bitwise_not(warped_mask)

    # Warp template by appropriate homography
    warped_template = warpH(template, H1to2, img.shape)

    # Use mask to combine the warped template and the image
    template_masked = cv2.bitwise_and(warped_template, warped_mask)
    img_masked = cv2.bitwise_and(img, warped_mask_inv)
    composite_image = cv2.add(img_masked,template_masked)

    return composite_image