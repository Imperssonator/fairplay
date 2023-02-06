from PIL import Image
import numpy as np


def rect_mask_grayscale(
    img,
    corners,
    mask_color=255,
    invert=False
):
    """Given the corners of a bounding box, corners,
    overlay pixels of the specified mask_color either
    within the bounding box (invert=False) or outside
    the bounding box (invert=True)
    
    Parameters
    ----------
    img: np.array (3d)
    corners: np.array
        [x1, y1, x2, y2], where x is pixels from the left, y is pixels from the top
        in other words +x points East, +y points South
    mask_color: uint8
        integer from 0 to 255, 0 for black, 255 for white. Default 255.
    invert: bool
        default False; if False, mask inside the corners,
        if True, apply mask outside the corners
        
    Returns
    -------
    img_masked: np.array (3d)
    """
    img_masked = img.copy()
    m, n, p = img_masked.shape
    
    mask = np.zeros([m, n]).astype('bool')
    mask[corners[1]:corners[3], corners[0]:corners[2]] = 1
    if invert:
        mask = ~mask
        
    for cc in range(p):
        img_ch = img_masked[:, :, cc]
        img_ch[mask] = mask_color
    
    return img_masked