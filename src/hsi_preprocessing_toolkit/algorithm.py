from scipy.ndimage import rotate
import numpy as np
from copy import deepcopy
from .common import LOGGER

# Return A HWC Image, the C should work with both RGB and HSI
def composite_img(imgs :list[np.ndarray], transforms:list[dict]):
    imgs = deepcopy(imgs)
    # Transform
    LOGGER.info(f"{transforms=}")
    for i, (img,trans) in enumerate(zip(imgs, transforms)):
        rotate_deg = trans['rotation']
        if rotate_deg % 360 != 0:
            img = rotate(img, angle=rotate_deg, axes=(0, 1), reshape=True)

        # Crop
        crop_top, crop_left, crop_bottom, crop_right = trans['crop']
        if crop_top > 0 or crop_left > 0 or crop_bottom > 0 or crop_right > 0:
            crop_bottom = None if crop_bottom == 0 else -crop_bottom
            crop_right  = None if crop_right == 0  else -crop_right
            img = img[crop_top:crop_bottom, crop_left:crop_right, :]
        
        imgs[i] = img
        
    # Caclate canvas size
    shapes = [x.shape for x in imgs]
    offsets = [ x['location'] for x in transforms ]
    canvas_c = imgs[0].shape[-1]
    sizes = [ ((h+abs(x)),w+abs(y)) for (h,w,_),(x,y) in zip(shapes, offsets)]
    canvas_h, canvas_w = ( max([x[0] for x in sizes]), max([x[1] for x in sizes]) )
    canvas_shape = (canvas_h, canvas_w, canvas_c)
    LOGGER.info(f"{sizes=} {canvas_shape=} {shapes=}")
    canvas = np.zeros(shape=canvas_shape, dtype=imgs[0].dtype)

    # 合成 Compositing
    for img,trans in zip(imgs, transforms):
        x,y = trans['location']
        h,w,_ = img.shape
        canvas[x:x+h, y:y+w] = img
    return canvas