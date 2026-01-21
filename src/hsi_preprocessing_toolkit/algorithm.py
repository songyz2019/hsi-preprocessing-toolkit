from scipy.ndimage import rotate
import numpy as np
from copy import deepcopy
from .common import LOGGER
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import einops

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

def _min_max_normalize(x :np.ndarray):
    r = x.max()-x.min()
    if r<=np.finfo(x.dtype).eps:
        return np.zeros_like(x, dtype=x.dtype)
    return (x-x.min())/r

def compose_spectral_profile(x: np.ndarray) -> np.ndarray:
    '''
    x: a [H C] or [W C] array, any dtype
    return: HW4 RGBA Uint8 Image (0~225), x (as row direction) is height or width, y (as col direction) is spectral
    '''
    x = x.transpose(1,0) # convert to [C X]
    x = _min_max_normalize(x)
    mapper = matplotlib.colormaps.get_cmap('viridis')
    rgb = mapper(x)
    rgb = (rgb*255).astype('uint8')
    return rgb




def compose_bread_edge(hsi: np.ndarray) -> (np.ndarray, np.ndarray): # input HWC, output HWC
    top = _min_max_normalize(hsi[:,0,:])   # H C
    right = _min_max_normalize(hsi[0,:,:]) # W C
    
    # top_fig = plt.figure()
    # top_im = plt.imshow(top, cmap='viridis')
    # plt.axis('off')
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # # top_tr = mtransforms.Affine2D()
    # # top_tr.rotate_deg(45).scale(1, 0.5)
    # # top_im.set_transform(top_tr)
    # # top_im.set_clip_on(False)

    # right_fig = plt.figure()
    # plt.imshow(right, cmap='viridis')
    # plt.axis('off')
    # plt.margins(0, 0)

    return compose_spectral_profile(top), compose_spectral_profile(right)
    