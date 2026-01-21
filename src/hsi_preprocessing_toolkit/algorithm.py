from scipy.ndimage import rotate
import numpy as np
from copy import deepcopy
from .common import LOGGER
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import einops
import cv2

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
    return: HC4 RGB Uint8 Image (0~225), x (as row direction) is height or width, y (as col direction) is spectral
    '''
    x = _min_max_normalize(x)
    mapper = matplotlib.colormaps.get_cmap('viridis')
    rgb = mapper(x)[:,:,:3]
    rgb = (rgb*255).astype('uint8')
    return rgb

def generate_oblique_cube(front, top, right):
    """
    符号说明:
    w, h, d : 原长方体的宽、高、深
    W, H    : 合成画布(Canvas)的宽、高
    a     : 投影偏移量 (d * 0.5 * sin(45°))
    """
    # top:   w d 3
    # right: h d 3
    # front: h w 3
    top = top[:,::-1,:] # dirty fix
    h, w = front.shape[:2]
    d = top.shape[1]
    assert d==right.shape[1]
    assert h==right.shape[0]
    assert w==  top.shape[0]

    a = int(d * 0.5 * 0.7071)
    W, H = w + a, h + a
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # 加减1是为了防止出现缝隙，这也算是一种off by one了...
    canvas_r = cv2.warpAffine(right, 
        cv2.getAffineTransform(
            np.array([[0, 0], [0, h], [d, 0]], dtype=np.float32),
            np.array([[w, a], [w, H], [W, 0]], dtype=np.float32)),
        (W, H), flags=cv2.INTER_LANCZOS4)
    canvas_t = cv2.warpAffine(top, 
        cv2.getAffineTransform(
            np.array([[0, 0], [0, w], [d, 0]], dtype=np.float32),
            np.array([[a, 0], [W, 0], [0, a]], dtype=np.float32)),
        (W, H), flags=cv2.INTER_LANCZOS4)
    canvas[a:H, 0:w] = front

    m_r = np.any(canvas_r > 0, axis=-1)
    m_t = np.any(canvas_t > 0, axis=-1)
    canvas[m_r] = canvas_r[m_r]
    canvas[m_t] = canvas_t[m_t]
    return canvas

def compose_hsi_cube(hsi: np.ndarray, front: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray): 
    '''
    input HWC,HW3(u8)
    output HWC,HWC,HWC
    '''
    top   = _min_max_normalize(hsi[0,:,:])   # W C, 有宽度W和深度，是侧面
    right = _min_max_normalize(hsi[:,-1,:])   # H C
    top   = compose_spectral_profile(top)    # W C 3
    right = compose_spectral_profile(right)  # H C 3 

    cube = generate_oblique_cube(front, top, right)

    return cube, einops.rearrange(top, 'H C c -> C H c'), einops.rearrange(right, 'H C c -> C H c')
    
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