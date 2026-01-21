import einops
import cv2
import numpy as np

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
    h, w = front.shape[:2]
    d = top.shape[1]
    assert d==right.shape[1]
    assert h==right.shape[0]
    assert w==  top.shape[0]

    a = int(d * 0.5 * 0.7071)
    W, H = w + a, h + a
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    canvas[a:H, 0:w] = front
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

    m_r = np.any(canvas_r > 0, axis=-1)
    m_t = np.any(canvas_t > 0, axis=-1)
    canvas[m_r] = canvas_r[m_r]
    canvas[m_t] = canvas_t[m_t]
    return canvas

if __name__ == "__main__":
    h, w, d = 166, 333, 63
    
    # 构造测试图像 (BGR)
    f = np.full((h, w, 3), (100, 100, 255), dtype=np.uint8) # 浅红
    t = np.full((w, d, 3), (100, 255, 100), dtype=np.uint8) # 浅绿
    r = np.full((h, d, 3), (255, 100, 100), dtype=np.uint8) # 浅蓝
    
    # 加点网格线方便观察对齐
    f[::20, :] = 0; f[:, ::20] = 0
    t[::20, :] = 0; t[:, ::20] = 0
    r[::20, :] = 0; r[:, ::20] = 0

    result = generate_oblique_cube(f, t, r)
    
    cv2.imshow("Cube Test", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()