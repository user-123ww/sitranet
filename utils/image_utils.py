import numpy as np
from PIL import Image
def pad_and_center_crop_pil(img_pil: Image.Image, target: int = 256):
    arr = np.array(img_pil)
    if arr.ndim == 2:
        arr = arr[..., None]
    h, w = arr.shape[:2]
    pad_h = max(0, target - h)
    pad_w = max(0, target - w)
    if pad_h > 0 or pad_w > 0:
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        arr = np.pad(arr, ((top, bottom), (left, right), (0, 0)), mode='reflect')
        h, w = arr.shape[:2]
    if h > target or w > target:
        start_h = (h - target) // 2
        start_w = (w - target) // 2
        arr = arr[start_h:start_h + target, start_w:start_w + target, ...]
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return Image.fromarray(arr)
def integer_roll_pil(img_pil: Image.Image, max_shift: int = 1):
    arr = np.array(img_pil)
    dx = np.random.randint(-max_shift, max_shift + 1)
    dy = np.random.randint(-max_shift, max_shift + 1)
    if dx == 0 and dy == 0:
        return img_pil
    rolled = np.roll(np.roll(arr, shift=dy, axis=0), shift=dx, axis=1)
    return Image.fromarray(rolled)
