import numpy as np
import cv2,os


# it will extract the image list 
def _get_img_lists(img_path):
    frame_list = []
    for frame in os.listdir(img_path):
        if os.path.splitext(frame)[1] == '.jpg':
            frame_list.append(os.path.join(img_path, frame)) 
    return frame_list

def _linear_mapping(img):
        return (img - img.min()) / (img.max() - img.min())

def _window_func_2d(width,height):
    win_col = np.hanning(width)
    win_row = np.hanning(height)
    mask_col, mask_row = np.meshgrid(win_col, win_row)
    return mask_col * mask_row

def _get_gauss_response(size,sigma):
    w,h=size

    # get the mesh grid
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))

    # get the center of the object
    center_x, center_y = w / 2, h / 2

    # cal the distance...
    dist = ((xs - center_x) ** 2 + (ys - center_y) ** 2) / (2*sigma**2)

    # get the response map
    response = np.exp(-dist)

    # normalize
    response = _linear_mapping(response)
    return response

# pre-processing the image...
def pre_process(img):
    # get the size of the img...
    height, width = img.shape
    img = np.log(img + 1)
    img = (img - np.mean(img)) / (np.std(img) + 1e-5)
    # use the hanning window...
    window = window_func_2d(height, width)
    img = img * window

    return img

def window_func_2d(height, width):
    win_col = np.hanning(width)
    win_row = np.hanning(height)
    mask_col, mask_row = np.meshgrid(win_col, win_row)

    win = mask_col * mask_row

    return win

def random_warp(img):
    a = -180 / 16
    b = 180 / 16
    r = a + (b - a) * np.random.uniform()
    # rotate the image...
    matrix_rot = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), r, 1)
    img_rot = cv2.warpAffine(np.uint8(img * 255), matrix_rot, (img.shape[1], img.shape[0]))
    img_rot = img_rot.astype(np.float32) / 255
    return img_rot


