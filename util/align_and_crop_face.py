# adapted from https://github.com/man805/Diffusion-Video-Autoencoders/blob/64e5458c4f9cec2f5d8cad4bc60b90845c2b1122/editing_CLIP.py#L49

from torchvision.utils import *
import os
import glob
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys
from losses.clip_loss import CLIPLoss
import dlib
import numpy as np
import skimage.io as io
from PIL import Image
import scipy
from scipy.ndimage import gaussian_filter1d
import PIL
import argparse
import imageio


def get_landmark(filepath, predictor, detector=None, fa=None):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    if fa is not None:
        image = io.imread(filepath)
        lms, _, bboxes = fa.get_landmarks(image, return_bboxes=True)
        if len(lms) == 0:
            return None
        return lms[0]

    if detector is None:
        detector = dlib.get_frontal_face_detector()
    if isinstance(filepath, PIL.Image.Image):
        img = np.array(filepath)
    else:
        img = dlib.load_rgb_image(filepath)
    dets = detector(img)

    for k, d in enumerate(dets):
        shape = predictor(img, d)
        break
    else:
        return None
    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    return lm

def compute_transform(filepath, predictor, detector=None, scale=1.0, fa=None):
    lm = get_landmark(filepath, predictor, detector, fa)
    if lm is None:
        raise Exception(f'Did not detect any faces in image: {filepath}')
    lm_chin = lm[0: 17]  # left-right
    lm_eyebrow_left = lm[17: 22]  # left-right
    lm_eyebrow_right = lm[22: 27]  # left-right
    lm_nose = lm[27: 31]  # top-down
    lm_nostrils = lm[31: 36]  # top-down
    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise
    lm_mouth_inner = lm[60: 68]  # left-clockwise
    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg
    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)

    x *= scale
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    return c, x, y

def crop_image(filepath, output_size, quad, enable_padding=False):
    x = (quad[3] - quad[1]) / 2
    qsize = np.hypot(*x) * 2
    # read image
    if isinstance(filepath, PIL.Image.Image):
        img = filepath
    else:
        img = PIL.Image.open(filepath)
    transform_size = output_size
    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink
    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if (crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]):
        img = img.crop(crop)
        quad -= crop[0:2]
    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]
    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)
    return img

def calc_alignment_coefficients(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    a = np.matrix(matrix, dtype=float)
    b = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(a.T * a) * a.T, b)
    return np.array(res).reshape(8)

predictor = dlib.shape_predictor("pretrained_models/shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

# get crop params with landmark predictor and face detector
images = []
video_path = "./Rerender_A_Video/tmp_video_frames"
dir_name = video_path
for fname in sorted(os.listdir(dir_name)):
    path = os.path.join(dir_name, fname)
    fname = fname.split('.')[0]
    images.append((fname, path))

cs, xs, ys = [], [], []
for _, path in images:
    c, x, y = compute_transform(path, predictor, detector=detector, scale=1.0)
    cs.append(c)
    xs.append(x)
    ys.append(y)
cs = np.stack(cs)
xs = np.stack(xs)
ys = np.stack(ys)
cs = gaussian_filter1d(cs, sigma=1.0, axis=0)
xs = gaussian_filter1d(xs, sigma=3.0, axis=0)
ys = gaussian_filter1d(ys, sigma=3.0, axis=0)
quads = np.stack([cs - xs - ys, cs - xs + ys, cs + xs + ys, cs + xs - ys], axis=1)
quads = list(quads)
orig_images = []

# do crop
for quad, (_, path) in tqdm(zip(quads, images), total=len(quads)):
    crop = crop_image(path, 1024, quad.copy())
    crop = crop.convert('RGB')
    crop.save(f"data/kwai_videos/cropped/{path.split('/')[-1].split('.')[0]}.jpg")#, quality=100, subsampling=0)
    orig_image = Image.open(path)
    orig_images.append(orig_image)

# get inverse transform
image_size = 256
inverse_transforms = [
    calc_alignment_coefficients(quad + 0.5, [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]])
    for quad in quads]