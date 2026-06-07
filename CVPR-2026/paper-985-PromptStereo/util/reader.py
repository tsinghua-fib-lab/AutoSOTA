import re
import cv2
import json
import imageio
import numpy as np

def pfm_reader(filename):
    file = open(filename, 'rb')
    color, width, height, scale, endian = None, None, None, None, None
    header = file.readline().rstrip()

    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())

    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())

    if scale < 0:
        endian = '<'
        scale = -scale
    else:
        endian = '>'

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = np.flipud(data)

    return np.ascontiguousarray(data)

def sceneflow_disp_reader(filename, mask):
    disp = pfm_reader(filename)
    valid = disp > 0

    return disp, valid.astype(np.float32)

def kitti_disp_reader(filename, mask):
    disp = None
    if mask == 'all':
        disp = imageio.imread(filename).astype(np.float32) / 256
    elif mask == 'noc':
        disp = imageio.imread(filename.replace('disp_occ', 'disp_noc')).astype(np.float32) / 256
    else:
        raise Exception(f'Invalid mask name: {mask}.')

    valid = disp > 0

    return disp, valid.astype(np.float32)

def middlebury_disp_reader(filename, mask):
    disp = pfm_reader(filename)

    if 'disp0GT.pfm' in filename:
        nocc = imageio.imread(filename.replace('disp0GT.pfm', 'mask0nocc.png'))
    elif 'disp0.pfm' in filename:
        nocc = imageio.imread(filename.replace('disp0.pfm', 'mask0nocc.png'))
        
    valid = None

    if mask == 'all':
        valid = nocc > 0
    elif mask == 'noc':
        valid = nocc == 255
    else:
        raise Exception(f'Invalid mask name: {mask}.')

    return disp, valid.astype(np.float32)

def eth3d_disp_reader(filename, mask):
    disp = pfm_reader(filename)
    nocc = imageio.imread(filename.replace('disp0GT.pfm', 'mask0nocc.png'))
    valid = None

    if mask == 'all':
        valid = nocc > 0
    elif mask == 'noc':
        valid = nocc == 255
    else:
        raise Exception(f'Invalid mask name: {mask}.')

    return disp, valid.astype(np.float32)

def drivingstereo_disp_reader(filename, mask):
    disp = imageio.imread(filename).astype(np.float32)

    if  'full' in filename:
        disp = disp / 128
    elif 'half' in filename:
        disp = disp / 256

    valid = disp > 0

    return disp, valid.astype(np.float32)

def booster_disp_reader(filename, mask):
    disp = np.load(filename).astype(np.float32)
    valid = imageio.imread(filename.replace('disp_00.npy', 'mask_00.png'))
    valid = valid > 0

    return disp, valid.astype(np.float32)

def foundationstereo_disp_reader(filename, mask):
    disp = imageio.imread(filename).astype(np.float32)
    disp = (disp[..., 0] * 255 * 255 + disp[..., 1] * 255 + disp[..., 2]) / 1000
    valid = disp > 0

    return disp, valid.astype(np.float32)

def tartanair_disp_reader(filename, mask):
    depth = np.load(filename).astype(np.float32)
    disp = 80 / depth
    valid = disp > 0

    return disp, valid.astype(np.float32)

def crestereo_disp_reader(filename, mask):
    disp = imageio.imread(filename).astype(np.float32)
    disp = disp / 32
    valid = disp > 0

    return disp, valid.astype(np.float32)

def fallingthings_disp_reader(filename, mask):
    a = imageio.imread(filename).astype(np.float32)

    with open('/'.join(filename.split('/')[:-1] + ['_camera_settings.json']), 'r') as f:
        intrinsics = json.load(f)
        fx = intrinsics['camera_settings'][0]['intrinsic_settings']['fx']

    disp = (fx * 6 * 100) / a
    valid = disp > 0

    return disp, valid.astype(np.float32)