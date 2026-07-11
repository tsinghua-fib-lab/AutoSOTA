import cv2
import numpy as np
from skimage import filters, morphology, measure
from skimage.color import rgb2hed
from skimage.feature import peak_local_max
from scipy.ndimage import distance_transform_edt

def remove_lines(mask, r=3):
    se = morphology.disk(r)
    er = morphology.binary_erosion(mask, se)
    opened = morphology.reconstruction(er.astype(np.uint8), mask.astype(np.uint8),
                                       method='dilation').astype(bool)
    return opened

def find_white_mask(rgb, white_v=0.85, white_s=0.25, min_white_area=10000):
    """
    Near-white = high value (brightness) + low saturation.
    Keep sufficiently large connected components as 'white background'.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    S = hsv[..., 1].astype(np.float32) / 255.0
    V = hsv[..., 2].astype(np.float32) / 255.0
    white = (V > white_v) & (S < white_s)
    white = morphology.remove_small_objects(white, 40)

    # Keep only big white regions
    lbl = morphology.label(white, connectivity=2)
    big_white = np.zeros_like(white, dtype=bool)
    for lab in range(1, lbl.max() + 1):
        area = np.count_nonzero(lbl == lab)
        if area >= min_white_area:
            big_white[lbl == lab] = True

    return big_white

def check_mask(mask):
    labels = morphology.label(mask, connectivity=2)
    props = measure.regionprops(labels)

    for component_props in props:
        component_image = component_props.image
        distance = distance_transform_edt(component_image)
        coordinates = peak_local_max(distance, min_distance=5, labels=component_image)
        num_peaks = coordinates.shape[0]
        
        if num_peaks > 5:
            return False
        
    return True

def otsu_top(image_rgb, keep_ratio=0.7, white_frac=0.55):
    image_rgb = np.array(image_rgb)
    image = cv2.GaussianBlur(image_rgb, (7, 7), 0)

    hed = rgb2hed(image)
    h_channel, e_channel = hed[:, :, 0], hed[:, :, 1]
    white_mask = find_white_mask(image_rgb)
    frac_white = white_mask.mean()
    high_conf = False

    # High white content, iterative multi-Otsu.
    if frac_white > white_frac:
        mask_fg = None

        # 2 to 4 classes multi-Otsu.
        for n_classes in range(2, 5):
            try:
                thresholds = filters.threshold_multiotsu(h_channel, classes=n_classes)
                current_mask = (h_channel > thresholds[-1])
            except ValueError:
                break

            mask_fg = current_mask
            if check_mask(mask_fg):
                break

        mask_bg = white_mask | (h_channel < thresholds[0])

    # Medium white content, Otsu once.
    else:
        work_region = ~white_mask
        h_vals = h_channel[work_region]

        if h_vals.size > 0:
            otsu_thresh = filters.threshold_otsu(h_vals)
            mask_fg = np.zeros_like(h_channel, dtype=bool)
            mask_fg[work_region] = h_channel[work_region] > otsu_thresh
        else:
            mask_fg = np.zeros_like(h_channel, dtype=bool)

        fg_ratio = float(mask_fg.mean())

        # small foreground area
        if fg_ratio < 0.1:
            mask_bg = ~mask_fg

        # normal foreground area
        else:
            high_conf = True
            t1 = filters.threshold_otsu(h_channel)
            mask_h = (h_channel > t1)

            # Keep top 70% of foreground
            h_fg_pixels = h_channel[mask_h]
            if h_fg_pixels.size > 0:
                cutoff_fg = max(1, int(np.floor(keep_ratio * h_fg_pixels.size)))
                # Using partition is faster than sorting the whole array
                thr_fg = np.partition(h_fg_pixels, -cutoff_fg)[-cutoff_fg]
                mask_fg = np.zeros_like(mask_h, dtype=bool)
                mask_fg[mask_h] = h_channel[mask_h] >= thr_fg
            else:
                mask_fg = np.zeros_like(mask_h, dtype=bool)

            # Keep top 70% of background pixels
            bg_channel = 1.0 - h_channel
            in_bg_zone = ~mask_h
            bg_pixels = bg_channel[in_bg_zone]
            if bg_pixels.size > 0:
                cutoff_bg = max(1, int(np.floor(keep_ratio * bg_pixels.size)))
                thr_bg = np.partition(bg_pixels, -cutoff_bg)[-cutoff_bg]
                mask_bg = np.zeros_like(mask_h, dtype=bool)
                mask_bg[in_bg_zone] = bg_channel[in_bg_zone] >= thr_bg
            else:
                mask_bg = np.zeros_like(mask_h, dtype=bool)
    
    return mask_fg, mask_bg, high_conf

def generate_ref_mask(img, keep_ratio=0.7):
    mask_fg_ref, mask_bg, high_conf = otsu_top(img, keep_ratio=keep_ratio)
    mask_fg_ref = remove_lines(mask_fg_ref)
    mask_bg_inv = remove_lines(~mask_bg)
    mask_bg_ref = ~mask_bg_inv

    mask_fg_ref = morphology.remove_small_objects(mask_fg_ref, min_size=40)
    mask_fg_ref = morphology.remove_small_holes(mask_fg_ref, area_threshold=40)

    mask_bg_ref = morphology.remove_small_objects(mask_bg_ref, min_size=40)
    mask_bg_ref = morphology.remove_small_holes(mask_bg_ref, area_threshold=40)

    return mask_fg_ref, mask_bg_ref, high_conf