import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, binary_opening, disk

def is_mask_valid(new_mask, best_mask,
                  area_thresh = 20000,
                  top_n = 3,
                  overlap_thresh = 5):

    new_lbl = label(new_mask)
    old_lbl = label(best_mask)

    # sort areas in descending
    props = sorted(regionprops(new_lbl),
                   key=lambda p: p.area,
                   reverse=True)

    # area check
    max_area = props[0].area
    if max_area > area_thresh:
        return False

    # merge check top_n components
    for prop in props[:top_n]:
        comp_mask = (new_lbl == prop.label)
        overlaps = np.unique(old_lbl[comp_mask])
        overlaps = overlaps[overlaps != 0]
        if len(overlaps) > overlap_thresh:
            return False

    return True

def is_candidate_mask(
    m, left, top, H, W,
    min_pixel, 
    neg_pts,
    border_offset,
    max_area=20000,
):
    # filter out too small or too large masks
    mask_area = m.sum()
    if mask_area < min_pixel or mask_area > max_area:
        return False
    
    # filter out masks containing >3 negative points
    if len(neg_pts) > 0:
        neg_ys, neg_xs = neg_pts[:, 0], neg_pts[:, 1]
        neg_in_mask = m[neg_ys, neg_xs].sum()
    
        if neg_in_mask > 3:
            return False

    h_p, w_p = m.shape
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return False
   
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    
    # filter out masks touch invalid boundaries
    touch_left   = (min_x <= border_offset)
    touch_top    = (min_y <= border_offset)
    touch_right  = (max_x >= w_p - 1 - border_offset)
    touch_bottom = (max_y >= h_p - 1 - border_offset)

    if touch_left and left > 0:
        return False 
    if touch_top and top > 0:
        return False
    if touch_right and (left + w_p) < W:
        return False
    if touch_bottom and (top + h_p) < H:
        return False

    return True

def keep_and_smooth(mask, smooth_radius=2):
    mask = mask.astype(bool)

    lbls = label(mask, connectivity=1)
    num = lbls.max()
    if num <= 1:
        return mask 
    # count pixels in each label
    counts = np.bincount(lbls.ravel())
    counts[0] = 0 
    core = (lbls == counts.argmax())

    selem = disk(smooth_radius)
    core = binary_closing(core, selem)
    core = binary_opening(core, selem)
    core = binary_fill_holes(core)
    return core 