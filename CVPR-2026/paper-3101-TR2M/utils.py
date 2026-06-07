import torch
import torch.nn as nn
from torchvision import transforms
import random
import os, sys
import numpy as np
import torch
from collections import Counter
import inflect
import matplotlib
import cv2

import matplotlib as mpl
import matplotlib.cm as cm
import PIL.Image as pil
from collections import OrderedDict

def load_model_full(model, pretrain_dir, log=True):
    state_dict_ = torch.load(pretrain_dir, map_location='cuda:0')
    print('loaded pretrained weights form %s !' % pretrain_dir)
    state_dict = OrderedDict()

    # convert data_parallal to model
    for key in state_dict_:
        if key.startswith('module') and not key.startswith('module_list'):
            state_dict[key[7:]] = state_dict_[key]
        else:
            state_dict[key] = state_dict_[key]

  # check loaded parameters and created model parameters
    model_state_dict = model.state_dict()

    for key in state_dict:
        if key in model_state_dict:
#       print(key,state_dict[key].shape,model_state_dict[key].shape)
            if state_dict[key].shape != model_state_dict[key].shape:
                if log:
                    print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(key, model_state_dict[key].shape, state_dict[key].shape))
                state_dict[key] = model_state_dict[key]
        else:
            if log:
                print('Drop parameter {}.'.format(key))
    for key in model_state_dict:
        if key not in state_dict:
            if log:
                print('No param {}.'.format(key))
            state_dict[key] = model_state_dict[key]
    model.load_state_dict(state_dict, strict=False)
    print('load model finished!')

    return model

def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s

def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)

def visualize_eval(image, gt_depth, relative_depth, rel_abs_depth, transfered_absolute_depth, scale_pred, shift_pred, min_depth, max_depth, image_h, image_w, sample_path, vis_save_path):

    im_save_path = os.path.join(vis_save_path, sample_path[:-4] + '_image.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 若 image2 是 BGR 格式
    cv2.imwrite(im_save_path, cv2.resize(image, (image_w, image_h)))

    gt_depth_save_path = os.path.join(vis_save_path, sample_path[:-4] + '_gt.png')
    gt_depth_save = render_depth(gt_depth, min=min_depth, max=max_depth)
    cv2.imwrite(gt_depth_save_path, cv2.resize(gt_depth_save, (image_w, image_h)))

    gt_remap_depth_save_path = os.path.join(vis_save_path, sample_path[:-4] + '_gt_remap.png')
    gt_remap_depth_save = render_depth(gt_depth)
    cv2.imwrite(gt_remap_depth_save_path, cv2.resize(gt_remap_depth_save, (image_w, image_h)))

    relative_depth_save_path = os.path.join(vis_save_path, sample_path[:-4] + '_relative.png')
    relative_depth_save = render_depth(relative_depth)
    cv2.imwrite(relative_depth_save_path, cv2.resize(relative_depth_save, (image_w, image_h)))

    if rel_abs_depth is not None:
        absolute_re_depth_save_path = os.path.join(vis_save_path, sample_path[:-4] + '_absolute_re.png')
        absolute_depth_save = render_depth(rel_abs_depth)
        cv2.imwrite(absolute_re_depth_save_path, cv2.resize(absolute_depth_save, (image_w, image_h)))

    absolute_depth_save_path = os.path.join(vis_save_path, sample_path[:-4] + '_absolute.png')
    absolute_depth_save = render_depth(transfered_absolute_depth, min=min_depth, max=max_depth)
    cv2.imwrite(absolute_depth_save_path, cv2.resize(absolute_depth_save, (image_w, image_h)))

    absolute_remap_depth_save_path = os.path.join(vis_save_path, sample_path[:-4] + '_absolute_remap.png')
    absolute_remap_depth_save = render_depth(transfered_absolute_depth)
    cv2.imwrite(absolute_remap_depth_save_path, cv2.resize(absolute_remap_depth_save, (image_w, image_h)))

    scale_pred_save_path = os.path.join(vis_save_path, sample_path[:-4] + '_scale_pred.png')
    scale_pred_save = render_depth(scale_pred, cv2.COLORMAP_HOT)
    cv2.imwrite(scale_pred_save_path, cv2.resize(scale_pred_save, (image_w, image_h)))

    shift_pred_save_path = os.path.join(vis_save_path, sample_path[:-4] + '_shift_pred.png')
    shift_pred_save = render_depth(shift_pred, cv2.COLORMAP_HOT)
    cv2.imwrite(shift_pred_save_path, cv2.resize(shift_pred_save, (image_w, image_h)))

    return

def depth2disp(depth, min_depth=0.1, max_depth=150):
    """Convert depth to disp
    """
    depth = torch.clamp(depth, min=min_depth, max=max_depth)
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = 1 / depth
    disp = scaled_disp - min_disp / (max_disp - min_disp)

    return disp
 # cv2.COLORMAP_INFERNO
def render_depth(depth, color_map = cv2.COLORMAP_MAGMA, min=None, max=None):
    d_min = min if min is not None else depth[depth > 0].min()
    d_max = max if max is not None else depth.max()
    depth[depth > d_max] = d_max
    depth[depth < d_min] = d_min

    depth = (depth - d_min) / (d_max  - d_min) * 255.0
    depth = depth.astype(np.uint8)
    depth_color = cv2.applyColorMap(depth, color_map)
    return depth_color


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

def get_text(data_path, sample_path, mode="train", dataset=None, combine_words_no_area = False, close_car_percent=0.01, far_car_percent=0.001):
    text_list = []
    for i in range(len(sample_path)):  # B=4
        if dataset == "void":
            # if mode == "train":
            #     print(sample_path[i].split(" ")[0], flush=True)
            #     txt_path = data_path+"/" + sample_path[i].split(" ")[0].replace("image_02", "image")[:-4]+'.txt', flush=True)
            # else:
            txt_path = data_path+"/"+sample_path[i][:-4]+'.txt'
        else:
            txt_path = data_path+"/"+sample_path[i].split(' ')[0][:-4]+'.txt'
        with open(txt_path, 'r', encoding='utf-8') as file:
            # print(txt_path, flush=True)
            # for multi captions
            if mode=="train":
                random_number = random.randint(0, 14)
            else:
                random_number = 0
            for j, line in enumerate(file):
                if j == random_number:
                    text = line
            # if dataset == "void":
            #     text_list.append(text.replace("\n", ""))
            # else:
            text_list.append(text)

    return text_list

def remove_repetitive_words(text):
    # Split the text into individual words
    words = text.split()

    # Keep track of encountered words
    encountered_words = set()

    # List to store unique words
    unique_words = []

    # Iterate through the words
    for word in words:
        # If the word is not encountered yet, add it to the unique_words list
        if word not in encountered_words:
            unique_words.append(word)
            encountered_words.add(word)

    # Reconstruct the string without repetitive words
    result = ' '.join(unique_words)
    return result



def combine_repetitive_words(text):
    p = inflect.engine()
    # Split the text into individual words
    words = text.split(", ")
    buffer=words[0][0:13]
    words[0]=words[0][14:]
    words.insert(0, buffer)
    words=words[:-1]

    # Count the occurrences of each word
    word_counts = Counter(words)

    # Iterate through the counted words
    combined_words = []
    init=True
    for word, count in word_counts.items():
        # If the count is greater than 1, combine the word with its count
        if init is True:
            combined_words.append(word)
            if word=="An image with":
                init = False
        else:
            if count > 1:
                plural_word = p.plural(word)
                combined_words.append(f"{count} {plural_word},")
            else:
                combined_words.append(word+",")

    # Join the words back into a single string
    combined_text = ' '.join(combined_words)
    return combined_text

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


def block_print():
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def colorize(value, vmin=None, vmax=None, cmap='Greys'):
    value = value.cpu().numpy()[:, :, :]
    value = np.log10(value)

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value*0.

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)

    img = value[:, :, :3]

    return img.transpose((2, 0, 1))


def normalize_result(value, vmin=None, vmax=None):
    value = value.cpu().numpy()[0, :, :]

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.

    return np.expand_dims(value, 0)


inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)


eval_metrics = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']


def compute_errors(gt, pred):
    a = (gt / pred)
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]


def flip_lr(image):
    """
    Flip image horizontally

    Parameters
    ----------
    image : torch.Tensor [B,3,H,W]
        Image to be flipped

    Returns
    -------
    image_flipped : torch.Tensor [B,3,H,W]
        Flipped image
    """
    assert image.dim() == 4, 'You need to provide a [B,C,H,W] image to flip'
    return torch.flip(image, [3])

def fuse_inv_depth(inv_depth, inv_depth_hat, method='mean'):
    """
    Fuse inverse depth and flipped inverse depth maps

    Parameters
    ----------
    inv_depth : torch.Tensor [B,1,H,W]
        Inverse depth map
    inv_depth_hat : torch.Tensor [B,1,H,W]
        Flipped inverse depth map produced from a flipped image
    method : str
        Method that will be used to fuse the inverse depth maps

    Returns
    -------
    fused_inv_depth : torch.Tensor [B,1,H,W]
        Fused inverse depth map
    """
    if method == 'mean':
        return 0.5 * (inv_depth + inv_depth_hat)
    elif method == 'max':
        return torch.max(inv_depth, inv_depth_hat)
    elif method == 'min':
        return torch.min(inv_depth, inv_depth_hat)
    else:
        raise ValueError('Unknown post-process method {}'.format(method))

def post_process_depth(depth, depth_flipped, method='mean'):
    """
    Post-process an inverse and flipped inverse depth map

    Parameters
    ----------
    inv_depth : torch.Tensor [B,1,H,W]
        Inverse depth map
    inv_depth_flipped : torch.Tensor [B,1,H,W]
        Inverse depth map produced from a flipped image
    method : str
        Method that will be used to fuse the inverse depth maps

    Returns
    -------
    inv_depth_pp : torch.Tensor [B,1,H,W]
        Post-processed inverse depth map
    """
    B, C, H, W = depth.shape
    inv_depth_hat = flip_lr(depth_flipped)
    inv_depth_fused = fuse_inv_depth(depth, inv_depth_hat, method=method)
    xs = torch.linspace(0., 1., W, device=depth.device,
                        dtype=depth.dtype).repeat(B, C, H, 1)
    mask = 1.0 - torch.clamp(20. * (xs - 0.05), 0., 1.)
    mask_hat = flip_lr(mask)
    return mask_hat * depth + mask * inv_depth_hat + \
           (1.0 - mask - mask_hat) * inv_depth_fused

# Old get text which performs text processing and augmentation for strctured text
# # Do lanagugage description augmentation here
# def get_text(data_path, sample_path, mode="train", dataset=None, combine_words_no_area = False, close_car_percent=0.01, far_car_percent=0.001):
#     text_list = []
#     for i in range(len(sample_path)):  # B=4
#         txt_path = data_path+"/"+sample_path[i].split(' ')[0][:-4]+'.txt'
#         if mode == "train":
#             room_name = ""
#             room_name_list = sample_path[i].split(' ')[0].split("_")[:-2]
#             for i in range(len(room_name_list)):
#                 word = room_name_list[i]
#                 if i == 0:
#                     room_name = room_name+word[1:]+" "
#                 else:
#                     room_name = room_name+word+" "
#         elif mode == "eval":
#             room_name = sample_path[i].split(' ')[0].split("/")[0]+" "

#         if dataset == "kitti":
#             # room_name = "outdoor scene "
#             image_area = 1216 * 352
#         if dataset == "nyu":
#             image_area = 480 * 640
#         with open(txt_path, 'r') as file:
#             # text = "A "+room_name+"with "
#             text = "An image with "
#             object_list = []
#             area_percent_list = []
#             for j, line in enumerate(file):
#                 # if j % 2 == 0:
#                 #     word = line.strip()
#                 #     object_list.append(word)
#                 # else:
#                 #     coords = line.split(' ')
#                 #     area = (float(coords[3])-float(coords[1]))*(float(coords[2])-float(coords[0]))
#                 #     area_list.append(area)
#                 object_list.append(line[:line.rfind(" ")])
#                 area_percent_list.append(float(line[line.rfind(" "):]))

#             # remove instance based on prob=lamda/box area
#             # assert len(object_list) == len(area_percent_list)
#             # if mode == "train":
#             #     i = 0
#             #     while i < len(object_list):
#             #         area_percent = area_percent_list[i]
#             #         remove_prob = 1 / (1 + np.exp(-area_percent))  # sigmoid
#             #         remove_prob = 1 - remove_prob
#             #         # print(object_list[i], round(remove_prob,4))
#             #         if random.random() < remove_prob:
#             #             del object_list[i]
#             #             del area_percent_list[i]
#             #         else:
#             #             i += 1

#             # swap word as augmentation
#             length = len(object_list)
#             if mode == "train":
#                 indices = list(range(length))
#                 random.shuffle(indices)
#                 object_list = [object_list[i] for i in indices]
#                 area_percent_list = [area_percent_list[i] for i in indices]

#             for i in range(length):
#                 # add object scale
#                 if object_list[i] == "car":
#                     if area_percent_list[i] > close_car_percent:
#                         object_list[i] = "close car"
#                     if area_percent_list[i] < far_car_percent:
#                         object_list[i] = "far car"


#                 text += object_list[i]
#                 # include area percent
#                 if combine_words_no_area is False:
#                     text += " occupied " + str(round(area_percent_list[i]*100, 2)) + "% of image, "
#                 else:
#                     text += ", "
#                 # text += ", "  # for combine words
#                 # text += ", " + str(round(area_list[i]/image_area*100, 2)) + "%; "
#             if combine_words_no_area is True:
#                 text = combine_repetitive_words(text)  # for combine words
#             text = text.replace("_", " ")

#             if combine_words_no_area is True:
#                 text = text[:-1] + "."  # for combine words
#             else:
#                 text = text[:-2] + "."

#             # This handles nested parentheses
#             pattern = r' \([^()]*\)'
#             while re.search(pattern, text):
#                 text = re.sub(pattern, '', text)

#             # print(text, flush=True)

#             text_list.append(text)
#     # print(text_list, flush=True)
#     return text_list


def count_parameters(model):
    if hasattr(model, 'parameters'):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        total = model.numel()
        trainable = model.numel() if model.requires_grad else 0
    return total, trainable


def print_model_parameters(models, model_name="Model", print_or_log="print"):
    total = 0
    trainable = 0
    for m in models:
        t, tr = count_parameters(m)
        total += t
        trainable += tr

    total_m = total / 1e6
    trainable_m = trainable / 1e6
    non_trainable_m = (total - trainable) / 1e6
    mem_mb = total * 4 / (1024 ** 2)

    lines = [
        '=' * 50,
        f'Parameter stats - {model_name}',
        '=' * 50,
        f'Total params: {total_m:.1f}M',
        f'Trainable params: {trainable_m:.1f}M',
        f'Non-trainable params: {non_trainable_m:.1f}M',
        f'Memory (FP32): {mem_mb:.2f} MB',
        '=' * 50,
    ]
    emit = print if print_or_log == "print" else __import__('logging').info
    emit('\n' + '\n'.join(lines))


class _Tee:
    """Write each line to multiple file-like targets (e.g. stdout + a log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return any(getattr(s, 'isatty', lambda: False)() for s in self.streams)


def setup_tee_logging(log_path):
    """Mirror stdout and stderr to *log_path* in addition to the console.

    Returns the open file handle so the caller can close it explicitly if needed
    (otherwise it stays open until the process exits, which is fine for a script).
    """
    import sys
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    log_file = open(log_path, 'a', buffering=1)  # line-buffered
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    print(f"[log] mirroring stdout/stderr to {log_path}")
    return log_file