# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Miscellaneous functions."""

import os
import sys
import datetime
import pytz
import math
import subprocess
import numpy as np
from scipy.spatial import distance

from kasal.bop_toolkit_lib import transform


def log(s):
  """A logging function.

  :param s: String to print (with the current date and time).
  """
  # Use UTC time for logging.
  utc_now = pytz.utc.localize(datetime.datetime.utcnow())
  # pst_now = utc_now.astimezone(pytz.timezone("America/Los_Angeles"))
  utc_now_str = '{}/{}|{:02d}:{:02d}:{:02d}'.format(
    utc_now.month, utc_now.day, utc_now.hour, utc_now.minute, utc_now.second)

  # sys.stdout.write('{}: {}\n'.format(time.strftime('%m/%d|%H:%M:%S'), s))
  sys.stdout.write('{}: {}\n'.format(utc_now_str, s))
  sys.stdout.flush()


def ensure_dir(path):
  """Ensures that the specified directory exists.

  :param path: Path to the directory.
  """
  if not os.path.exists(path):
    os.makedirs(path)


def get_symmetry_transformations(model_info, max_sym_disc_step):
  """Returns a set of symmetry transformations for an object model.

  :param model_info: See files models_info.json provided with the datasets.
  :param max_sym_disc_step: The maximum fraction of the object diameter which
    the vertex that is the furthest from the axis of continuous rotational
    symmetry travels between consecutive discretized rotations.
  :return: The set of symmetry transformations.
  """
  # Discrete symmetries.
  trans_disc = [{'R': np.eye(3), 't': np.array([[0, 0, 0]]).T}]  # Identity.
  if 'symmetries_discrete' in model_info:
    for sym in model_info['symmetries_discrete']:
      sym_4x4 = np.reshape(sym, (4, 4))
      R = sym_4x4[:3, :3]
      t = sym_4x4[:3, 3].reshape((3, 1))
      trans_disc.append({'R': R, 't': t})

  # Discretized continuous symmetries.
  trans_cont = []
  if 'symmetries_continuous' in model_info:
    for sym in model_info['symmetries_continuous']:
      axis = np.array(sym['axis'])
      offset = np.array(sym['offset']).reshape((3, 1))

      # (PI * diam.) / (max_sym_disc_step * diam.) = discrete_steps_count
      discrete_steps_count = int(np.ceil(np.pi / max_sym_disc_step))

      # Discrete step in radians.
      discrete_step = 2.0 * np.pi / discrete_steps_count

      for i in range(1, discrete_steps_count):
        R = transform.rotation_matrix(i * discrete_step, axis)[:3, :3]
        t = -R.dot(offset) + offset
        trans_cont.append({'R': R, 't': t})

  # Combine the discrete and the discretized continuous symmetries.
  trans = []
  for tran_disc in trans_disc:
    if len(trans_cont):
      for tran_cont in trans_cont:
        R = tran_cont['R'].dot(tran_disc['R'])
        t = tran_cont['R'].dot(tran_disc['t']) + tran_cont['t']
        trans.append({'R': R, 't': t})
    else:
      trans.append(tran_disc)

  return trans


def project_pts(pts, K, R, t):
  """Projects 3D points.

  :param pts: nx3 ndarray with the 3D points.
  :param K: 3x3 ndarray with an intrinsic camera matrix.
  :param R: 3x3 ndarray with a rotation matrix.
  :param t: 3x1 ndarray with a translation vector.
  :return: nx2 ndarray with 2D image coordinates of the projections.
  """
  assert (pts.shape[1] == 3)
  P = K.dot(np.hstack((R, t)))
  pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))
  pts_im = P.dot(pts_h.T)
  pts_im /= pts_im[2, :]
  return pts_im[:2, :].T


class Precomputer(object):
  """ Caches pre_Xs, pre_Ys for a 30% speedup of depth_im_to_dist_im()
  """
  xs, ys = None, None
  pre_Xs,pre_Ys = None,None
  depth_im_shape = None
  K = None

  @staticmethod
  def precompute_lazy(depth_im, K):
    """ Lazy precomputation for depth_im_to_dist_im() if depth_im.shape or K changes

    :param depth_im: hxw ndarray with the input depth image, where depth_im[y, x]
      is the Z coordinate of the 3D point [X, Y, Z] that projects to pixel [x, y],
      or 0 if there is no such 3D point (this is a typical output of the
      Kinect-like sensors).
    :param K: 3x3 ndarray with an intrinsic camera matrix.
    :return: hxw ndarray (Xs/depth_im, Ys/depth_im)
    """
    if depth_im.shape != Precomputer.depth_im_shape:
      Precomputer.depth_im_shape = depth_im.shape
      Precomputer.xs, Precomputer.ys = np.meshgrid(
        np.arange(depth_im.shape[1]), np.arange(depth_im.shape[0]))

    if depth_im.shape != Precomputer.depth_im_shape \
          or not np.all(K == Precomputer.K):
      Precomputer.K = K
      Precomputer.pre_Xs = (Precomputer.xs - K[0, 2]) / np.float64(K[0, 0])
      Precomputer.pre_Ys = (Precomputer.ys - K[1, 2]) / np.float64(K[1, 1])

    return Precomputer.pre_Xs, Precomputer.pre_Ys
  

def depth_im_to_dist_im_fast(depth_im, K):
  """Converts a depth image to a distance image.

  :param depth_im: hxw ndarray with the input depth image, where depth_im[y, x]
    is the Z coordinate of the 3D point [X, Y, Z] that projects to pixel [x, y],
    or 0 if there is no such 3D point (this is a typical output of the
    Kinect-like sensors).
  :param K: 3x3 ndarray with an intrinsic camera matrix.
  :return: hxw ndarray with the distance image, where dist_im[y, x] is the
    distance from the camera center to the 3D point [X, Y, Z] that projects to
    pixel [x, y], or 0 if there is no such 3D point.
  """
  # Only recomputed if depth_im.shape or K changes.
  pre_Xs, pre_Ys = Precomputer.precompute_lazy(depth_im, K)

  dist_im = np.sqrt(
    np.multiply(pre_Xs, depth_im)**2 +
    np.multiply(pre_Ys, depth_im)**2 +
    depth_im.astype(np.float64)**2)

  return dist_im


def depth_im_to_dist_im(depth_im, K):
  """Converts a depth image to a distance image.
  :param depth_im: hxw ndarray with the input depth image, where depth_im[y, x]
    is the Z coordinate of the 3D point [X, Y, Z] that projects to pixel [x, y],
    or 0 if there is no such 3D point (this is a typical output of the
    Kinect-like sensors).
  :param K: 3x3 ndarray with an intrinsic camera matrix.
  :return: hxw ndarray with the distance image, where dist_im[y, x] is the
    distance from the camera center to the 3D point [X, Y, Z] that projects to
    pixel [x, y], or 0 if there is no such 3D point.
  """
  xs, ys = np.meshgrid(
    np.arange(depth_im.shape[1]), np.arange(depth_im.shape[0]))

  Xs = np.multiply(xs - K[0, 2], depth_im) * (1.0 / K[0, 0])
  Ys = np.multiply(ys - K[1, 2], depth_im) * (1.0 / K[1, 1])

  dist_im = np.sqrt(
    Xs.astype(np.float64)**2 +
    Ys.astype(np.float64)**2 +
    depth_im.astype(np.float64)**2)
  # dist_im = np.linalg.norm(np.dstack((Xs, Ys, depth_im)), axis=2)  # Slower.

  return dist_im

def clip_pt_to_im(pt, im_size):
  """Clips a 2D point to the image frame.

  :param pt: 2D point (x, y).
  :param im_size: Image size (width, height).
  :return: Clipped 2D point (x, y).
  """
  return [min(max(pt[0], 0), im_size[0] - 1),
          min(max(pt[1], 0), im_size[1] - 1)]


def calc_2d_bbox(xs, ys, im_size=None, clip=False):
  """Calculates 2D bounding box of the given set of 2D points.

  :param xs: 1D ndarray with x-coordinates of 2D points.
  :param ys: 1D ndarray with y-coordinates of 2D points.
  :param im_size: Image size (width, height) (used for optional clipping).
  :param clip: Whether to clip the bounding box (default == False).
  :return: 2D bounding box (x, y, w, h), where (x, y) is the top-left corner
    and (w, h) is width and height of the bounding box.
  """
  bb_min = [xs.min(), ys.min()]
  bb_max = [xs.max(), ys.max()]
  if clip:
    assert (im_size is not None)
    bb_min = clip_pt_to_im(bb_min, im_size)
    bb_max = clip_pt_to_im(bb_max, im_size)
  return [bb_min[0], bb_min[1], bb_max[0] - bb_min[0], bb_max[1] - bb_min[1]]


def calc_3d_bbox(xs, ys, zs):
  """Calculates 3D bounding box of the given set of 3D points.

  :param xs: 1D ndarray with x-coordinates of 3D points.
  :param ys: 1D ndarray with y-coordinates of 3D points.
  :param zs: 1D ndarray with z-coordinates of 3D points.
  :return: 3D bounding box (x, y, z, w, h, d), where (x, y, z) is the top-left
    corner and (w, h, d) is width, height and depth of the bounding box.
  """
  bb_min = [xs.min(), ys.min(), zs.min()]
  bb_max = [xs.max(), ys.max(), zs.max()]
  return [bb_min[0], bb_min[1], bb_min[2],
          bb_max[0] - bb_min[0], bb_max[1] - bb_min[1], bb_max[2] - bb_min[2]]


def iou(bb_a, bb_b):
  """Calculates the Intersection over Union (IoU) of two 2D bounding boxes.

  :param bb_a: 2D bounding box (x1, y1, w1, h1) -- see calc_2d_bbox.
  :param bb_b: 2D bounding box (x2, y2, w2, h2) -- see calc_2d_bbox.
  :return: The IoU value.
  """
  # [x1, y1, width, height] --> [x1, y1, x2, y2]
  tl_a, br_a = (bb_a[0], bb_a[1]), (bb_a[0] + bb_a[2], bb_a[1] + bb_a[3])
  tl_b, br_b = (bb_b[0], bb_b[1]), (bb_b[0] + bb_b[2], bb_b[1] + bb_b[3])

  # Intersection rectangle.
  tl_inter = max(tl_a[0], tl_b[0]), max(tl_a[1], tl_b[1])
  br_inter = min(br_a[0], br_b[0]), min(br_a[1], br_b[1])

  # Width and height of the intersection rectangle.
  w_inter = br_inter[0] - tl_inter[0]
  h_inter = br_inter[1] - tl_inter[1]

  if w_inter > 0 and h_inter > 0:
    area_inter = w_inter * h_inter
    area_a = bb_a[2] * bb_a[3]
    area_b = bb_b[2] * bb_b[3]
    iou = area_inter / float(area_a + area_b - area_inter)
  else:
    iou = 0.0

  return iou


def transform_pts_Rt(pts, R, t):
  """Applies a rigid transformation to 3D points.

  :param pts: nx3 ndarray with 3D points.
  :param R: 3x3 ndarray with a rotation matrix.
  :param t: 3x1 ndarray with a translation vector.
  :return: nx3 ndarray with transformed 3D points.
  """
  assert (pts.shape[1] == 3)
  pts_t = R.dot(pts.T) + t.reshape((3, 1))
  return pts_t.T


def calc_pts_diameter(pts):
  """Calculates the diameter of a set of 3D points (i.e. the maximum distance
  between any two points in the set).

  :param pts: nx3 ndarray with 3D points.
  :return: The calculated diameter.
  """
  diameter = -1.0
  for pt_id in range(pts.shape[0]):
    pt_dup = np.tile(np.array([pts[pt_id, :]]), [pts.shape[0] - pt_id, 1])
    pts_diff = pt_dup - pts[pt_id:, :]
    max_dist = math.sqrt((pts_diff * pts_diff).sum(axis=1).max())
    if max_dist > diameter:
      diameter = max_dist
  return diameter


def calc_pts_diameter2(pts):
  """Calculates the diameter of a set of 3D points (i.e. the maximum distance
  between any two points in the set). Faster but requires more memory than
  calc_pts_diameter.

  :param pts: nx3 ndarray with 3D points.
  :return: The calculated diameter.
  """
  dists = distance.cdist(pts, pts, 'euclidean')
  diameter = np.max(dists)
  return diameter


def overlapping_sphere_projections(radius, p1, p2):
  """Checks if projections of two spheres overlap (approximated).

  :param radius: Radius of the two spheres.
  :param p1: [X1, Y1, Z1] center of the first sphere.
  :param p2: [X2, Y2, Z2] center of the second sphere.
  :return: True if the projections of the two spheres overlap.
  """
  if p1[2] == 0 or p2[2] == 0:
    return False

  # 2D projections of centers of the spheres.
  proj1 = (p1 / p1[2])[:2]
  proj2 = (p2 / p2[2])[:2]

  # Distance between the center projections.
  proj_dist = np.linalg.norm(proj1 - proj2)

  # The max. distance of the center projections at which the sphere projections,
  # i.e. sphere silhouettes, still overlap (approximated).
  proj_dist_thresh = radius * (1.0 / p1[2] + 1.0 / p2[2])

  return proj_dist < proj_dist_thresh


def get_error_signature(error_type, n_top, **kwargs):
  """Generates a signature for the specified settings of pose error calculation.

  :param error_type: Type of error.
  :param n_top: Top N pose estimates (with the highest score) to be evaluated
    for each object class in each image.
  :return: Generated signature.
  """
  error_sign = 'error=' + error_type + '_ntop=' + str(n_top)
  if error_type == 'vsd':
    if kwargs['vsd_tau'] == float('inf'):
      vsd_tau_str = 'inf'
    else:
      vsd_tau_str = '{:.3f}'.format(kwargs['vsd_tau'])
    error_sign += '_delta={:.3f}_tau={}'.format(
      kwargs['vsd_delta'], vsd_tau_str)
  return error_sign


def get_score_signature(correct_th, visib_gt_min):
  """Generates a signature for a performance score.

  :param visib_gt_min: Minimum visible surface fraction of a valid GT pose.
  :return: Generated signature.
  """
  eval_sign = 'th=' + '-'.join(['{:.3f}'.format(t) for t in correct_th])
  eval_sign += '_min-visib={:.3f}'.format(visib_gt_min)
  return eval_sign


def run_meshlab_script(meshlab_server_path, meshlab_script_path, model_in_path,
                       model_out_path, attrs_to_save):
  """Runs a MeshLab script on a 3D model.

  meshlabserver depends on X server. To remove this dependence (on linux), run:
  1) Xvfb :100 &
  2) export DISPLAY=:100.0
  3) meshlabserver <my_options>

  :param meshlab_server_path: Path to meshlabserver.exe.
  :param meshlab_script_path: Path to an MLX MeshLab script.
  :param model_in_path: Path to the input 3D model saved in the PLY format.
  :param model_out_path: Path to the output 3D model saved in the PLY format.
  :param attrs_to_save: Attributes to save:
    - vc -> vertex colors
    - vf -> vertex flags
    - vq -> vertex quality
    - vn -> vertex normals
    - vt -> vertex texture coords
    - fc -> face colors
    - ff -> face flags
    - fq -> face quality
    - fn -> face normals
    - wc -> wedge colors
    - wn -> wedge normals
    - wt -> wedge texture coords
  """
  meshlabserver_cmd = [meshlab_server_path, '-s', meshlab_script_path, '-i',
                       model_in_path, '-o', model_out_path]

  if len(attrs_to_save):
    meshlabserver_cmd += ['-m'] + attrs_to_save

  log(' '.join(meshlabserver_cmd))
  if subprocess.call(meshlabserver_cmd) != 0:
    exit(-1)
