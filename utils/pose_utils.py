"""
ref:
https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
https://github.com/facebookresearch/QuaterNet/blob/master/common/quaternion.py
https://github.com/arraiyopensource/kornia/blob/master/kornia/geometry/conversions.py
"""
# from math import acos, cos, pi, sin
# import numpy as np
import torch
# import torch.nn.functional as F
# from numba import jit, njit
# from numpy import linalg as LA
# from transforms3d.axangles import axangle2mat, mat2axangle
# from transforms3d.euler import _AXES2TUPLE, _NEXT_AXIS, _TUPLE2AXES, euler2mat, euler2quat, mat2euler, quat2euler
# from transforms3d.quaternions import mat2quat, quat2mat

from utils.pose_error import re

pixel_coords = None

def get_closest_rot(rot_est, rot_gt, sym_info):
    """get the closest rot_gt given rot_est and sym_info.

    rot_est: ndarray
    rot_gt: ndarray
    sym_info: None or Kx3x3 ndarray, m2m
    """
    if sym_info is None:
        return rot_gt
    if isinstance(sym_info, torch.Tensor):
        sym_info = sym_info.cpu().numpy()
    if len(sym_info.shape) == 2:
        sym_info = sym_info.reshape((1, 3, 3))
    # find the closest rot_gt with smallest re
    r_err = re(rot_est, rot_gt)
    closest_rot_gt = rot_gt
    for i in range(sym_info.shape[0]):
        # R_gt_m2c x R_sym_m2m ==> R_gt_sym_m2c
        rot_gt_sym = rot_gt.dot(sym_info[i])
        cur_re = re(rot_est, rot_gt_sym)
        if cur_re < r_err:
            r_err = cur_re
            closest_rot_gt = rot_gt_sym

    return closest_rot_gt



