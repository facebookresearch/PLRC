# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from scipy.stats import ortho_group
import torch.distributed as du
import sys
import numpy as np
from PIL import Image
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import PLRC.models.plrc_loss as plrc_loss
from PLRC.models.resnet import ResNetPLRC
import PLRC.models.builder as builder


def build_mlp1(input_dim, output_dim):

    return build_more_fcs(
        input_dim, output_dim, False, 1, 2048, "none", True, "none", True, True
    )


def append_fc_layers(fcs, norm_fc, use_bias, input_dim, dim_fc):

    fcs.append(nn.Linear(input_dim, dim_fc, bias=use_bias))
    fcs.append(nn.ReLU(inplace=True))

    return fcs


def append_output_layer(
    fcs, norm_out, use_bias_out, use_weight_out, input_dim, output_dim
):

    fcs.append(nn.Linear(input_dim, output_dim, bias=use_bias_out))

    return fcs


def build_more_fcs(
    input_dim,
    output_dim,
    first_relu,
    more_fc,
    dim_fc,
    norm_fc,
    use_bias,
    norm_out,
    use_bias_out,
    use_weight_out,
):
    fcs = []
    for _ in range(more_fc):
        fcs = append_fc_layers(fcs, norm_fc, use_bias, input_dim, dim_fc)
        input_dim = dim_fc
    fcs = append_output_layer(
        fcs, norm_out, use_bias_out, use_weight_out, input_dim, output_dim
    )
    fcs = nn.Sequential(*fcs)

    return fcs


def overlap(q, k, coord_q, coord_k, mask_size, pos_ratio=0.5):
    """q, k: N * C * H * W
    coord_q, coord_k: N * 4 (x_upper_left, y_upper_left, x_lower_right, y_lower_right)
    """
    N, _, C = q.shape  # -1 49 c
    H, W = mask_size, mask_size
    # [1, 7, 7]
    x_array = (
        torch.arange(0.0, float(W), dtype=coord_q.dtype, device=coord_q.device)
        .view(1, 1, -1)
        .repeat(1, H, 1)
    )
    y_array = (
        torch.arange(0.0, float(H), dtype=coord_q.dtype, device=coord_q.device)
        .view(1, -1, 1)
        .repeat(1, 1, W)
    )
    # [bs, 1, 1]
    q_bin_width = ((coord_q[:, 2] - coord_q[:, 0]) / W).view(-1, 1, 1)
    q_bin_height = ((coord_q[:, 3] - coord_q[:, 1]) / H).view(-1, 1, 1)
    k_bin_width = ((coord_k[:, 2] - coord_k[:, 0]) / W).view(-1, 1, 1)
    k_bin_height = ((coord_k[:, 3] - coord_k[:, 1]) / H).view(-1, 1, 1)
    # [bs, 1, 1]
    q_start_x = coord_q[:, 0].view(-1, 1, 1)
    q_start_y = coord_q[:, 1].view(-1, 1, 1)
    k_start_x = coord_k[:, 0].view(-1, 1, 1)
    k_start_y = coord_k[:, 1].view(-1, 1, 1)

    # [bs, 1, 1]
    q_bin_diag = torch.sqrt(q_bin_width**2 + q_bin_height**2)
    k_bin_diag = torch.sqrt(k_bin_width**2 + k_bin_height**2)
    max_bin_diag = torch.max(q_bin_diag, k_bin_diag)

    # [bs, 7, 7]
    center_q_x = (x_array + 0.5) * q_bin_width + q_start_x
    center_q_y = (y_array + 0.5) * q_bin_height + q_start_y
    center_k_x = (x_array + 0.5) * k_bin_width + k_start_x
    center_k_y = (y_array + 0.5) * k_bin_height + k_start_y

    # [bs, 49, 49]
    dist_center = (
        torch.sqrt(
            (center_q_x.view(-1, H * W, 1) - center_k_x.view(-1, 1, H * W)) ** 2
            + (center_q_y.view(-1, H * W, 1) - center_k_y.view(-1, 1, H * W)) ** 2
        )
        / max_bin_diag
    )
    pos_mask = (dist_center < pos_ratio).float().detach()
    return pos_mask
