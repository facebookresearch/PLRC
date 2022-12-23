# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import math
from scipy.stats import ortho_group

import torch
from torch import nn
import torch.nn.functional as F

import PLRC.models.builder as builder

import torch.distributed as du

import sys
import numpy as np

import matplotlib.pyplot as plt

from PIL import Image
import random
from PLRC.models.utils import *


class General_Loss(nn.Module):
    def __init__(self, args, apply=False):
        super().__init__()
        self.skip_first_few = False
        self.T = 0.2
        self.total_batch_size = args.batch_size
        self.world_size = args.world_size
        self.per_gpu_batch_size = self.total_batch_size // self.world_size
        self.gpu_rank = args.rank
        self.stop_grad_query = False
        self.stop_grad_key = False
        assert not self.stop_grad_key or not self.stop_grad_query
        self.loss_input_dim = 128
        self.crops_per_iter = 2
        self.apply = apply

        self.mask_size = 56
        self.mask_neg_num = 15
        self.mask_pos_num = 1
        self.mask_grid_num = 4
        self.mask_area_avgnum = 32

        self.ts_ratio = args.ts_ratio
        self.cl_ratio = args.cl_ratio
        self.im_ratio = args.im_ratio

        self.keep_negs = True
        self.logis_sum = "exp_logits"
        self.logis_avg = False
        self.scale = False
        self.keep_point = 16

    def forward(
        self,
        fxs,
        fxs_pre,
        fxs_hist,
        fxs_hist_pre,
        ids,
        gxs,
        pxs,
        rxs,
        is_train,
        cur_epoch,
        fxs_moco=None,
        fxs_pre_moco=None,
        fxs_hist_moco=None,
        fxs_hist_pre_moco=None,
        mix_image=None,
        mode="point",
        len_obj=None,
        fxs_ts=None,
        fxs_pre_ts=None,
        fxs_hist_ts=None,
        fxs_hist_pre_ts=None,
        coord_multiv=None,
        coord_multiv_2=None,
        fxs_ts_f=None,
        fxs_pre_ts_f=None,
        fxs_hist_ts_f=None,
        fxs_hist_pre_ts_f=None,
    ):
        if mode == "point":
            losses = self.forward_cont(
                fxs,
                fxs_pre,
                fxs_hist,
                fxs_hist_pre,
                ids,
                is_train,
                cur_epoch,
                mode=mode,
                len_obj=len_obj,
            )
            to_vis = None
            losses_point, losses_moco = losses
            losses_point.append(torch.zeros(1, dtype=torch.float32).cuda())
            losses_point.append(torch.zeros(1, dtype=torch.float32).cuda())

            losses_moco.append(torch.zeros(1, dtype=torch.float32).cuda())
            losses_moco.append(torch.zeros(1, dtype=torch.float32).cuda())
            return losses_point, losses_moco, to_vis

        else:
            losses = self.forward_cont(
                fxs,
                fxs_pre,
                fxs_hist,
                fxs_hist_pre,
                ids,
                is_train,
                cur_epoch,
                mode=mode,
                len_obj=len_obj,
                fxs_ts=fxs_ts,
                fxs_pre_ts=fxs_pre_ts,
                fxs_hist_ts=fxs_hist_ts,
                fxs_hist_pre_ts=fxs_hist_pre_ts,
                coord_multiv=coord_multiv,
                coord_multiv_2=coord_multiv_2,
                fxs_ts_f=fxs_ts_f,
                fxs_pre_ts_f=fxs_pre_ts_f,
                fxs_hist_ts_f=fxs_hist_ts_f,
                fxs_hist_pre_ts_f=fxs_hist_pre_ts_f,
            )
            to_vis = None

            losses.append(torch.zeros(1, dtype=torch.float32).cuda())
            losses.append(torch.zeros(1, dtype=torch.float32).cuda())
            return losses, to_vis

    def forward_cont(self, fxs, fxs_pre, fxs_hist, fxs_hist_pre, ids, is_train):
        raise NotImplementedError


class Memory_Loss(General_Loss):
    def __init__(self, args, apply=False):
        super().__init__(args)
        self.skip_first_few = True
        self.dim = 128
        dataset_size = 1281167
        self.Ks = [65536]
        self.build_loss()
        self.self_distillation = SelfDistillation(
            args,
            out_dim=self.mask_size * self.mask_size,
            warmup_teacher_temp=0.04,
            teacher_temp=0.07,
            warmup_teacher_temp_epochs=30,
            nepochs=100,
            student_temp=0.1,
            center_momentum=0.9,
        )

        stdv = 1.0 / math.sqrt(self.dim / 3)

        for ni, ns in enumerate([65536]):
            append = ""
            if ni > 0:
                append = "_%d" % ni
            self.register_buffer(
                "ptr" + append,
                torch.zeros(
                    [
                        1,
                    ]
                ),
            )
            self.register_buffer(
                "queue_x" + append, torch.rand(ns, self.dim).mul_(2 * stdv).add_(-stdv)
            )
            getattr(self, "ptr" + append).requires_grad = False
            getattr(self, "queue_x" + append).requires_grad = False

    def build_loss(self):
        raise NotImplementedError

    def compute_cont_loss(self, out_pos, out_neg, targets):
        raise NotImplementedError

    def build_positive(self, q, p, nq, np, ncrops):
        return torch.einsum("nbc,nc->nb", [q.view(np, ncrops, -1), p]).view(nq, 1)

    def build_positive_points(self, q, p, nq, np, ncrops):
        assert ncrops == 1
        assert nq == np
        B, N, C = q.shape
        assert B == p.shape[0] and C == p.shape[2] and N == p.shape[1]
        return torch.matmul(q, p.permute(0, 2, 1))

    def build_negative(self, q, append):
        return torch.einsum(
            "nc,kc->nk", [q, getattr(self, "queue_x" + append).clone().detach()]
        )

    def update_memory(self, fxs_hist):
        with torch.no_grad():

            for ki, ni in zip(fxs_hist, [0]):

                append = ""
                if ni > 0:
                    append = "_%d" % ni

                ptr = int(getattr(self, "ptr" + append).item())
                if self.world_size > 1:
                    ki = builder.concat_all_gather(ki)

                num_items = ki.size(0)
                K = self.Ks[ni]  # 66

                if num_items > K - ptr:
                    num_items = K - ptr
                getattr(self, "queue_x" + append)[ptr : (ptr + num_items), :] = ki[
                    :num_items
                ]
                ptr += num_items
                if ptr == K:
                    ptr = 0

                getattr(self, "ptr" + append)[0] = ptr

    def forward(
        self,
        fxs,
        fxs_pre,
        fxs_hist,
        fxs_hist_pre,
        ids,
        gxs,
        pxs,
        rxs,
        is_train,
        cur_epoch,
        fxs_moco=None,
        fxs_pre_moco=None,
        fxs_hist_moco=None,
        fxs_hist_pre_moco=None,
        mix_image=None,
        mode="point",
        len_obj=None,
        fxs_ts=None,
        fxs_pre_ts=None,
        fxs_hist_ts=None,
        fxs_hist_pre_ts=None,
        coord_multiv=None,
        coord_multiv_2=None,
        fxs_ts_f=None,
        fxs_pre_ts_f=None,
        fxs_hist_ts_f=None,
        fxs_hist_pre_ts_f=None,
    ):

        if mode == "point":
            # Still needs to change
            fxs = [F.normalize(fx, dim=1) for fx in fxs]
            fxs_pre = [F.normalize(fx, dim=1) for fx in fxs_pre]
            fxs_hist_pre = [F.normalize(fx, dim=1) for fx in fxs_hist_pre]

            loss = torch.zeros(1, dtype=torch.float32).cuda()
            loss_ts = torch.zeros(1, dtype=torch.float32).cuda()
            loss_cl = torch.zeros(1, dtype=torch.float32).cuda()

            for ii, tri in enumerate([[0, 1, 0]]):
                qi, p_2i, pi, ni = 0, 1, 2, 0
                q = fxs[qi].detach() if self.stop_grad_query else fxs[qi]
                p_2 = fxs_pre[p_2i].detach() if self.stop_grad_key else fxs_pre[p_2i]
                p = fxs_pre[pi].detach() if self.stop_grad_key else fxs_pre[pi]

                q_ts = (
                    fxs_pre_ts[qi].detach() if self.stop_grad_query else fxs_pre_ts[qi]
                )

                p_2_ts = (
                    fxs_pre_ts[p_2i].detach()
                    if self.stop_grad_key
                    else fxs_pre_ts[p_2i]
                )
                p_ts = fxs_pre_ts[pi].detach() if self.stop_grad_key else fxs_pre_ts[pi]

                q_ts_f = (
                    fxs_pre_ts_f[qi].detach()
                    if self.stop_grad_query
                    else fxs_pre_ts_f[qi]
                )
                p_2_ts_f = (
                    fxs_pre_ts_f[p_2i].detach()
                    if self.stop_grad_key
                    else fxs_pre_ts_f[p_2i]
                )
                p_ts_f = (
                    fxs_pre_ts_f[pi].detach()
                    if self.stop_grad_key
                    else fxs_pre_ts_f[pi]
                )

                append = ""
                if ni > 0:
                    append = "_%d" % ni
                nq = q.shape[0]
                np = p.shape[0]
                assert nq >= np
                ncrops = nq // np

                loss_cl += self.compute_cont_loss(p, q)

                _, c_ = q_ts.shape

                q_ts = q_ts.reshape(-1, self.mask_size * self.mask_size, c_)  # B, 49, C

                p_ts = p_ts.reshape(-1, self.mask_size * self.mask_size, c_)

                p_2_ts = p_2_ts.reshape(-1, self.mask_size * self.mask_size, c_)

                _, c_f = q_ts_f.shape
                q_ts_f = q_ts_f.reshape(
                    -1, self.mask_size * self.mask_size, c_f
                )  # B, 49, C

                p_ts_f = p_ts_f.reshape(-1, self.mask_size * self.mask_size, c_f)

                p_2_ts_f = p_2_ts_f.reshape(-1, self.mask_size * self.mask_size, c_f)

                overlap_mask = overlap(
                    q_ts, p_2_ts, coord_multiv[0], coord_multiv_2[0], self.mask_size
                )

                b_, hw, hw_ = torch.nonzero(
                    overlap_mask, as_tuple=True
                )  # B, 49, 49 --- B, q_ts, p_2_ts

                out_pos_f = self.build_positive_points(
                    q_ts_f, p_ts_f, nq, np, ncrops
                )  # Bx49*49

                out_pos_feat_f = out_pos_f[b_, hw, :]  # Mx49

                out_pos_t_f = self.build_positive_points(
                    p_2_ts_f, p_ts_f, nq, np, ncrops
                )  # BxB

                out_pos_t_feat_f = out_pos_t_f[b_, hw_, :]  # Mx49

                out_pos_feat_f = out_pos_feat_f / 128.0
                out_pos_t_feat_f = out_pos_t_feat_f / 128.0

                q_ts_feat = q_ts[b_, hw, :]

                q_ts_feat = torch.cat([out_pos_feat_f, q_ts_feat], dim=-1)

                p_2_ts_feat = p_2_ts[b_, hw_, :]

                p_2_ts_feat = torch.cat([out_pos_t_feat_f, p_2_ts_feat], dim=-1)

                loss_ts += self.self_distillation(
                    student_output=q_ts_feat,
                    teacher_output=p_2_ts_feat,
                    epoch=cur_epoch,
                )

                loss += self.ts_ratio * loss_ts + self.cl_ratio * loss_cl

            if is_train:
                self.update_memory(fxs_hist_pre)

            return [loss, torch.zeros(1, dtype=torch.float32).cuda()]

        elif mode == "image":

            fxs = [F.normalize(fx, dim=1) for fx in fxs]
            fxs_pre = [F.normalize(fx, dim=1) for fx in fxs_pre]
            fxs_hist_pre = [F.normalize(fx, dim=1) for fx in fxs_hist_pre]

            loss = torch.zeros(1, dtype=torch.float32).cuda()

            for ii, tri in enumerate([[0, 1, 0]]):
                qi, pi, ni = tri[0], tri[1], tri[2]
                q = fxs[qi].detach() if self.stop_grad_query else fxs[qi]
                p = fxs_pre[pi].detach() if self.stop_grad_key else fxs_pre[pi]

                append = ""
                if ni > 0:
                    append = "_%d" % ni
                nq = q.shape[0]
                np = p.shape[0]
                assert nq >= np
                ncrops = nq // np
                out_pos = self.build_positive(q, p, nq, np, ncrops)  # BxB
                out_neg = self.build_negative(q, append)  # BxN
                targets = torch.tensor(list(range(nq)), dtype=torch.long).cuda()

                loss += self.compute_cont_loss(out_pos, out_neg, targets)

            if is_train:
                self.update_memory(fxs_hist_pre)

            return [loss, torch.zeros(1, dtype=torch.float32).cuda()]


class SelfDistillation(nn.Module):
    def __init__(
        self,
        args,
        out_dim,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp_epochs,
        nepochs,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum

        self.register_buffer("center", torch.zeros(1, out_dim + 4096))

        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(
                    warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs
                ),
                np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )
        self.teacher_temp = teacher_temp

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach()  # .chunk(2)

        total_loss = 0
        n_loss_terms = 0
        loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)
        total_loss += loss.mean()
        self.update_center(teacher_output)

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        du.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * du.get_world_size())
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )


class NCEAverage(Memory_Loss):
    def build_loss(self):
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn_mask = nn.CrossEntropyLoss(reduction="none")

    def compute_cont_loss(self, out_pos, out_neg, targets, T=0.2):
        out_x = torch.cat([out_pos, out_neg], dim=1)
        out_x = torch.div(out_x, T)

        return self.loss_fn(out_x, targets)

    def compute_cont_loss_mask(self, out_pos, out_neg, targets, mask, T=0.2):
        out_x = torch.cat([out_pos, out_neg], dim=1)
        if out_x.shape[1] > len(mask):
            mask = torch.cat(
                [mask, torch.ones(out_x.shape[1] - len(mask)).cuda()], dim=0
            )
        out_x = torch.div(out_x, T)
        loss_fn = nn.CrossEntropyLoss(weight=mask)
        loss = loss_fn(out_x, targets)

        return loss


class SupConLoss(Memory_Loss):
    def build_loss(self):
        self.temperature = 0.2
        self.contrast_mode = "all"
        self.base_temperature = 0.07
        self.loss_fn = nn.CrossEntropyLoss()

    def compute_cont_loss(self, p, q):

        B, C, N, P = p.shape

        features = torch.cat(
            [
                p.permute(0, 2, 3, 1).reshape(-1, P, C),
                q.permute(0, 2, 3, 1).reshape(-1, P, C),
            ],
            dim=1,
        )

        batch_size = features.shape[0]

        mask = torch.eye(batch_size, dtype=torch.float32).cuda()

        contrast_count = features.shape[1]

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).cuda(),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
