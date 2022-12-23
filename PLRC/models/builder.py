# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import PLRC.models.plrc_loss as plrc_loss
from PLRC.models.resnet import ResNetPLRC
from PLRC.models.utils import *
import torch.nn.functional as F


class PLRC(nn.Module):
    def __init__(self, args, dim=128, K=65536, m=0.999, T=0.07, mlp=False):

        super(PLRC, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = ResNetPLRC()
        self.encoder_k = ResNetPLRC()

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        self.plrc_loss = plrc_loss.SupConLoss(args)
        self.plrc_loss_image = plrc_loss.NCEAverage(args)

        self.mask_size = 56
        self.mask_neg_num = 15
        self.mask_pos_num = 1
        self.mask_grid_num = 4
        self.mask_area_avgnum = 32

        self.keep_negs = True
        self.logis_sum = "exp_logits"
        self.logis_avg = False
        self.scale = False
        self.keep_point = 16

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(
        self,
        xs,
        ids,
        cur_epoch,
        masks_all,
        obj,
        xs_2=None,
        coord_multiv=None,
        coord_multiv_2=None,
    ):

        input_shuffle = True
        fxs, fxs_pre, fxs_hist, fxs_hist_pre = [], [], [], []
        gxs, pxs = [], []
        fxs_moco, fxs_pre_moco, fxs_hist_moco, fxs_hist_pre_moco = [], [], [], []
        gxs_moco, pxs_moco = [], []

        fxs_ts, fxs_pre_ts, fxs_hist_ts, fxs_hist_pre_ts = [], [], [], []
        fxs_ts_f, fxs_pre_ts_f, fxs_hist_ts_f, fxs_hist_pre_ts_f = [], [], [], []

        for gx, gx_2, is_key, masks in zip(xs, xs_2, [0, 1], masks_all):

            if is_key:
                with torch.no_grad():
                    self._momentum_update_key_encoder()

                    x, idx_unshuffle = self._batch_shuffle_ddp(gx)

                    fx, fx_2_f, fx_2, px = self.encoder_k(x, mode="point", dy=True)

                    fx = self._batch_unshuffle_ddp(fx, idx_unshuffle)

                    if fx_2 is not None:
                        fx_2 = self._batch_unshuffle_ddp(fx_2, idx_unshuffle)
                        fx_2_f = self._batch_unshuffle_ddp(fx_2_f, idx_unshuffle)
                    else:
                        fx_2 = fx

                    fx_moco = torch.mean(fx, dim=(2, 3))

                    H, W = self.mask_size, self.mask_size
                    grids = masks

                    grids[:, :, :, 0], grids[:, :, :, 1] = (
                        grids[:, :, :, 0] / (H - 1) * 2 - 1,
                        grids[:, :, :, 1] / (W - 1) * 2 - 1,
                    )
                    pixel_feat = nn.functional.grid_sample(fx, grids)

                    B_pf, C_pf, N_pf, P_pf = pixel_feat.shape

                    pixel = F.normalize(
                        pixel_feat_q_arch.permute(0, 1, 3, 2)[:, :, :, 0].permute(
                            0, 2, 1
                        ),
                        dim=-1,
                        p=2,
                    )
                    pixel_top = F.normalize(
                        pixel_feat[:, :, 0:1, :]
                        .permute(0, 1, 3, 2)[:, :, :, 0]
                        .permute(0, 2, 1),
                        dim=-1,
                        p=2,
                    )
                    m = (
                        torch.matmul(pixel_top, pixel.permute(0, 2, 1))
                        .max(dim=2)[1]
                        .unsqueeze(-1)
                        .repeat(1, 1, C_pf)
                        .cuda()
                    )
                    pixel_feat_top = (
                        pixel_feat[:, :, 0:1, :]
                        .permute(0, 1, 3, 2)[:, :, :, 0]
                        .permute(0, 2, 1)
                        .gather(dim=1, index=m)
                        .unsqueeze(0)
                        .permute(1, 3, 0, 2)[:, :, :, : self.keep_point]
                    )

                    fx_append = pixel_feat[:, :, 0, :]

                    fx_append = fx_append.permute(0, 2, 1).reshape(-1, C_pf)

                    fxs_hist_pre.append(fx_append)
                    fxs_hist_pre_moco.append(fx_moco)

                fxs_pre.append(pixel_feat_top)
                fxs.append(pixel_feat_top)
                fxs_hist.append(pixel_feat_top)

                B_fx, C_fx, H_fx, W_fx = fx.shape
                fx = fx.permute(0, 2, 3, 1).reshape(B_fx * H_fx * W_fx, C_fx)

                fx_2 = torch.nn.functional.interpolate(
                    fx_2, size=(self.mask_size, self.mask_size), mode="bilinear"
                )
                B_fx_, C_fx_, H_fx_, W_fx_ = fx_2.shape

                fx_2 = fx_2.permute(0, 2, 3, 1).reshape(B_fx_ * H_fx_ * W_fx_, C_fx_)

                fxs_pre_ts.append(fx_2)
                fxs_ts.append(fx_2)
                fxs_hist_ts.append(fx_2)

                fx_2_f = torch.nn.functional.interpolate(
                    fx_2_f, size=(self.mask_size, self.mask_size), mode="bilinear"
                )
                B_fx_f, C_fx_f, H_fx_f, W_fx_f = fx_2_f.shape

                fx_2_f = fx_2_f.permute(0, 2, 3, 1).reshape(
                    B_fx_f * H_fx_f * W_fx_f, C_fx_f
                )

                fxs_pre_ts_f.append(fx_2_f)
                fxs_ts_f.append(fx_2_f)
                fxs_hist_ts_f.append(fx_2_f)

                fxs_pre_moco.append(fx_moco)
                fxs_moco.append(fx_moco)
                fxs_hist_moco.append(fx_moco)

            else:
                fx_q, fx_q_2_f, fx_q_2, px = self.encoder_q(gx, mode="point")

                if fx_q_2 is None:
                    fx_q_2 = fx_q
                fx_moco = torch.mean(fx_q, dim=(2, 3))
                H, W = self.mask_size, self.mask_size
                grids_q = masks.clone()
                grids_q[:, :, :, 0], grids_q[:, :, :, 1] = (
                    grids_q[:, :, :, 0] / (H - 1) * 2 - 1,
                    grids_q[:, :, :, 1] / (W - 1) * 2 - 1,
                )
                pixel_feat_q = nn.functional.grid_sample(fx_q, grids_q)
                B_pf, C_pf, N_pf, P_pf = pixel_feat_q.shape
                pixel_feat_q_arch = pixel_feat_q[:, :, 0:1, :]

                fxs_pre.append(pixel_feat_q[:, :, 0:1, : self.keep_point])

                fxs.append(pixel_feat_q[:, :, 0:1, : self.keep_point])

                fx_q_2 = torch.nn.functional.interpolate(
                    fx_q_2, size=(self.mask_size, self.mask_size), mode="bilinear"
                )
                B_fxq, C_fxq, H_fxq, W_fxq = fx_q_2.shape
                fx_q_2 = fx_q_2.permute(0, 2, 3, 1).reshape(
                    B_fxq * H_fxq * W_fxq, C_fxq
                )

                fxs_pre_ts.append(fx_q_2)

                fxs_ts.append(fx_q_2)

                fx_q_2_f = torch.nn.functional.interpolate(
                    fx_q_2_f, size=(self.mask_size, self.mask_size), mode="bilinear"
                )
                B_fxq_f, C_fxq_f, H_fxq_f, W_fxq_f = fx_q_2_f.shape
                fx_q_2_f = fx_q_2_f.permute(0, 2, 3, 1).reshape(
                    B_fxq_f * H_fxq_f * W_fxq_f, C_fxq_f
                )
                fxs_pre_ts_f.append(fx_q_2_f)

                fxs_ts_f.append(fx_q_2_f)

                gxs.append(gx)
                pxs.append(px)

                with torch.no_grad():
                    if input_shuffle and self.training:
                        x, idx_restore = self._batch_shuffle_ddp(gx_2)
                    else:
                        x = gx_2
                    fx, fx_2_f, fx_2, px = self.encoder_k(x, mode="point")

                    if input_shuffle and self.training:
                        fx = self._batch_unshuffle_ddp(fx, idx_restore)
                        if fx_2 is not None:
                            fx_2 = self._batch_unshuffle_ddp(fx_2, idx_restore)

                            fx_2_f = self._batch_unshuffle_ddp(fx_2_f, idx_restore)
                        else:
                            fx_2 = fx

                    grids = masks.clone()

                    grids[:, :, :, 0], grids[:, :, :, 1] = (
                        grids[:, :, :, 0] / (H - 1) * 2 - 1,
                        grids[:, :, :, 1] / (W - 1) * 2 - 1,
                    )

                    pixel_feat = nn.functional.grid_sample(fx, grids)
                    B_pf, C_pf, N_pf, P_pf = pixel_feat.shape

                fxs_pre.append(pixel_feat)
                fxs.append(pixel_feat)
                fxs_hist.append(pixel_feat)

                B_fx, C_fx, H_fx, W_fx = fx.shape
                fx = fx.permute(0, 2, 3, 1).reshape(B_fx * H_fx * W_fx, C_fx)

                fx_2 = torch.nn.functional.interpolate(
                    fx_2, size=(self.mask_size, self.mask_size), mode="bilinear"
                )
                B_fx_, C_fx_, H_fx_, W_fx_ = fx_2.shape
                fx_2 = fx_2.permute(0, 2, 3, 1).reshape(B_fx_ * H_fx_ * W_fx_, C_fx_)

                fxs_pre_ts.append(fx_2)
                fxs_ts.append(fx_2)
                fxs_hist_ts.append(fx_2)

                fx_2_f = torch.nn.functional.interpolate(
                    fx_2_f, size=(self.mask_size, self.mask_size), mode="bilinear"
                )
                B_fx_f, C_fx_f, H_fx_f, W_fx_f = fx_2_f.shape
                fx_2_f = fx_2_f.permute(0, 2, 3, 1).reshape(
                    B_fx_f * H_fx_f * W_fx_f, C_fx_f
                )

                fxs_pre_ts_f.append(fx_2_f)
                fxs_ts_f.append(fx_2_f)
                fxs_hist_ts_f.append(fx_2_f)
                fxs_pre_moco.append(fx_moco)
                fxs_moco.append(fx_moco)

        loss_image = self.plrc_loss_image(
            fxs_moco,
            fxs_pre_moco,
            fxs_hist_moco,
            fxs_hist_pre_moco,
            ids,
            gxs_moco,
            pxs_moco,
            None,
            self.training,
            cur_epoch,
            mode="image",
        )

        loss_point = self.plrc_loss(
            fxs,
            fxs_pre,
            fxs_hist,
            fxs_hist_pre,
            ids,
            gxs,
            pxs,
            None,
            self.training,
            cur_epoch,
            mode="point",
            len_obj=obj,
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

        return loss_image, loss_point


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
