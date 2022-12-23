# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""ImageNet dataset."""

import cv2
import numpy as np
import os

import torch
import torch.utils.data
import PLRC.datasets.transforms as transforms
import torchvision as tv
import torchvision.transforms.functional as tvf
import io
from PIL import Image
from shapely.geometry import Polygon
import json

# Data location

rng = np.random.default_rng()


class ImageNet(torch.utils.data.Dataset):
    def __init__(self, split, args, path=""):
        self.path = path
        self._split = split
        self._first_k = 0
        self.random_resizedcrop = None
        self._construct_imdb()

        self.mask_size = 56
        self.mask_neg_num = 15
        self.mask_pos_num = 1
        self.mask_grid_num = 4
        self.mask_area_avgnum = 32
        self.im_size = 224

        normalize = tv.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        mask_np = np.zeros((self.im_size, self.im_size))
        for hh in range(self.mask_grid_num):
            for ww in range(self.mask_grid_num):
                start_h, end_h = (
                    hh * (self.im_size // self.mask_grid_num),
                    (hh + 1) * (self.im_size // self.mask_grid_num)
                    if hh != self.mask_grid_num - 1
                    else self.im_size,
                )
                start_w, end_w = (
                    ww * (self.im_size // self.mask_grid_num),
                    (ww + 1) * (self.im_size // self.mask_grid_num)
                    if ww != self.mask_grid_num - 1
                    else self.im_size,
                )
                mask_np[start_h:end_h, start_w:end_w] = hh * self.mask_grid_num + ww + 1
        self.pano_mask = Image.fromarray(np.uint8(mask_np), "L")  #

        self.random_resizedcrop = transforms.RandomResizedCropCoord(
            self.im_size, scale=(0.2, 1.0)
        )

        self.random_resizedcrop_mask = transforms.RandomResizedCropCoord(
            self.mask_size, scale=(0.2, 1.0), interpolation=Image.NEAREST
        )
        self.randomhflip = transforms.RandomHorizontalFlipCoord()

        color_jitter = tv.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        rnd_color_jitter = tv.transforms.RandomApply([color_jitter], p=0.8)
        self._transform = tv.transforms.Compose(
            [
                rnd_color_jitter,
                tv.transforms.RandomGrayscale(p=0.2),
                tv.transforms.RandomApply(
                    [transforms.GaussianBlurSimple([0.1, 2.0])], p=0.5
                ),
                tv.transforms.ToTensor(),
                normalize,
            ]
        )
        self._transform_mask = tv.transforms.Compose(
            [
                tv.transforms.ToTensor(),
            ]
        )

    def _apply_single_transformation(self, n, im):
        if n % 2 == 1 and hasattr(self, "_transform_prime"):
            return self._transform_prime(im)
        else:
            return self._transform(im)

    def _construct_imdb(self):
        """Constructs the imdb."""
        data_dir = os.path.join(self.path, self._split)
        assert os.path.exists(data_dir), "{} dir not found".format(data_dir)

        # Map ImageNet class ids to contiguous ids
        self._class_ids = os.listdir(data_dir)
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}

        # Construct the image db
        self._imdb = []
        for class_id in self._class_ids:
            cont_id = self._class_id_cont_id[class_id]
            im_dir = os.path.join(data_dir, class_id)
            for ii, im_name in enumerate(sorted(os.listdir(im_dir))):
                if self._first_k and ii >= self._first_k:
                    break

                self._imdb.append(
                    {
                        "im_path": os.path.join(im_dir, im_name),
                        "class": cont_id,
                    }
                )

        self.num_classes = len(self._imdb)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index, scales, repeats = index
        else:
            scales = [None, None]
            repeats = [1]
        flag = 1
        anno_mask = True
        im = tv.datasets.folder.default_loader(self._imdb[index]["im_path"])

        width, height = im.size
        im_name = self._imdb[index]["im_path"].split("/")[-1].split(".")[0]
        pano_mask = self.pano_mask.resize((width, height), resample=Image.NEAREST)

        im_multiv = []
        im_multiv_2 = []
        mask_multiv = []

        obj_list_multiv = []

        coord_multiv = []
        coord_multiv_2 = []

        # augmentations
        for n, s in enumerate(scales):
            if s is not None:
                self._set_crop_size(s)

            iii = 0
            while True:
                im_ = im
                im_2 = im
                pano_mask_ = pano_mask

                i, j, h, w, height, width = self.random_resizedcrop.get_params(
                    im_, self.random_resizedcrop.scale, self.random_resizedcrop.ratio
                )
                im_, coord = self.random_resizedcrop.resized_crop(
                    im_, i, j, h, w, height, width
                )

                pano_mask_, _ = self.random_resizedcrop_mask.resized_crop(
                    pano_mask_, i, j, h, w, height, width
                )

                (
                    i_2,
                    j_2,
                    h_2,
                    w_2,
                    height_2,
                    width_2,
                ) = self.random_resizedcrop.get_params(
                    im_2, self.random_resizedcrop.scale, self.random_resizedcrop.ratio
                )
                im_2, coord_2 = self.random_resizedcrop.resized_crop(
                    im_2, i_2, j_2, h_2, w_2, height_2, width_2
                )

                polygon = Polygon([(i, j), (i + h, j), (i + h, j + w), (i, j + w)])
                other_polygon = Polygon(
                    [
                        (i_2, j_2),
                        (i_2 + h_2, j_2),
                        (i_2 + h_2, j_2 + w_2),
                        (i_2, j_2 + w_2),
                    ]
                )
                intersection = polygon.intersection(other_polygon)

                iii += 1

                if intersection.area / max(h * w, h_2 * w_2) >= 0.5 or iii > 100:
                    break

            if torch.rand(1) < 0.5:
                im_, coord = self.randomhflip(im_, coord)
                pano_mask_, _ = self.randomhflip(pano_mask_, coord)

            if torch.rand(1) < 0.5:
                im_2, coord_2 = self.randomhflip(im_2, coord_2)

            coord_multiv.append(coord)
            coord_multiv_2.append(coord_2)

            pano_mask_np_ = np.array(pano_mask_)

            obj_list_ = np.unique(pano_mask_np_)

            str_ = str(len(obj_list_))

            if len(obj_list_) <= 1:
                flag = -1

            obj_list_multiv.append(obj_list_)
            mask_multiv.append(pano_mask_np_)

            im_multiv.append(np.array(self._apply_single_transformation(n, im_)))
            im_multiv_2.append(np.array(self._apply_single_transformation(n, im_2)))

        if flag == 1:
            common_objects = list(
                set(obj_list_multiv[0]).intersection(set(obj_list_multiv[1]))
            )
            if len(common_objects) == 0:
                flag = -1
            else:
                obj = common_objects[np.random.randint(0, len(common_objects))]

        multiple_mask_multiv = []

        for n, s in enumerate(scales):
            if flag == 1:
                masks_list = []
                obj_list_ = obj_list_multiv[n]
                pano_mask_np_ = mask_multiv[n]

                bg_list = obj_list_[obj_list_ != obj]
                mask_np_pos_ = (pano_mask_np_ == obj).astype(np.int32)

                xs, ys = np.nonzero(mask_np_pos_)
                tmp_points = np.stack((xs, ys), axis=-1).astype(np.int32)
                mask_np_pos_ = rng.choice(
                    tmp_points, self.mask_pos_num * self.mask_area_avgnum
                )

                mask_np_pos_ = mask_np_pos_.reshape(
                    self.mask_pos_num, self.mask_area_avgnum, 2
                )

                masks_list.append(mask_np_pos_)

                for time_ in range(self.mask_neg_num):
                    obj_neg = bg_list[np.random.randint(0, len(bg_list))]
                    mask_np_neg = (pano_mask_np_ == obj_neg).astype(np.int32)
                    xs, ys = np.nonzero(mask_np_neg)
                    tmp_points = np.stack((xs, ys), axis=-1).astype(np.int32)
                    mask_np_neg = rng.choice(
                        tmp_points, self.mask_pos_num * self.mask_area_avgnum
                    )  # 1+Negative, Avg_num, 2

                    mask_np_neg = mask_np_neg.reshape(
                        self.mask_pos_num, self.mask_area_avgnum, 2
                    )

                    masks_list.append(mask_np_neg)

            else:
                masks_list = []
                obj_list_ = obj_list_multiv[n]
                pano_mask_np_ = mask_multiv[n]

                for time_ in range(1 + self.mask_neg_num):
                    obj = obj_list_[np.random.randint(0, len(obj_list_))]
                    mask_np_ = (pano_mask_np_ == obj).astype(np.int32)

                    xs, ys = np.nonzero(mask_np_)
                    tmp_points = np.stack((xs, ys), axis=-1).astype(np.int32)
                    mask_np_ = rng.choice(
                        tmp_points, self.mask_pos_num * self.mask_area_avgnum
                    )
                    mask_np_ = mask_np_.reshape(
                        self.mask_pos_num, self.mask_area_avgnum, 2
                    )
                    masks_list.append(mask_np_)

            masks_list = np.stack(masks_list, axis=0)
            masks_list = masks_list.reshape(
                (1 + self.mask_neg_num) * self.mask_pos_num, self.mask_area_avgnum, 2
            )  # Negative, Avg_num, 2 -- N, POS_NUM, AVG_NUM, 2
            multiple_mask_multiv.append(masks_list)

        cls_labels = self._imdb[index]["class"]
        return (
            im_multiv,
            index,
            cls_labels,
            multiple_mask_multiv,
            flag,
            im_multiv_2,
            coord_multiv,
            coord_multiv_2,
        )

    def __len__(self):
        return len(self._imdb)
