# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import os
import subprocess
from typing import Any, Dict, List, Optional

import torch
import torchvision
from torch import Tensor

## 该函数的作用主要是 输出git相关的信息
def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))
    #subprocess.check_output可以运行以字符串输入的shell，git rev-parse HEAD是git 在终端中运行的命令，
    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message

## 定义dataloader中的 collate_fn函数，其中的do_round一般默认为False,不会对图像的大小进行舍入
def collate_fn(do_round, batch):
    #batch中的元素应该由两部分组成，一部分是[c，h,w]的原始图像信息，一部分是与该图像相关的数据字典
    #将batch个元素拼接成一个有两个元素的列表，其中元素0中是图像信息，元素1中是图像相关数据
    batch = list(zip(*batch))
    final_batch = {}
    ## 将batch的两个元素赋值到final_batch中
    final_batch["samples"] = NestedTensor.from_tensor_list(batch[0], do_round)  #（[bs,c,w,h]),[bs,w,h]）,c,w,h是一个维度中的最大值
    final_batch["targets"] = batch[1]  #一个列表，所以不用考虑不同样本数据维度不同的问题
    ## 对positive_map,应该也就是单词分布进行单独处理，将一个batch中的数据加载到一个参数中
    if "positive_map" in batch[1][0]:
        # we batch the positive maps here
        # Since in general each batch element will have a different number of boxes,
        # we collapse a single batch dimension to avoid padding. This is sufficient for our purposes.
        max_len = max([v["positive_map"].shape[1] for v in batch[1]])
        nb_boxes = sum([v["positive_map"].shape[0] for v in batch[1]])
        batched_pos_map = torch.zeros((nb_boxes, max_len), dtype=torch.bool)
        cur_count = 0
        for v in batch[1]:
            cur_pos = v["positive_map"]
            batched_pos_map[cur_count : cur_count + len(cur_pos), : cur_pos.shape[1]] = cur_pos
            cur_count += len(cur_pos)

        assert cur_count == len(batched_pos_map)
        # assert batched_pos_map.sum().item() == sum([v["positive_map"].sum().item() for v in batch[1]])
        final_batch["positive_map"] = batched_pos_map.float()# [num_boxes,batch_max_len]
    ## 完全同上
    if "positive_map_eval" in batch[1][0]:
        # we batch the positive maps here
        # Since in general each batch element will have a different number of boxes,
        # we collapse a single batch dimension to avoid padding. This is sufficient for our purposes.
        max_len = max([v["positive_map_eval"].shape[1] for v in batch[1]])
        nb_boxes = sum([v["positive_map_eval"].shape[0] for v in batch[1]])
        batched_pos_map = torch.zeros((nb_boxes, max_len), dtype=torch.bool)
        cur_count = 0
        for v in batch[1]:
            cur_pos = v["positive_map_eval"]
            batched_pos_map[cur_count : cur_count + len(cur_pos), : cur_pos.shape[1]] = cur_pos
            cur_count += len(cur_pos)

        assert cur_count == len(batched_pos_map)
        # assert batched_pos_map.sum().item() == sum([v["positive_map"].sum().item() for v in batch[1]])
        final_batch["positive_map_eval"] = batched_pos_map.float()
    ## 对answer相关信息的处理
    if "answer" in batch[1][0] or "answer_type" in batch[1][0]:
        answers = {}
        for f in batch[1][0].keys():
            if "answer" not in f:
                continue
            answers[f] = torch.stack([b[f] for b in batch[1]])
        final_batch["answers"] = answers #[bs,shape(answer[f])]

    return final_batch

# DETR中的设置的数据结构，就是将b*c*h*w的图像tensor和其对应的b*h*w的mask值封装在一起
# 需要有mask是因为图像并不是一样大小，bchw中都是取得最大尺寸，需要用mask记录原图像大小
class NestedTensor(object):
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask
    # 对pytorch中to函数的包装，在这里的调用一次to函数可以晚上对tensor的mask两个的to函数运行
    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        cast_mask = self.mask.to(*args, **kwargs) if self.mask is not None else None
        return type(self)(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    @classmethod
    def from_tensor_list(cls, tensor_list, do_round=False):
        # TODO make this more general
        # tensor_list 应该就是batch个原始图像tensor数据组成的列表
        # do_round就是决定是否将图片的大小四舍五入到128的倍数
        if tensor_list[0].ndim == 3:
            # TODO make it support different-sized images
            #找到各个维度的最大尺寸
            max_size = tuple(max(s) for s in zip(*[img.shape for img in tensor_list]))
            # min_size = tuple(min(tokenzier) for tokenzier in zip(*[img.shape for img in tensor_list]))
            #取各个维度最大值组成尺寸
            batch_shape = (len(tensor_list),) + max_size
            b, c, h, w = batch_shape
            ##如果需要舍入，则将h和w转换为p的倍数
            if do_round:
                # Round to an even size to avoid rounding issues in fpn
                p = 128
                h = h if h % p == 0 else (h // p + 1) * p
                w = w if w % p == 0 else (w // p + 1) * p
                batch_shape = b, c, h, w
            ## 将tensor_list 中的值封装成一个tensor和一个mask值
            dtype  = tensor_list[0].dtype
            device = tensor_list[0].device
            tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
            mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
            # 使用 zip可以令相同位置的不同属性对应在一起，这里根据img的属性对tensor和mask的值进行了调整
            for img, pad_img, m in zip(tensor_list, tensor, mask):
                pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
                m[: img.shape[1], : img.shape[2]] = False
        else:
            raise ValueError("not supported")
        return cls(tensor, mask) # [b, c, h, w]、[b,h, w]，,c,w,h是一个维度中的最大值
    #本意是显示属性的特殊方法
    def __repr__(self):
        return repr(self.tensors)


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty channel sizes.
    """
    if input.numel() > 0:
        return torch.nn.functional.interpolate(input, size, scale_factor, mode, align_corners)

    assert input.shape[0] != 0 or input.shape[1] != 0, "At least one of the two first dimensions must be non zero"

    if input.shape[1] == 0:
        # Pytorch doesn't support null dimension on the channel dimension, so we transpose to fake a null batch dim
        return torch.nn.functional.interpolate(input.transpose(0, 1), size, scale_factor, mode, align_corners).transpose(0, 1)

    # empty batch dimension is now supported in pytorch
    return torch.nn.functional.interpolate(input, size, scale_factor, mode, align_corners)


## 将 targrt中的一些 tensor转移到gpu中
def targets_to(targets: List[Dict[str, Any]], device):
    """Moves the target dicts to the given device."""
    excluded_keys = [
        "questionId",
        "tokens_positive",
        "tokens",
        "dataset_name",
        "sentence_id",
        "original_img_id",
        "nb_eval",
        "task_id",
        "original_id",
    ]
    return [{k: v.to(device) if k not in excluded_keys else v for k, v in t.items() if k != "caption"} for t in targets]


