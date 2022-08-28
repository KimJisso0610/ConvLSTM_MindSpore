# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


import mindspore as ms
import mindspore.nn as nn
import numpy as np


class SSIM(nn.Cell):

    def __init__(self):
        super(SSIM, self).__init__()
        self.criterion = nn.SSIM()

    def reshape_clamp(self, pred: ms.Tensor, target: ms.Tensor):
        r"""
        Reshapes and clamps the input tensors, returning a 4D tensor where batch and time dimension are combined.
        Args:
            pred (torch.Tensor): The predicted frame sequence as a 5D tensor (batch, frames, c, h, w).
            target (torch.Tensor): The ground truth frame sequence as a 5D tensor (batch, frames, c, h, w)
        Returns: the reshaped and clamped pred and target tensors.
        """
        pred = pred.reshape(-1, *pred.shape[2:])  # [b*t, ...]

        min_value = ms.Tensor(0.0, dtype=ms.dtype.float32)
        max_value = ms.Tensor(1.0, dtype=ms.dtype.float32)
        pred = ms.ops.clip_by_value((pred+1)/2, clip_value_min=min_value, clip_value_max=max_value)
        target = target.reshape(-1, *target.shape[2:])  # [b*t, ...]
        target = ms.ops.clip_by_value((target+1)/2, clip_value_min=min_value, clip_value_max=max_value)

        pred = np.repeat(pred[..., np.newaxis], 3, 1).squeeze()
        target = np.repeat(target[..., np.newaxis], 3, 1).squeeze()

        return pred, target

    def construct(self, pred, target):
        pred, target = self.reshape_clamp(pred, target)
        mean = ms.ops.ReduceMean()
        return mean(self.criterion(pred, target))

    @classmethod
    def to_display(cls, x):
        return 1.0-x
