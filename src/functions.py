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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import mindspore as ms
import mindspore.nn as nn
from src.SSIM import SSIM
from src.OneStep import CustomTrainOneStepCell, CustomEvalOneStepCell


def train_epoch(args, epoch, model, train_dataset, criterion, optimizer):
    batch_num = len(train_dataset)
    ssim_sum = 0.0
    mae_sum = 0.0
    mse_sum = 0.0

    loss_net = nn.WithLossCell(model, criterion)
    train_net = CustomTrainOneStepCell(loss_net, model, optimizer)
    for batch in range(batch_num):
        inputs = train_dataset.input[batch]
        targets = train_dataset.target[batch]

        inputs = ms.Tensor(inputs, dtype=ms.dtype.float32)
        targets = ms.Tensor(targets, dtype=ms.dtype.float32)
        train_net.set_train()
        outputs, loss = train_net(inputs, targets)

        mae_criterion = nn.MAELoss()
        mae_loss = mae_criterion(outputs, targets)/args.batch_size
        mae_sum += mae_loss

        mse_criterion = nn.MSELoss()
        mse_loss = mse_criterion(outputs, targets)/args.batch_size
        mse_sum += mse_loss

        ssim_criterion = SSIM()
        ssim = ssim_criterion(outputs, targets)
        ssim_sum += ssim

        if batch+1 and (batch+1)%args.display == 0:
            print('[Train] Epoch:{}, Batch:{}/{}, MAELoss:{}, MSELoss:{}, SSIM:{}'
                  .format(epoch+1, batch+1, batch_num, mae_loss, mse_loss, ssim))

    print('[Train-aver] Epoch:{}, MAELoss aver:{}, MSELoss aver:{}, SSIM aver:{}'
          .format(epoch+1, mae_sum/batch_num, mse_sum/batch_num, ssim_sum/batch_num))


def eval_epoch(args, epoch, model, test_dataset, criterion):
    batch_num = len(test_dataset)
    ssim_sum = 0.0
    mae_sum = 0.0
    mse_sum = 0.0

    loss_net = nn.WithLossCell(model, criterion)
    eval_net = CustomEvalOneStepCell(loss_net, model)
    for batch in range(batch_num):
        inputs = test_dataset.input[batch]
        targets = test_dataset.target[batch]

        inputs = ms.Tensor(inputs, dtype=ms.dtype.float32)
        targets = ms.Tensor(targets, dtype=ms.dtype.float32)

        outputs, loss = eval_net(inputs, targets)

        mae_criterion = nn.MAELoss()
        mae_loss = mae_criterion(outputs, targets)/args.batch_size
        mae_sum += mae_loss

        mse_criterion = nn.MSELoss()
        mse_loss = mse_criterion(outputs, targets)/args.batch_size
        mse_sum += mse_loss

        ssim_criterion = SSIM()
        ssim = ssim_criterion(outputs, targets)
        ssim_sum += ssim

        if batch+1 and (batch+1)%args.display == 0:
            print('[Eval] Epoch:{}, Batch:{}/{}, MAELoss:{}, MSELoss:{}, SSIM:{}'
                  .format(epoch+1, batch+1, batch_num, mae_loss, mse_loss, ssim))

    print('[Eval-aver] Epoch:{}, MAELoss aver:{}, MSELoss aver:{}, SSIM aver:{}'
          .format(epoch+1, mae_sum/batch_num, mse_sum/batch_num, ssim_sum/batch_num))
