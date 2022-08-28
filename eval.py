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
import numpy as np
from src.dataset import MovingMNISTDataset
from src.ConvLSTM import ConvLSTM
import mindspore as ms
import mindspore.nn as nn
from src.functions import eval_epoch
from src.config import get_args, Config
from mindspore import context


def main(args, config):
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device,
                        device_id=args.device_id)

    model = ConvLSTM(config)

    criterion = nn.MAELoss()
    optimizer = nn.Adam(model.trainable_params(), learning_rate=1e-3)

    data = np.load(args.dataset)
    data = data[:, :, np.newaxis, :, :]
    data = data.transpose(1, 0, 2, 3, 4)
    train_size = int(0.8*len(data))

    ms.set_seed(0)
    np.random.shuffle(data)
    train_data = data[:train_size]
    eval_data = data[train_size:]

    train_dataset = MovingMNISTDataset(train_data, args)
    eval_dataset = MovingMNISTDataset(eval_data, args)

    param_dict = ms.load_checkpoint(config.model_dir+'/mae99.ckpt')
    ms.load_param_into_net(model, param_dict)
    # train_epoch(args, epoch, model, train_dataset, criterion, optimizer)
    eval_epoch(args, 0, model, eval_dataset, criterion)
    # model_name = str(epoch)+'.ckpt'
    # ms.save_checkpoint(model, config.model_dir+model_name)


if __name__ == '__main__':
    args = get_args()
    config = Config()
    main(args, config)
