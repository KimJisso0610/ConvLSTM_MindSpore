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
import os
import argparse

__all__ = ['get_args', 'Config']


def get_args():
    parser = argparse.ArgumentParser(description='ConvLSTM')
    parser.add_argument('--device', default='Ascend', help='enables npu')
    parser.add_argument('--device_id', default=0, help='npu id')
    parser.add_argument('--dataset', default='dataset/mnist_test_seq.npy', help='dataset dir')
    parser.add_argument('--display', default=10,
                        help='print acc info every batch you set')
    parser.add_argument('-e', '--epochs', default=1, help='train epoch')
    parser.add_argument('-b', '--batch_size', default=100, help='batch size')
    parser.add_argument('--pretrain', default='',
                        help='use pretrain model, saved in \'./output/model\'')

    parser.add_argument('--num_workers', default=0)
    parser.add_argument('--num_frames_input', default=10)
    parser.add_argument('--num_frames_output', default=10)
    parser.add_argument('--image_size', default=(28, 28))
    parser.add_argument('--input_size', default=(64, 64))
    parser.add_argument('--step_length', default=0.1)
    parser.add_argument('--num_objects', default=[3])
    
    args = parser.parse_args()
    
    return args


class Config:

    def __init__(self):
        # (type, activation, in_ch, out_ch, kernel_size, padding, stride)
        self.encoder = [('conv', 'leaky', 1, 32, 3, 1, 2),
                   ('convlstm', '', 32, 32, 3, 1, 1),
                   ('conv', 'leaky', 32, 64, 3, 1, 2),
                   ('convlstm', '', 64, 64, 3, 1, 1),
                   ('conv', 'leaky', 64, 128, 3, 1, 2),
                   ('convlstm', '', 128, 128, 3, 1, 1)]
        self.decoder = [('deconv', 'leaky', 128, 64, 4, 1, 2),
                   ('convlstm', '', 128, 64, 3, 1, 1),
                   ('deconv', 'leaky', 64, 32, 4, 1, 2),
                   ('convlstm', '', 64, 32, 3, 1, 1),
                   ('deconv', 'leaky', 32, 32, 4, 1, 2),
                   ('convlstm', '', 33, 32, 3, 1, 1),
                   ('conv', 'sigmoid', 32, 1, 1, 0, 1)]

        self.root_dir = os.path.join(os.getcwd(), '.')
        self.output_dir = os.path.join(self.root_dir, 'output')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.model_dir = os.path.join(self.output_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
