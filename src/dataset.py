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


class MovingMNISTDataset:

    def __init__(self, data, args):
        self.input, self.target = self.get_input_and_target(data, args.batch_size)

    def get_input_and_target(self, data, batch_size):
        input_list = []
        target_list = []
        for i in range(int(len(data)/batch_size)):
            data_batch = data[batch_size*i:batch_size*(i+1)]
            data_batch = data_batch.transpose(1, 0, 2, 3, 4)

            input, target = data_batch[:10], data_batch[10:]
            input = input.transpose(1, 0, 2, 3, 4)
            target = target.transpose(1, 0, 2, 3, 4)
            input_list.append(input)
            target_list.append(target)

        return input_list, target_list

    def __len__(self):
        return len(self.input)
