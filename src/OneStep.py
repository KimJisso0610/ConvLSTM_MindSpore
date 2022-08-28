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


import mindspore.ops as ops
import mindspore.nn as nn


class CustomTrainOneStepCell(nn.Cell):
    """Customized training network cell"""

    def __init__(self, network, forward, optimizer):
        """
        Args:
            network: define forward network (includes loss function)
            forward: define forward network (does not include loss function)
            optimizer: define optimizer
        """
        super(CustomTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.forward = forward  # only getting the forecast value of network
        self.network.set_grad()  # create backward network
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters  # parameters to be updated
        self.grad = ops.GradOperation(get_by_list=True)  # backward for getting grad

    def construct(self, *inputs):
        output = self.forward(inputs[0])  # get forecast value
        loss = self.network(*inputs)  # get value of loss function
        grads = self.grad(self.network, self.weights)(*inputs)  # get grad
        self.optimizer(grads)  # update parameters using optimizer
        return output, loss


class CustomEvalOneStepCell(nn.Cell):
    """Customized evaluation network cell"""

    def __init__(self, network, forward):
        """
        Args:
            network: define forward network (includes loss function)
            forward: define forward network (does not include loss function)
        """
        super(CustomEvalOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.forward = forward

    def construct(self, *inputs):
        output = self.forward(inputs[0])  # only getting the forecast value of network
        loss = self.network(*inputs)  # get value of loss function
        return output, loss
