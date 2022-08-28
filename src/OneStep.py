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
    """自定义训练网络"""

    def __init__(self, network, forward, optimizer):
        """入参有两个：训练网络，优化器"""
        super(CustomTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network  # 定义前向网络
        self.forward = forward
        self.network.set_grad()  # 构建反向网络
        self.optimizer = optimizer  # 定义优化器
        self.weights = self.optimizer.parameters  # 待更新参数
        self.grad = ops.GradOperation(get_by_list=True)  # 反向传播获取梯度

    def construct(self, *inputs):
        # print(type(inputs))
        # print(*inputs)
        output = self.forward(inputs[0])
        loss = self.network(*inputs)  # 计算当前输入的损失函数值
        grads = self.grad(self.network, self.weights)(*inputs)  # 进行反向传播，计算梯度
        self.optimizer(grads)  # 使用优化器更新权重参数
        return output, loss


class CustomEvalOneStepCell(nn.Cell):
    """自定义训练网络"""

    def __init__(self, network, forward):
        """入参有两个：训练网络，优化器"""
        super(CustomEvalOneStepCell, self).__init__(auto_prefix=False)
        self.network = network  # 定义前向网络
        self.forward = forward

    def construct(self, *inputs):
        output = self.forward(inputs[0])
        loss = self.network(*inputs)  # 计算当前输入的损失函数值
        return output, loss
