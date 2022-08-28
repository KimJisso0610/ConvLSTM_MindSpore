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
