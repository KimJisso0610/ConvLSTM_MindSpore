from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import mindspore as ms
import mindspore.nn as nn


class ConvLSTMBlock(nn.Cell):
    def __init__(self, in_channels, num_features, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.num_features = num_features
        self.conv = self._make_layer(in_channels+num_features, num_features*4,
                                     kernel_size, padding, stride)

    def _make_layer(self, in_channels, out_channels, kernel_size, padding, stride):
        seq = nn.SequentialCell()
        return seq(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, pad_mode='same', padding=0, stride=stride),
            nn.BatchNorm2d(out_channels))

    def construct(self, inputs):
        '''

        :param inputs: (B, S, C, H, W)
        :param hidden_state: (hx: (B, S, C, H, W), cx: (B, S, C, H, W))
        :return:
        '''
        outputs = []
        B, S, C, H, W = inputs.shape

        hx = ms.numpy.zeros((B, self.num_features, H, W))
        cx = ms.numpy.zeros((B, self.num_features, H, W))
        for t in range(S):
            op = ms.ops.Concat(1)
            combined = op((inputs[:, t], hx))  # ms.ops.Concat([inputs[:, t],  # (B, C, H, W)
            # hx], dim=1)
            gates = self.conv(combined)
            ms_split = ms.ops.Split(axis=1, output_num=4)
            # print(self.num_features)
            ingate, forgetgate, cellgate, outgate = ms_split(gates)
            # print(ingate.shape)
            sigmoid = ms.ops.Sigmoid()
            ingate = sigmoid(ingate)
            forgetgate = sigmoid(forgetgate)
            outgate = sigmoid(outgate)

            cy = (forgetgate*cx)+(ingate*cellgate)
            tanh = ms.ops.Tanh()
            hy = outgate*tanh(cy)
            outputs.append(hy)
            hx = hy
            cx = cy

        stack = ms.ops.Stack()
        transpose = ms.ops.Transpose()
        return stack(outputs).transpose(1, 0, 2, 3, 4)  # .contiguous()  # (S, B, C, H, W) -> (B, S, C, H, W)


class Encoder(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.layers = []
        for idx, params in enumerate(config.encoder):
            setattr(self, params[0]+'_'+str(idx), self._make_layer(*params))
            self.layers.append(params[0]+'_'+str(idx))

    def _make_layer(self, type, activation, in_ch, out_ch, kernel_size, padding, stride):
        layers = []
        if type == 'conv':
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                                    pad_mode='same', padding=0, stride=stride))
            layers.append(nn.BatchNorm2d(out_ch))
            if activation == 'leaky':
                leaky_relu = nn.LeakyReLU()
                layers.append(leaky_relu())
            elif activation == 'relu':
                relu = nn.ReLU()
                layers.append(relu())
        elif type == 'convlstm':
            layers.append(ConvLSTMBlock(in_ch, out_ch, kernel_size=kernel_size,
                                        padding=0, stride=stride))

        seq = nn.SequentialCell()
        return seq(layers)

    def construct(self, x):
        '''
        :param x: (B, S, C, H, W)
        :return:
        '''
        outputs = [x]
        for layer in self.layers:
            if 'conv_' in layer:
                B, S, C, H, W = x.shape
                x = x.view(B*S, C, H, W)
            x = getattr(self, layer)(x)
            if 'conv_' in layer: x = x.view(B, S, x.shape[1], x.shape[2], x.shape[3])
            if 'convlstm' in layer: outputs.append(x)
        return outputs


class Decoder(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.layers = []
        for idx, params in enumerate(config.decoder):
            setattr(self, params[0]+'_'+str(idx), self._make_layer(*params))
            self.layers.append(params[0]+'_'+str(idx))

    def _make_layer(self, type, activation, in_ch, out_ch, kernel_size, padding, stride):
        layers = []
        if type == 'conv':
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                                    pad_mode='same', padding=0, stride=stride))
            layers.append(nn.BatchNorm2d(out_ch))
            if activation == 'leaky':
                leaky_relu = nn.LeakyReLU()
                layers.append(leaky_relu())
            elif activation == 'relu':
                relu = nn.ReLU()
                layers.append(relu())
            elif activation == 'sigmoid':
                sigmoid = nn.Sigmoid()
                layers.append(sigmoid())
        elif type == 'convlstm':
            layers.append(ConvLSTMBlock(in_ch, out_ch, kernel_size=kernel_size,
                                        padding=1, stride=stride))
        elif type == 'deconv':
            layers.append(
                nn.Conv2dTranspose(in_ch, out_ch, kernel_size=kernel_size,
                                   pad_mode='same', padding=0, stride=stride, has_bias=True))
            layers.append(nn.BatchNorm2d(out_ch))
            if activation == 'leaky':
                layers.append(nn.LeakyReLU())
            elif activation == 'relu':
                layers.append(nn.ReLU())
        seq = nn.SequentialCell()
        return seq(layers)

    def construct(self, encoder_outputs):
        '''
        :param x: (B, S, C, H, W)
        :return:
        '''
        idx = len(encoder_outputs)-1
        for layer in self.layers:
            if 'conv_' in layer or 'deconv_' in layer:
                x = encoder_outputs[idx]
                B, S, C, H, W = x.shape
                x = x.view(B*S, C, H, W)
                x = getattr(self, layer)(x)
                x = x.view(B, S, x.shape[1], x.shape[2], x.shape[3])
            elif 'convlstm' in layer:
                idx -= 1
                op = ms.ops.Concat(2)
                x = op((encoder_outputs[idx], x))
                x = getattr(self, layer)(x)
                encoder_outputs[idx] = x
        return x


class ConvLSTM(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def construct(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    from thop import profile
    from src.config import Config

    config = Config()

    model = ConvLSTM(config)
    flops, params = profile(model, inputs=(ms.Tensor(0.0, shape=(4, 10, 1, 64, 64))), )
