from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from src.dataset import MovingMNISTDataset
from src.ConvLSTM import ConvLSTM
import mindspore as ms
import mindspore.nn as nn
from src.functions import train_epoch, eval_epoch
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
    
    if args.pretrain != '':
        param_dict = ms.load_checkpoint(config.model_dir+'/'+args.pretrain)
        ms.load_param_into_net(model, param_dict)
    for epoch in range(args.epochs):
        train_epoch(args, epoch, model, train_dataset, criterion, optimizer)
        eval_epoch(args, epoch, model, eval_dataset, criterion)
        model_name = str(epoch)+'.ckpt'
        ms.save_checkpoint(model, config.model_dir+model_name)


if __name__ == '__main__':
    args = get_args()
    config = Config()
    main(args, config)
