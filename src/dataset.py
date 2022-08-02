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
