# coding = utf-8
import time
import os
import sys
import torch


class Logger(object):

    def __init__(self, stream=sys.stdout):
        output_dir = "log"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        log_name = '{}.log'.format(time.strftime('%Y-%m-%d--%H-%M-%S'))
        filename = os.path.join(output_dir, log_name)

        self.terminal = stream
        self.log = open(filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def save_model(save_dir, model, datasets_name, vgg_choice, epoch, use_random_mask, use_noshare_mask,
               single_task, a, b, train_all_data):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    path = (save_dir + '/' +str(datasets_name) + '_' + str(vgg_choice) + '_' + str(epoch) + '_' +
            str(use_random_mask) + '_' + str(use_noshare_mask) + str(single_task) + '_' +
            str(a) + '_' + str(b) + '_' + str(train_all_data) + '.pth')
    torch.save(model.state_dict(), path)
