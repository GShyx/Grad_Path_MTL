# coding = utf-8
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import vgg_module
import sys
from train import create_ori_mask, pre_train, train, test, train_single_task, test_single_task, train_test
from load_data import load_fashionMNIST, load_CIFAR
from save_log import Logger
import os


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Graduation Project - SZ170110132 MaHaixuan')

    parser.add_argument('--datasets', type=str, default='fashionMNIST',
                        help='datasets: fashionMNIST or CIFAR (default: fashionMNIST)')
    parser.add_argument('--vgg', type=str, default='11',
                        help='vgg version: 11 or 16 (default: 11)')
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    #                     help='disables CUDA training')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training and testing (default: 64)')
    parser.add_argument('--share-unit-rate', type=float, default=0.8,
                        help='input share unit rate a(default: 0.8)')
    parser.add_argument('--epoch', type=int, default=5,
                        help='number of epoch to train (default: 5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('-r', action='store_true', default=False,
                        help='use random mask will skip the pretrain (default: False)')
    parser.add_argument('-t', action='store_true', default=False,
                        help='not train all data for test code (default: False)')
    parser.add_argument('-s', action='store_true', default=False,
                        help='train single task (default: False)')
    parser.add_argument('-i', action='store_true', default=False,
                        help='test in train (default: False)')
    parser.add_argument('--which-single-task', type=int, default=0,
                        help='if use-noshare-mask is true, which task do you want to train (default: 0)')

    # parser.add_argument('--log-interval', type=int, default=10, metavar='N',
    #                     help='how many batches to wait before logging training status')

    # parser.add_argument('--save-model', action='store_true', default=False,
    #                     help='For Saving the current Model')


    args = parser.parse_args()

    torch.manual_seed(args.seed)
    # use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = args.batch_size
    a = args.share_unit_rate
    b = a

    use_random_mask = args.r
    use_noshare_mask = args.s
    test_in_train = args.i
    single_task = args.which_single_task

    total_itts = 1
    train_all_data = not args.t
    epoch = args.epoch if train_all_data else 1
    datasets_choice = args.datasets
    vgg_choice = args.vgg

    # batch_size = 64
    # a = 0.8
    # b = a
    #
    # use_random_mask = True
    # use_noshare_mask = False
    # single_task = 0
    #
    # epoch = 5
    # total_itts = 1
    # train_all_data = True
    # datasets_choice = 'fashionMNIST'
    # vgg_choice = '11'

    if datasets_choice == 'fashionMNIST':
        train_loader, test_loader, task_count, channels, datasets_name = load_fashionMNIST(batch_size)
    elif datasets_choice == 'CIFAR':
        train_loader, test_loader, task_count, channels, datasets_name = load_CIFAR(batch_size)
    else:
        print('wrong datasets')
        exit(1)

    if vgg_choice == '11':
        conv_layer_list = [64, 128, 256, 256, 512, 512, 512, 512]
    elif vgg_choice == '16':
        conv_layer_list = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    else:
        print('wrong vgg')
        exit(1)

    # conv_layer_list = [64, 128, 256, 256, 512, 512, 512, 512]
    # conv_layer_list = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]

    ones_unit_mapping_list = create_ori_mask(conv_layer_list, task_count, mask_type='ones_tensor')
    zeros_unit_mapping_list = create_ori_mask(conv_layer_list, task_count, mask_type='zeros_numpy')

    if use_random_mask:
        random_unit_mapping_list = create_ori_mask(conv_layer_list, task_count, mask_type='random_tensor', a=a, b=b)
        unit_mapping_list = random_unit_mapping_list
    elif use_noshare_mask:
        noshare_unit_mapping_list = create_ori_mask(conv_layer_list, task_count, mask_type='noshare_tensor', a=a, b=b, single_task=single_task)
        unit_mapping_list = noshare_unit_mapping_list
    else:
        if vgg_choice == '11':
            pre_m = vgg_module.vgg11(task_count=task_count, unit_mapping_list=ones_unit_mapping_list, channels=channels)
        elif vgg_choice == '16':
            pre_m = vgg_module.vgg16(task_count=task_count, unit_mapping_list=ones_unit_mapping_list, channels=channels)
        else:
            print('wrong vgg')
            exit(1)

        # pre_m = routed_vgg.vgg16(sigma=0, unit_mapping_list=ones_unit_mapping_list, channels=channels)

        pre_model = pre_m.to(device)

        pre_optimizer = optim.SGD(pre_model.parameters(), lr=args.lr, momentum=args.momentum)
        pre_scheduler = optim.lr_scheduler.MultiStepLR(pre_optimizer, milestones=[12, 24], gamma=0.1)
        pre_criterion = nn.CrossEntropyLoss()
        unit_mapping_list = pre_train(args, pre_model, task_count, device, train_loader, pre_optimizer, pre_criterion,
                                      total_itts, zeros_unit_mapping_list, a, b, train_all_data=train_all_data)
    # for epoch in range(1, args.epochs + 1):
    # train(args, model, device, train_loader, optimizer, epoch)

    if vgg_choice == '11':
        m = vgg_module.vgg11(task_count=task_count, unit_mapping_list=unit_mapping_list, channels=channels)
    elif vgg_choice == '16':
        m = vgg_module.vgg16(task_count=task_count, unit_mapping_list=unit_mapping_list, channels=channels)
    else:
        print('wrong vgg')
        exit(1)
    # m = routed_vgg.vgg16(sigma=0, unit_mapping_list=unit_mapping_list, channels=channels)
    model = m.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[12, 24], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    if test_in_train:
        train_test(args, model, task_count, device, train_loader, test_loader, optimizer, criterion, epoch, total_itts,
                   train_all_data=train_all_data)
    elif use_noshare_mask:
        train_single_task(args, model, task_count, device, train_loader, optimizer, criterion, epoch, total_itts,
                          single_task, train_all_data=train_all_data)
        test_single_task(args, model, task_count, device, test_loader, optimizer, criterion, epoch, total_itts,
                         single_task, train_all_data=train_all_data)
    else:
        train(args, model, task_count, device, train_loader, optimizer, criterion, epoch, total_itts,
              train_all_data=train_all_data)
        test(args, model, task_count, device, test_loader, optimizer, criterion, epoch, total_itts,
             train_all_data=train_all_data)

    print('datasets:{}\nvgg:{}\nepoch:{}\nuse random mask:{}\nuse noshare mask:{}\nsingle task:{}\na:{}\nb:{}'
          .format(datasets_name, vgg_choice, epoch, use_random_mask, use_noshare_mask, single_task, a, b))



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    sys.stdout = Logger(sys.stdout)  # 将输出和错误记录到log
    sys.stderr = Logger(sys.stderr)
    main()




