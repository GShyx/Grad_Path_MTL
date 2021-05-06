# coding = utf-8
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from random import sample

class MY_FASHIONMNISTorCIFAR(Dataset):

    def __init__(self, ori_data):
        # self.transform = transform
        self.ori_data = ori_data
        # self.data, self.targets = torch.load(root)

    def __getitem__(self, index):
        img = self.ori_data[index][0]
        target_temp = self.ori_data[index][1]
        target = [0]*10
        target[target_temp] = 1
        # img, target = self.data[index], int(self.targets[index])
        # img = Image.fromarray(img.numpy(), mode='L')

        # if self.transform is not None:
        #     img = self.transform(img)
        # img = transforms.ToTensor()(img)

        # sample = {'img': img, 'target': target}
        sample = (img, target)
        return sample

    def __len__(self):
        return len(self.ori_data)


class MY_CELEBA(Dataset):

    def __init__(self, attr_path, img_path, train, attr_count, transform):
        self.attr_path = attr_path
        self.img_path = img_path
        self.train = train
        self.attr_count = attr_count
        self.transform = transform

        attr_file = open(attr_path, "r")
        self.line_list = attr_file.readlines()

        if self.train:
            self.len = 162770
            self.line_list = self.line_list[2:162772]
            # print(self.line_list[0])
            # print(self.line_list[-1])

        else:
            self.len = 19867
            self.line_list = self.line_list[162772:182639]
            # print(self.line_list[0])
            # print(self.line_list[-1])

        self.attr_cols = sample(range(40), self.attr_count)
        # print(self.attr_cols)

    def __getitem__(self, index):
        img_attr_list = self.line_list[index].split()
        img_name = img_attr_list[0]
        img = Image.open(self.img_path + img_name)
        img = self.transform(img)

        attr_list = img_attr_list[1:]
        target = [0] * self.attr_count
        for index, value in enumerate(self.attr_cols):
            if attr_list[value] == '1':
                target[index] = 1
            else:
                target[index] = 0

        sample = (img, target)
        return sample

    def __len__(self):
        return self.len


def load_fashionMNIST(batch_size):

    fashionmnist_train_data = datasets.FashionMNIST('./fashionmnist_data/', train=True, download=False,
                                              transform=transforms.Compose([
                                                  transforms.Resize(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.2846, ), (0.3521, ))
                                              ]))
    train_data = MY_FASHIONMNISTorCIFAR(ori_data=fashionmnist_train_data)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # fashionmnist_test_data = datasets.FashionMNIST('./fashionmnist_data/', train=False,
    #                                                transform=transforms.Compose([
    #                                                     transforms.ToTensor(),
    #                                                     transforms.Normalize((0.1307,), (0.3081,))
    #                                                 ]))

    fashionmnist_test_data = datasets.FashionMNIST('./fashionmnist_data/', train=False,
                                                   transform=transforms.Compose([
                                                       transforms.Resize(224),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.2846,), (0.3521,))
                                                   ]))
    test_data = MY_FASHIONMNISTorCIFAR(ori_data=fashionmnist_test_data)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    task_count = 10
    channels = 1
    datasets_name = 'fashionMNIST'
    return train_loader, test_loader, task_count, channels, datasets_name


def load_CIFAR(batch_size):

    cifar_train_data = datasets.CIFAR10('./cifar_data/', train=True, download=False,
                                  transform=transforms.Compose([
                                      transforms.Resize(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4818, 0.4432), (0.2451, 0.2418, 0.2598))
                                  ]))
    train_data = MY_FASHIONMNISTorCIFAR(ori_data=cifar_train_data)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    cifar_test_data = datasets.CIFAR10('./cifar_data/', train=False,
                                  transform=transforms.Compose([
                                      transforms.Resize(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4818, 0.4432), (0.2451, 0.2418, 0.2598))
                                  ]))
    test_data = MY_FASHIONMNISTorCIFAR(ori_data=cifar_test_data)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    task_count = 10
    channels = 3
    datasets_name = 'CIFAR10'
    return train_loader, test_loader, task_count, channels, datasets_name



def load_CELEBA(batch_size, task_count=10):

    train_data = MY_CELEBA('./celeba/Anno/list_attr_celeba.txt', './celeba/img_align_celeba/', True, task_count,
                        transform=transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5109, 0.4291, 0.3858), (0.3096, 0.2901, 0.2889))
                        ]))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = MY_CELEBA('./celeba/Anno/list_attr_celeba.txt', './celeba/img_align_celeba/', False, task_count,
                        transform=transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5109, 0.4291, 0.3858), (0.3096, 0.2901, 0.2889))
                        ]))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    task_count = task_count
    channels = 3
    datasets_name = 'CELEBA'

    return train_loader, test_loader, task_count, channels, datasets_name