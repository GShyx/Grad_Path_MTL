# coding = utf-8
import argparse
import torch
import torch.nn as nn
import time
import numpy as np
from random import sample


def change_task(m):
    if hasattr(m, 'active_task'):
        m.set_active_task(active_task)


def create_ori_mask(conv_layer_list, task_count, mask_type, a=1, b=1, single_task=0):
    unit_mapping_list=[]
    if mask_type == 'ones_tensor':
        for index, unit_count in enumerate(conv_layer_list):
            unit_mapping_list.append(torch.ones((task_count, unit_count)))
    elif mask_type == 'zeros_numpy':
        for index, unit_count in enumerate(conv_layer_list):
            unit_mapping_list.append(np.zeros([task_count, unit_count]))
    elif mask_type == 'zeros_tensor':
        for index, unit_count in enumerate(conv_layer_list):
            unit_mapping_list.append(torch.zeros([task_count, unit_count]))
    elif mask_type == 'random_tensor':
        for index, unit_count in enumerate(conv_layer_list):
            share_unit = int(a*unit_count)
            nopass_unit = int((1-b)*unit_count)
            unit_mapping = np.zeros([task_count, unit_count])
            row, cols = unit_mapping.shape

            # 叠加similarity
            for t in range(row):
                for unit in range(cols):
                    unit_mapping[t, unit] += similarity_random()

            # 选取值最大部分通过
            unit_mapping_sort = np.argsort(-unit_mapping, axis=1)
            for i in range(row):
                for j in range(cols):
                        unit_mapping[i, unit_mapping_sort[i, j]] = 1 if j < share_unit else np.random.random(1)

            # 在剩余单元中随机选取(1-b)的单元不通过，由小到大排序，随机数在0-1，所以不影响已经确定通过的单元
            unit_mapping_sort = np.argsort(unit_mapping, axis=1)
            for i in range(row):
                for j in range(cols):
                        unit_mapping[i, unit_mapping_sort[i, j]] = 0 if j < nopass_unit else 1
            print(unit_mapping)
            unit_mapping_list.append(torch.Tensor(unit_mapping))
    elif mask_type == 'noshare_tensor':
        for index, unit_count in enumerate(conv_layer_list):
            unit_mapping = np.zeros([task_count, unit_count])
            row, cols = unit_mapping.shape
            for unit in range(cols):
                unit_mapping[single_task, unit] = 1
            print(unit_mapping)
            unit_mapping_list.append(torch.Tensor(unit_mapping))
    # elif mask_type == 'noshare_tensor':
    #     for index, unit_count in enumerate(conv_layer_list):
    #         task_unit_count = math.floor(unit_count/task_count)
    #         unit_mapping = np.zeros([task_count, unit_count])
    #         row, cols = unit_mapping.shape
    #
    #         #任务间使用的单元不共享
    #         current_unit = 0
    #         for t in range(row):
    #             for i in range(task_unit_count):
    #                 unit_mapping[t, current_unit] = 1
    #                 current_unit += 1
    #         print(unit_mapping)
    #         unit_mapping_list.append(torch.Tensor(unit_mapping))
    else:
        print('wrong mask type, mask_type = ones_tensor, zeros_tensor, zeros_numpy, random_tensor or noshare_tensor')
        exit(1)

    return unit_mapping_list


def calculate_similarity(old_param, new_param, save_grad_dict, task_count, layer_index, unit, t, random):
    sum_similarity = 0
    t_grad_change = np.array([0, save_grad_dict[t][layer_index][unit]])
    for t_temp in range(task_count):
        if random:
            sum_similarity += similarity_random()
        else:
            t_temp_grad_change = np.array([0, save_grad_dict[t_temp][layer_index][unit]])
            sum_similarity += similarity(t_grad_change, t_temp_grad_change)
    return sum_similarity


def similarity_random():
    return -1 + 2 * np.random.random(1)


def similarity(grad1, grad2):
    # print(grad1, grad2)
    grad1_norm = np.linalg.norm(grad1)
    grad2_norm = np.linalg.norm(grad2)

    if grad1_norm == 0 or grad2_norm == 0:
        return 0

    dire_similarity = (np.dot(grad1, grad2))/(grad1_norm * grad2_norm)
    norm_similarity = (2 * grad1_norm * grad2_norm) / (grad1_norm * grad1_norm + grad2_norm * grad2_norm)
    grad_similarity = dire_similarity * norm_similarity

    if np.isnan(grad_similarity):
        print('nan error')
        exit(1)

    return grad_similarity

def convert_to_mask(unit_mapping_list, a, b):
    for layer_index in range(len(unit_mapping_list)):
        current_unit_mapping = unit_mapping_list[layer_index]
        row, cols = current_unit_mapping.shape

        share_unit = int(a * cols)
        nopass_unit = int((1 - b) * cols)

        unit_mapping_sort = np.argsort(-current_unit_mapping, axis=1)
        for i in range(row):
            for j in range(cols):
                current_unit_mapping[i, unit_mapping_sort[i, j]] = 1 if j < share_unit else np.random.random(1)

        # 在剩余单元中随机选取(1-b)的单元不通过，由小到大排序，随机数在0-1，所以不影响已经确定通过的单元
        unit_mapping_sort = np.argsort(current_unit_mapping, axis=1)
        for i in range(row):
            for j in range(cols):
                current_unit_mapping[i, unit_mapping_sort[i, j]] = 0 if j < nopass_unit else 1

        unit_mapping_list[layer_index] = torch.Tensor(current_unit_mapping)

    return unit_mapping_list


def pre_train(args, model, task_count, device, train_loader, optimizer, criterion, total_itts, zeros_unit_mapping_list, a, b, random=False, train_all_data=True):
    model.train()
    train_start = time.time()
    batch_total = len(train_loader)
    x = 0

    epoch_loss = 0
    individual_loss = [0 for i in range(task_count)]

    old_param_empty = True

    save_grad_dict = {}
    old_param = {}
    new_param = {}

    for ix in range(task_count):
        save_grad_dict[ix] = {}
        # old_param[ix] = {}
        # new_param[ix] = {}

    for enum_return in enumerate(train_loader):
        if not train_all_data:
            x += 1
            if x > 2:
                break
        batch_idx = enum_return[0]
        print(batch_idx / batch_total)

        data = enum_return[1][0]
        targets = enum_return[1][1]
        # print(data.size())
        # print(data[0])
        data = data.to(device)

        for ix in sample(range(task_count), task_count):
            # print(ix)
            target = targets[ix].to(device)
            # print(data)
            # print(targets[ix])
            global active_task
            active_task = ix

            model = model.apply(change_task)
            out = model(data)

            labels = target

            _, predicted = torch.max(out.data, 1)

            train_loss = criterion(out, labels)
            print('pretrain: 任务:{} loss:{}'.format(ix, train_loss))
            optimizer.zero_grad()
            train_loss.backward()

            #参数是任务间共同更新的
            conv_layer_num = 0
            for name, param in model.features.named_parameters():
                if 'bias' in name:
                    # print(name)
                    # if old_param_empty:
                    #     old_param[ix][conv_layer_num] = param.cpu().detach().clone().numpy()
                    # new_param[ix][conv_layer_num] = param.cpu().detach().clone().numpy()
                    save_grad_dict[ix][conv_layer_num] = param.grad.cpu().clone().numpy()
                    conv_layer_num += 1

            optimizer.step()
        old_param_empty = False
        # print(old_param)
        # 叠加本轮任务组的梯度更新相似度
        for layer_index in range(len(zeros_unit_mapping_list)):
            layer_unit_mapping = zeros_unit_mapping_list[layer_index]
            row, cols = layer_unit_mapping.shape
            for t in range(row):
                for unit in range(cols):
                    layer_unit_mapping[t, unit] += calculate_similarity(old_param, new_param, save_grad_dict, task_count, layer_index, unit, t, random)
            zeros_unit_mapping_list[layer_index] = layer_unit_mapping
        # old_param = new_param

    unit_mapping_list = convert_to_mask(zeros_unit_mapping_list, a, b)

    train_end = time.time()
    print(unit_mapping_list)
    print("pretrain time:", train_end - train_start, "s.")
    return unit_mapping_list


def train(args, model, task_count, device, train_loader, optimizer, criterion, epoch, total_itts, train_all_data=True):
    model.train()
    train_start = time.time()
    batch_total = len(train_loader)
    # correct, positives, true_positives, score_list = initialize_evaluation_vars()

    epoch_loss = 0
    individual_loss = [0 for i in range(task_count)]

    for current_epoch in range(epoch):
        x = 0
        for enum_return in enumerate(train_loader):
            if not train_all_data:
                x += 1
                if x > 2:
                    break
            batch_idx = enum_return[0]
            print(batch_idx / batch_total)

            data = enum_return[1][0]
            targets = enum_return[1][1]
            # print(data.size())
            # print(data[0])
            data = data.to(device)

            for ix in sample(range(task_count), task_count):
            # for ix in range(1):
                # print(ix)
                target = targets[ix].to(device)
                # print(data)
                # print(targets[ix])
                global active_task
                active_task = ix

                model = model.apply(change_task)
                out = model(data)

                labels = target

                _, predicted = torch.max(out.data, 1)

                train_loss = criterion(out, labels)
                print('epoch:{}/{} 任务:{} loss:{}'.format(current_epoch+1, epoch, ix, train_loss))
                # print(current_epoch, ix, train_loss)
                optimizer.zero_grad()

                train_loss.backward()

                # if pretrain_mask:
                #     save_grad(model, batch_time, ix)

                optimizer.step()

    train_end = time.time()
    print("train time:", train_end - train_start, "s.")

    return total_itts


def test(args, model, task_count, device, test_loader, optimizer, criterion, epoch, total_itts, train_all_data=True):
    model.eval()
    batch_total = len(test_loader)
    test_loss = 0
    correct = 0
    total = 0
    FN = 0
    FP = 0
    TP = 0
    TN = 0
    x = 0

    task_result = [{} for i in range(task_count)]
    for i in range(task_count):
        task_result[i]['FN'] = 0
        task_result[i]['FP'] = 0
        task_result[i]['TP'] = 0
        task_result[i]['TN'] = 0

    with torch.no_grad():
        for enum_return in enumerate(test_loader):
            if not train_all_data:
                x += 1
                if x > 2:
                    break
            batch_idx = enum_return[0]
            print(batch_idx / batch_total)

            data = enum_return[1][0]
            targets = enum_return[1][1]
            # print(data.size())
            # print(data[0])
            data = data.to(device)

            for ix in sample(range(task_count), task_count): # 随机抽取一个任务
            # for ix in range(1):
                target = targets[ix].to(device)
                # print(data)
                # print(targets[ix])
                global active_task
                active_task = ix

                model = model.apply(change_task)
                out = model(data)
                # print(out)
                _, predicted = torch.max(out.data, 1)

                # ---------------------------------------
                # labels = target[:, ix]
                labels = target
                # ---------------------------------------
                print(predicted)
                print(labels)
                # test_loss += criterion(out, labels)
                # pred = out.argmax(dim=1, keepdim=True)
                # correct += pred.eq(target.view_as(pred)).sum().item()
                total_batch = labels.size(0)
                correct_batch = (predicted == labels).sum().item()

                zeros_tensor = torch.zeros(total_batch).type(torch.LongTensor).to(device)  # 全0变量
                ones_tensor = torch.ones(total_batch).type(torch.LongTensor).to(device)  # 全1变量

                FN_batch = ((predicted == zeros_tensor) & (labels == ones_tensor)).sum().item()  # 原标签为1，预测为0
                FP_batch = ((predicted == ones_tensor) & (labels == zeros_tensor)).sum().item()
                TP_batch = ((predicted == ones_tensor) & (labels == ones_tensor)).sum().item()
                TN_batch = ((predicted == zeros_tensor) & (labels == zeros_tensor)).sum().item()

                total += labels.size(0)
                correct += (predicted == labels).sum()
                FN += FN_batch
                FP += FP_batch
                TP += TP_batch
                TN += TN_batch
                task_result[ix]['FN'] += FN_batch
                task_result[ix]['FP'] += FP_batch
                task_result[ix]['TP'] += TP_batch
                task_result[ix]['TN'] += TN_batch
                # print('第%d个任务本batch的识别准确率为：%d%%' % (ix, (100 * correct_temp / total_temp)))
                try:
                    print(100 * (correct_batch / total_batch))
                    print ('第{}个任务本batch的识别:\t准确率为：{:.2f}%'
                           .format(ix, 100 * ((TP_batch + TN_batch) / (TP_batch + TN_batch + FP_batch + FN_batch))))
                except:
                    print('TP+TN+FP+FN = 0')
    try:
        # 各任务的测试结果
        for i in range(task_count):
            task_accuracy = 100 * ((task_result[i]['TP']+task_result[i]['TN'])/
                             (task_result[i]['TP']+task_result[i]['TN']+task_result[i]['FP']+task_result[i]['FN']))
            task_precision = 100 * ((task_result[i]['TP'])/
                              (task_result[i]['TP']+task_result[i]['FP']))
            task_recall = 100 * ((task_result[i]['TP'])/
                           (task_result[i]['TP']+task_result[i]['FN']))
            print('第{}个任务:\t准确率为：{:.2f}%\t精确率为{:.2f}%\t召回率为{:.2f}%'
                  .format(i, task_accuracy, task_precision, task_recall))

        # 总测试结果
        print(correct, total)
        # print('总识别准确率为：%d%%' %  (100 * correct / total))
        print(100 * (correct / total))
        print('所有任务总识别:\t准确率为：{:.2f}%\t精确率为{:.2f}%\t召回率为{:.2f}%'
              .format(100 * ((TP + TN) / (TP + TN + FP + FN)),
                      100 * (TP / (TP + FP)),
                      100 * (TP / (TP + FN))))
    except:
        print('TP+TN+FP+FN = 0 or TP+FP = 0 or TP+FN = 0')
        for i in range(task_count):
            print('任务{}:\tTP:{}\tTN:{}\tFP:{}\tFN:{}'
                  .format(i, task_result[i]['TP'], task_result[i]['TN'], task_result[i]['FP'], task_result[i]['FN']))
            print(task_result)
        print('TP:{}\tTN:{}\tFP:{}\tFN:{}'.format(TP, TN, FP, FN))
    return total_itts


def train_single_task(args, model, task_count, device, train_loader, optimizer, criterion, epoch, total_itts, single_task, train_all_data=True):
    model.train()
    train_start = time.time()
    batch_total = len(train_loader)
    # correct, positives, true_positives, score_list = initialize_evaluation_vars()

    epoch_loss = 0
    individual_loss = [0 for i in range(task_count)]

    for current_epoch in range(epoch):
        x = 0
        for enum_return in enumerate(train_loader):
            if not train_all_data:
                x += 1
                if x > 2:
                    break
            batch_idx = enum_return[0]
            print(batch_idx / batch_total)

            data = enum_return[1][0]
            targets = enum_return[1][1]
            # print(data.size())
            # print(data[0])
            data = data.to(device)

            ix = single_task
            # print(ix)
            target = targets[ix].to(device)
            # print(data)
            # print(targets[ix])
            global active_task
            active_task = ix

            model = model.apply(change_task)
            out = model(data)

            labels = target

            _, predicted = torch.max(out.data, 1)

            train_loss = criterion(out, labels)
            print('epoch:{}/{} 任务:{} loss:{}'.format(current_epoch+1, epoch, ix, train_loss))
            # print(current_epoch, ix, train_loss)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

    train_end = time.time()
    print("train time:", train_end - train_start, "s.")

    return total_itts


def test_single_task(args, model, task_count, device, test_loader, optimizer, criterion, epoch, total_itts, single_task, train_all_data=True):
    model.eval()
    batch_total = len(test_loader)
    test_loss = 0
    correct = 0
    total = 0
    FN = 0
    FP = 0
    TP = 0
    TN = 0
    x = 0

    task_result = [{} for i in range(task_count)]
    for i in range(task_count):
        task_result[i]['FN'] = 0
        task_result[i]['FP'] = 0
        task_result[i]['TP'] = 0
        task_result[i]['TN'] = 0

    with torch.no_grad():
        for enum_return in enumerate(test_loader):
            if not train_all_data:
                x += 1
                if x > 2:
                    break
            batch_idx = enum_return[0]
            print(batch_idx / batch_total)

            data = enum_return[1][0]
            targets = enum_return[1][1]
            # print(data.size())
            # print(data[0])
            data = data.to(device)

            ix = single_task
            target = targets[ix].to(device)
            # print(data)
            # print(targets[ix])
            global active_task
            active_task = ix

            model = model.apply(change_task)
            out = model(data)
            # print(out)
            _, predicted = torch.max(out.data, 1)

            labels = target

            print(predicted)
            print(labels)

            total_batch = labels.size(0)
            correct_batch = (predicted == labels).sum().item()

            zeros_tensor = torch.zeros(total_batch).type(torch.LongTensor).to(device)  # 全0变量
            ones_tensor = torch.ones(total_batch).type(torch.LongTensor).to(device)  # 全1变量

            FN_batch = ((predicted == zeros_tensor) & (labels == ones_tensor)).sum().item()  # 原标签为1，预测为0
            FP_batch = ((predicted == ones_tensor) & (labels == zeros_tensor)).sum().item()
            TP_batch = ((predicted == ones_tensor) & (labels == ones_tensor)).sum().item()
            TN_batch = ((predicted == zeros_tensor) & (labels == zeros_tensor)).sum().item()

            total += labels.size(0)
            correct += (predicted == labels).sum()
            FN += FN_batch
            FP += FP_batch
            TP += TP_batch
            TN += TN_batch
            task_result[ix]['FN'] += FN_batch
            task_result[ix]['FP'] += FP_batch
            task_result[ix]['TP'] += TP_batch
            task_result[ix]['TN'] += TN_batch
            # print('第%d个任务本batch的识别准确率为：%d%%' % (ix, (100 * correct_temp / total_temp)))
            try:
                print(100 * (correct_batch / total_batch))
                print ('第{}个任务本batch的识别:\t准确率为：{:.2f}%'
                       .format(ix, 100 * ((TP_batch + TN_batch) / (TP_batch + TN_batch + FP_batch + FN_batch))))
            except:
                print('TP+TN+FP+FN = 0')
    try:
        # 各任务的测试结果
        i = single_task
        task_accuracy = 100 * ((task_result[i]['TP']+task_result[i]['TN'])/
                         (task_result[i]['TP']+task_result[i]['TN']+task_result[i]['FP']+task_result[i]['FN']))
        task_precision = 100 * ((task_result[i]['TP'])/
                          (task_result[i]['TP']+task_result[i]['FP']))
        task_recall = 100 * ((task_result[i]['TP'])/
                       (task_result[i]['TP']+task_result[i]['FN']))
        print('第{}个任务:\t准确率为：{:.2f}%\t精确率为{:.2f}%\t召回率为{:.2f}%'
              .format(i, task_accuracy, task_precision, task_recall))

        # 总测试结果
        print(correct, total)
        # print('总识别准确率为：%d%%' %  (100 * correct / total))
        print(100 * (correct / total))
        print('所有任务总识别:\t准确率为：{:.2f}%\t精确率为{:.2f}%\t召回率为{:.2f}%'
              .format(100 * ((TP + TN) / (TP + TN + FP + FN)),
                      100 * (TP / (TP + FP)),
                      100 * (TP / (TP + FN))))
    except:
        print('TP+TN+FP+FN = 0 or TP+FP = 0 or TP+FN = 0')
        i = single_task
        print('任务{}:\tTP:{}\tTN:{}\tFP:{}\tFN:{}'
              .format(i, task_result[i]['TP'], task_result[i]['TN'], task_result[i]['FP'], task_result[i]['FN']))
        print(task_result)
        print('TP:{}\tTN:{}\tFP:{}\tFN:{}'.format(TP, TN, FP, FN))
    return total_itts


def train_test(args, model, task_count, device, train_loader, test_loader, optimizer, criterion, epoch, total_itts, train_all_data=True):
    model.train()
    train_start = time.time()
    batch_total = len(train_loader)
    # correct, positives, true_positives, score_list = initialize_evaluation_vars()

    epoch_loss = 0
    individual_loss = [0 for i in range(task_count)]

    for current_epoch in range(epoch):
        x = 0
        for enum_return in enumerate(train_loader):
            if not train_all_data:
                x += 1
                if x > 2:
                    break
            batch_idx = enum_return[0]
            print(batch_idx / batch_total)

            data = enum_return[1][0]
            targets = enum_return[1][1]
            # print(data.size())
            # print(data[0])
            data = data.to(device)

            for ix in sample(range(task_count), task_count):
            # for ix in range(1):
                # print(ix)
                target = targets[ix].to(device)
                # print(data)
                # print(targets[ix])
                global active_task
                active_task = ix

                model = model.apply(change_task)
                out = model(data)

                labels = target

                _, predicted = torch.max(out.data, 1)
                # print(out, predicted, labels)

                # ---------------------------------------
                # print(predicted, labels)
                train_loss = criterion(out, labels)
                print('epoch:{}/{} 任务:{} loss:{}'.format(current_epoch+1, epoch, ix, train_loss))
                # print(current_epoch, ix, train_loss)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

        test(args, model, task_count, device, test_loader, optimizer, criterion, epoch, total_itts,
             train_all_data=train_all_data)
        model.train()


    train_end = time.time()
    print("train time:", train_end - train_start, "s.")

    return total_itts
