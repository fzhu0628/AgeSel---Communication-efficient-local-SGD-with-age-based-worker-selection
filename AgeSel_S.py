# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 14:40:57 2022

@author: ChandlerZhu
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.utils.data.sampler as sampler
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import copy
from torch import Tensor
transform=torchvision.transforms.Compose(
[transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
n_epochs = 3
num_straggler = 4
batch_size_train = 20
batch_size_test = 400
learning_rate = 0.3
momentum = 0
log_interval = 10
M = 20
typeMNIST = 'balanced'
length_out =47
miu = 1e4
scale = 2
miu_e = 1e-4
# transform=torchvision.transforms.Compose(
# [transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
dataset_workers = []
sampler_workers = []
loader_workers = []
'''dataset_global = torchvision.datasets.MNIST('./dataset/', train=True, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ]))'''
dataset_global = torchvision.datasets.EMNIST(
                            root='./data',
                            train=True,
                            transform=torchvision.transforms.ToTensor(),
                            download = False,
                            split = typeMNIST
                            )
# dataset_global = torchvision.datasets.CIFAR10(
#                             root='./data',
#                             train=True,
#                             transform=transforms.ToTensor(),
#                             download = False,
#                             )
length = len(dataset_global)
index = dataset_global.targets.argsort()
# index = np.array(dataset_global.targets).argsort()
# index = torch.randperm(length)
# index = np.array(torch.randperm(length))
def rand_pick(seq, probabilities):
    x = np.random.uniform(0,1)
    cumprob = 0.0
    for item, item_pro in zip(seq, probabilities):
        cumprob += item_pro
        if x < cumprob:
            break
    return item
def poc(seq, prob, S):
    out = []
    while True:
        itemm = rand_pick(seq, prob)
        out.append(itemm)
        ind = seq.index(itemm)
        del seq[ind]
        del prob[ind]
        prob = (np.array(prob)*(1/sum(prob))).tolist()
        if len(out) == S:
            break
    return out
# p = [10,10,10,10,30,30,30,30,40,40,40,40,20,20,20,20,10,10,10,10]
p = [50,50,50,50,50,50,50,50,50,50,10,10,10,10,10,10,10,10,10,10]
p = (p/sum(p)).tolist()
for i in range(M):
    '''dataset_workers.append(torchvision.datasets.MNIST('./dataset/', train=True, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])))'''
    dataset_workers.append(torchvision.datasets.EMNIST(
                            root='./data',
                            train=True,
                            transform=torchvision.transforms.ToTensor(),
                            download = False,
                            split = typeMNIST
                            ))
    # dataset_workers.append(torchvision.datasets.CIFAR10(
    #                         root='./data',
    #                         train=True,
    #                         transform=transform,
    #                         download = False,
    #                         ))
    '''dataset_workers.append(torchvision.datasets.MNIST('./dataset/', train=True, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ])))'''
    
    # dataset_workers[i].data, dataset_workers[i].targets = dataset_workers[i].data[index], np.array(dataset_workers[i].targets)[index].tolist()
    dataset_workers[i].data, dataset_workers[i].targets = dataset_workers[i].data[index], dataset_workers[i].targets[index]
    if i == 0:
        dataset_workers[i].data, dataset_workers[i].targets = dataset_workers[i].data[0 : int(length*p[0])]\
        , dataset_workers[i].targets[0 : int(length*p[0])]
        flag = int(length*p[0])
    else:
        dataset_workers[i].data, dataset_workers[i].targets = dataset_workers[i].data[flag + 1 : flag + int(length*p[i])]\
        , dataset_workers[i].targets[flag + 1 : flag + int(length*p[i])]
        flag = flag + int(length*p[i])
    # dataset_workers[i].data, dataset_workers[i].targets = dataset_workers[i].data[int(length / M * i) : int(length / M * (i + 1))]\
    # , dataset_workers[i].targets[int(length / M * i) : int(length / M * (i + 1))]

    sampler_workers.append(sampler.BatchSampler(sampler.RandomSampler(data_source=dataset_workers[i], replacement=True), batch_size=batch_size_train, drop_last=False))
    loader_workers.append(torch.utils.data.DataLoader(dataset_workers[i],batch_sampler=sampler_workers[i], shuffle=False))

'''test_dataset = torchvision.datasets.MNIST('./dataset/', train=False, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))'''
test_dataset = torchvision.datasets.EMNIST(
                            root='./data',
                            train=False,
                            transform=torchvision.transforms.ToTensor(),
                            download = False,
                            split = typeMNIST
                            )
# test_dataset = torchvision.datasets.CIFAR10(
#                             root='./data',
#                             train=False,
#                             transform=transform,
#                             download = False,
#                             )
'''test_dataset = torchvision.datasets.MNIST('./dataset/', train=False, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ]))'''

#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
test_sampler = sampler.BatchSampler(sampler.RandomSampler(data_source=test_dataset, replacement=False), batch_size=batch_size_test, drop_last=False)
test_loader = torch.utils.data.DataLoader(test_dataset
  ,batch_sampler=test_sampler)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 500) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(500, length_out)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return F.log_softmax(out)

network1 = Net()
S = 1
K = 5

acc = 80
A = 4
#%% ADA_Wireless_modified1
# iters = iters * 3
# iters3 = 800
c = 1.79
t = -1
S = 1
D = S
iters3 = 60000
t_ADA_Wireless_modified1 = [0]*(iters3+1)
comm_ADA_Wireless_modified1 = [0]*(iters3+1)
train_losses_ADA_Wireless_modified1 = []
test_losses_ADA_Wireless_modified1 = []
network = copy.deepcopy(network1)
# optimizer = optim.Adam(network.parameters(), lr=0.005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
optimizer = optim.SGD(network.parameters(), lr=1, momentum=momentum)
network.train()
optimizer_worker = []
for m in range(M):
    optimizer_worker.append(0)
grad_new = {}
grad_test = np.zeros(M).tolist()
age = [0] * M
diff = {}
for i in range(iters3):
    
    if i == 0:
        for batch_idx, (data, target) in enumerate(test_loader):
            output = network(data.reshape(batch_size_test, 784))
            # output = network(data)
            loss = F.nll_loss(output, target)
            break
        loss_orig = loss.item()
    if i % 10 == 0:
        network.eval()
        test_loss_ADA_Wireless_modified1 = 0
        correct_ADA_Wireless_modified1 = 0
        with torch.no_grad():
          for data, target in test_loader:
            output = network(data.reshape(batch_size_test, 784))
            # output = network(data)
            test_loss_ADA_Wireless_modified1 += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct_ADA_Wireless_modified1 += pred.eq(target.data.view_as(pred)).sum()
        test_loss_ADA_Wireless_modified1 /= len(test_loader.dataset)
        test_losses_ADA_Wireless_modified1.append(100. * correct_ADA_Wireless_modified1 / len(test_loader.dataset))

        # selection_temp = np.random.choice(range(M), M, replace=False).tolist()
    selection_temp = np.array(p).argsort()[::-1].tolist()
    age_structure = np.zeros(M, dtype='int32, float64')
    for m in range(M):
        age_structure[m] = tuple((age[m], grad_test[m]))
    age_temp = np.argsort(age_structure, order=['f0', 'f1']).tolist()[::-1]
    selection = set([])
    len_A = 0
    for m in range(M):
        if age[m] >= A:
            len_A += 1
    if len_A >= D:
        for m in age_temp:
            if age[m] >= A:
                selection.add(m)
                age[m] = 0
            if len(selection) == D:
                break
    else:
        p_temp = copy.deepcopy(p)
        for m in age_temp:
            if age[m] >= A:
                selection.add(m)
                age[m] = 0
        if len(selection)>0:
            for m in selection:
                p_temp.remove(p[m])
        noselectiontemp = list(set(list(range(M))).difference(set(selection)))
        sel = poc(noselectiontemp, (np.array(p_temp)*M/len(noselectiontemp)).tolist(), S-len(selection))
        # sel = np.random.choice(noselectiontemp, S-len(selection), replace=False)
        for se in sel:
            selection.add(se)
            # for m in range(M):
            #     selection.add(selection_temp[0])
            #     selection_temp.pop(0)
            #     if len(selection) == D:
            #         break
    selection = list(selection)
    noselection = list(set(list(range(M))).difference(set(selection)))
    for m in noselection:
        age[m] += 1
    grad_workers= {}
    for m in selection:
        network_worker = copy.deepcopy(network)
        network_worker.train()
        optimizer_worker[m] = optim.SGD(network_worker.parameters(), lr=0.1, momentum=momentum) 
        rounds = 0
        # local_rounds = int(np.max(time_workers) / time_workers[flag])
        for aa in range(10000):
            for batch_idx, (data, target) in enumerate(loader_workers[m]):
                optimizer_worker[m].zero_grad()
                output = network_worker(data.reshape(-1, 784))
                # output = network_worker(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer_worker[m].step()
                if i <= t:
                    grad_workers[m]=optimizer_worker[m].param_groups[0]['params'] 
                    break
                if batch_idx == K - 1:
                    grad_workers[m]=optimizer_worker[m].param_groups[0]['params']
                    break
            if batch_idx == K - 1 or i <= t:
                grad_new[m] = grad_workers[m]
                break
    if i >= 0 :
        for m in selection:
            for l in range(4):
                diff[l] = (grad_new[m][l] - optimizer.param_groups[0]['params'][l]).detach().numpy().reshape(-1)
            grad_test[m] = np.linalg.norm(np.concatenate((diff[0],diff[1],diff[2],diff[3])))
    temp = [0]*4
    tt = 0
    for m in selection:
        tt += p[m]
    for l in range(4):
        for m in selection:
            temp[l] += (grad_new[m][l]- optimizer.param_groups[0]['params'][l])*p[m]/tt
        # temp[l] /= len(selection)
    for batch_idx, (data, target) in enumerate(test_loader):
        optimizer.zero_grad()
        for l in range(4):
            optimizer.param_groups[0]['params'][l].grad = -temp[l]
        optimizer.step()
        output = network(data.reshape(batch_size_test, 784))
        # output = network(data)
        loss = F.nll_loss(output, target)
        break
    # optimizer.param_groups[0]['lr'] *= 0.99
    print(loss.item(), "ADA_Wireless_modified1", i)
    print(selection)
    train_losses_ADA_Wireless_modified1.append(loss.item())
    comm_ADA_Wireless_modified1[i] += len(selection)
    comm_ADA_Wireless_modified1[i+1] = comm_ADA_Wireless_modified1[i]   
    if test_losses_ADA_Wireless_modified1[-1] >= acc:
        break
#%% ADA_Wireless_modified2
# iters = iters * 3
# iters3 = 800
c = 1.79
t = -1
S = 5
D = S
iters3 = 60000
t_ADA_Wireless_modified2 = [0]*(iters3+1)
comm_ADA_Wireless_modified2 = [0]*(iters3+1)
train_losses_ADA_Wireless_modified2 = []
test_losses_ADA_Wireless_modified2 = []
network = copy.deepcopy(network1)
# optimizer = optim.Adam(network.parameters(), lr=0.005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
optimizer = optim.SGD(network.parameters(), lr=1, momentum=momentum)
network.train()
optimizer_worker = []
for m in range(M):
    optimizer_worker.append(0)
grad_new = {}
grad_test = np.zeros(M).tolist()
age = [0] * M
diff = {}
for i in range(iters3):
    
    if i == 0:
        for batch_idx, (data, target) in enumerate(test_loader):
            output = network(data.reshape(batch_size_test, 784))
            # output = network(data)
            loss = F.nll_loss(output, target)
            break
        loss_orig = loss.item()
    if i % 10 == 0:
        network.eval()
        test_loss_ADA_Wireless_modified2 = 0
        correct_ADA_Wireless_modified2 = 0
        with torch.no_grad():
          for data, target in test_loader:
            output = network(data.reshape(batch_size_test, 784))
            # output = network(data)
            test_loss_ADA_Wireless_modified2 += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct_ADA_Wireless_modified2 += pred.eq(target.data.view_as(pred)).sum()
        test_loss_ADA_Wireless_modified2 /= len(test_loader.dataset)
        test_losses_ADA_Wireless_modified2.append(100. * correct_ADA_Wireless_modified2 / len(test_loader.dataset))

        # selection_temp = np.random.choice(range(M), M, replace=False).tolist()
    selection_temp = np.array(p).argsort()[::-1].tolist()
    age_structure = np.zeros(M, dtype='int32, float64')
    for m in range(M):
        age_structure[m] = tuple((age[m], grad_test[m]))
    age_temp = np.argsort(age_structure, order=['f0', 'f1']).tolist()[::-1]
    selection = set([])
    len_A = 0
    for m in range(M):
        if age[m] >= A:
            len_A += 1
    if len_A >= D:
        for m in age_temp:
            if age[m] >= A:
                selection.add(m)
                age[m] = 0
            if len(selection) == D:
                break
    else:
        p_temp = copy.deepcopy(p)
        for m in age_temp:
            if age[m] >= A:
                selection.add(m)
                age[m] = 0
        if len(selection)>0:
            for m in selection:
                p_temp.remove(p[m])
        noselectiontemp = list(set(list(range(M))).difference(set(selection)))
        sel = poc(noselectiontemp, (np.array(p_temp)*M/len(noselectiontemp)).tolist(), S-len(selection))
        # sel = np.random.choice(noselectiontemp, S-len(selection), replace=False)
        for se in sel:
            selection.add(se)
            # for m in range(M):
            #     selection.add(selection_temp[0])
            #     selection_temp.pop(0)
            #     if len(selection) == D:
            #         break
    selection = list(selection)
    noselection = list(set(list(range(M))).difference(set(selection)))
    for m in noselection:
        age[m] += 1
    grad_workers= {}
    for m in selection:
        network_worker = copy.deepcopy(network)
        network_worker.train()
        optimizer_worker[m] = optim.SGD(network_worker.parameters(), lr=0.1, momentum=momentum) 
        rounds = 0
        # local_rounds = int(np.max(time_workers) / time_workers[flag])
        for aa in range(10000):
            for batch_idx, (data, target) in enumerate(loader_workers[m]):
                optimizer_worker[m].zero_grad()
                output = network_worker(data.reshape(-1, 784))
                # output = network_worker(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer_worker[m].step()
                if i <= t:
                    grad_workers[m]=optimizer_worker[m].param_groups[0]['params'] 
                    break
                if batch_idx == K - 1:
                    grad_workers[m]=optimizer_worker[m].param_groups[0]['params']
                    break
            if batch_idx == K - 1 or i <= t:
                grad_new[m] = grad_workers[m]
                break
    if i >= 0 :
        for m in selection:
            for l in range(4):
                diff[l] = (grad_new[m][l] - optimizer.param_groups[0]['params'][l]).detach().numpy().reshape(-1)
            grad_test[m] = np.linalg.norm(np.concatenate((diff[0],diff[1],diff[2],diff[3])))
    temp = [0]*4
    tt = 0
    for m in selection:
        tt += p[m]
    for l in range(4):
        for m in selection:
            temp[l] += (grad_new[m][l]- optimizer.param_groups[0]['params'][l])*p[m]/tt
        # temp[l] /= len(selection)
    for batch_idx, (data, target) in enumerate(test_loader):
        optimizer.zero_grad()
        for l in range(4):
            optimizer.param_groups[0]['params'][l].grad = -temp[l]
        optimizer.step()
        output = network(data.reshape(batch_size_test, 784))
        # output = network(data)
        loss = F.nll_loss(output, target)
        break
    # optimizer.param_groups[0]['lr'] *= 0.99
    print(loss.item(), "ADA_Wireless_modified2", i)
    print(selection)
    train_losses_ADA_Wireless_modified2.append(loss.item())
    comm_ADA_Wireless_modified2[i] += len(selection)
    comm_ADA_Wireless_modified2[i+1] = comm_ADA_Wireless_modified2[i]   
    if test_losses_ADA_Wireless_modified2[-1] >= acc:
        break
#%% ADA_Wireless_modified3
# iters = iters * 3
# iters3 = 800
c = 1.79
t = -1
S = 8
D = S
iters3 = 60000
t_ADA_Wireless_modified3 = [0]*(iters3+1)
comm_ADA_Wireless_modified3 = [0]*(iters3+1)
train_losses_ADA_Wireless_modified3 = []
test_losses_ADA_Wireless_modified3 = []
network = copy.deepcopy(network1)
# optimizer = optim.Adam(network.parameters(), lr=0.005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
optimizer = optim.SGD(network.parameters(), lr=1, momentum=momentum)
network.train()
optimizer_worker = []
for m in range(M):
    optimizer_worker.append(0)
grad_new = {}
grad_test = np.zeros(M).tolist()
age = [0] * M
diff = {}
for i in range(iters3):
    
    if i == 0:
        for batch_idx, (data, target) in enumerate(test_loader):
            output = network(data.reshape(batch_size_test, 784))
            # output = network(data)
            loss = F.nll_loss(output, target)
            break
        loss_orig = loss.item()
    if i % 10 == 0:
        network.eval()
        test_loss_ADA_Wireless_modified3 = 0
        correct_ADA_Wireless_modified3 = 0
        with torch.no_grad():
          for data, target in test_loader:
            output = network(data.reshape(batch_size_test, 784))
            # output = network(data)
            test_loss_ADA_Wireless_modified3 += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct_ADA_Wireless_modified3 += pred.eq(target.data.view_as(pred)).sum()
        test_loss_ADA_Wireless_modified3 /= len(test_loader.dataset)
        test_losses_ADA_Wireless_modified3.append(100. * correct_ADA_Wireless_modified3 / len(test_loader.dataset))

        # selection_temp = np.random.choice(range(M), M, replace=False).tolist()
    selection_temp = np.array(p).argsort()[::-1].tolist()
    age_structure = np.zeros(M, dtype='int32, float64')
    for m in range(M):
        age_structure[m] = tuple((age[m], grad_test[m]))
    age_temp = np.argsort(age_structure, order=['f0', 'f1']).tolist()[::-1]
    selection = set([])
    len_A = 0
    for m in range(M):
        if age[m] >= A:
            len_A += 1
    if len_A >= D:
        for m in age_temp:
            if age[m] >= A:
                selection.add(m)
                age[m] = 0
            if len(selection) == D:
                break
    else:
        p_temp = copy.deepcopy(p)
        for m in age_temp:
            if age[m] >= A:
                selection.add(m)
                age[m] = 0
        if len(selection)>0:
            for m in selection:
                p_temp.remove(p[m])
        noselectiontemp = list(set(list(range(M))).difference(set(selection)))
        sel = poc(noselectiontemp, (np.array(p_temp)*M/len(noselectiontemp)).tolist(), S-len(selection))
        # sel = np.random.choice(noselectiontemp, S-len(selection), replace=False)
        for se in sel:
            selection.add(se)
            # for m in range(M):
            #     selection.add(selection_temp[0])
            #     selection_temp.pop(0)
            #     if len(selection) == D:
            #         break
    selection = list(selection)
    noselection = list(set(list(range(M))).difference(set(selection)))
    for m in noselection:
        age[m] += 1
    grad_workers= {}
    for m in selection:
        network_worker = copy.deepcopy(network)
        network_worker.train()
        optimizer_worker[m] = optim.SGD(network_worker.parameters(), lr=0.1, momentum=momentum) 
        rounds = 0
        # local_rounds = int(np.max(time_workers) / time_workers[flag])
        for aa in range(10000):
            for batch_idx, (data, target) in enumerate(loader_workers[m]):
                optimizer_worker[m].zero_grad()
                output = network_worker(data.reshape(-1, 784))
                # output = network_worker(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer_worker[m].step()
                if i <= t:
                    grad_workers[m]=optimizer_worker[m].param_groups[0]['params'] 
                    break
                if batch_idx == K - 1:
                    grad_workers[m]=optimizer_worker[m].param_groups[0]['params']
                    break
            if batch_idx == K - 1 or i <= t:
                grad_new[m] = grad_workers[m]
                break
    if i >= 0 :
        for m in selection:
            for l in range(4):
                diff[l] = (grad_new[m][l] - optimizer.param_groups[0]['params'][l]).detach().numpy().reshape(-1)
            grad_test[m] = np.linalg.norm(np.concatenate((diff[0],diff[1],diff[2],diff[3])))
    temp = [0]*4
    tt = 0
    for m in selection:
        tt += p[m]
    for l in range(4):
        for m in selection:
            temp[l] += (grad_new[m][l]- optimizer.param_groups[0]['params'][l])*p[m]/tt
        # temp[l] /= len(selection)
    for batch_idx, (data, target) in enumerate(test_loader):
        optimizer.zero_grad()
        for l in range(4):
            optimizer.param_groups[0]['params'][l].grad = -temp[l]
        optimizer.step()
        output = network(data.reshape(batch_size_test, 784))
        # output = network(data)
        loss = F.nll_loss(output, target)
        break
    # optimizer.param_groups[0]['lr'] *= 0.99
    print(loss.item(), "ADA_Wireless_modified3", i)
    print(selection)
    train_losses_ADA_Wireless_modified3.append(loss.item())
    comm_ADA_Wireless_modified3[i] += len(selection)
    comm_ADA_Wireless_modified3[i+1] = comm_ADA_Wireless_modified3[i]   
    if test_losses_ADA_Wireless_modified3[-1] >= acc:
        break
#%% ADA_Wireless_modified4
# iters = iters * 3
# iters3 = 800
c = 1.79
t = -1
S = 15
D = S
iters3 = 60000
t_ADA_Wireless_modified4 = [0]*(iters3+1)
comm_ADA_Wireless_modified4 = [0]*(iters3+1)
train_losses_ADA_Wireless_modified4 = []
test_losses_ADA_Wireless_modified4 = []
network = copy.deepcopy(network1)
# optimizer = optim.Adam(network.parameters(), lr=0.005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
optimizer = optim.SGD(network.parameters(), lr=1, momentum=momentum)
network.train()
optimizer_worker = []
for m in range(M):
    optimizer_worker.append(0)
grad_new = {}
grad_test = np.zeros(M).tolist()
age = [0] * M
diff = {}
for i in range(iters3):
    
    if i == 0:
        for batch_idx, (data, target) in enumerate(test_loader):
            output = network(data.reshape(batch_size_test, 784))
            # output = network(data)
            loss = F.nll_loss(output, target)
            break
        loss_orig = loss.item()
    if i % 10 == 0:
        network.eval()
        test_loss_ADA_Wireless_modified4 = 0
        correct_ADA_Wireless_modified4 = 0
        with torch.no_grad():
          for data, target in test_loader:
            output = network(data.reshape(batch_size_test, 784))
            # output = network(data)
            test_loss_ADA_Wireless_modified4 += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct_ADA_Wireless_modified4 += pred.eq(target.data.view_as(pred)).sum()
        test_loss_ADA_Wireless_modified4 /= len(test_loader.dataset)
        test_losses_ADA_Wireless_modified4.append(100. * correct_ADA_Wireless_modified4 / len(test_loader.dataset))

        # selection_temp = np.random.choice(range(M), M, replace=False).tolist()
    selection_temp = np.array(p).argsort()[::-1].tolist()
    age_structure = np.zeros(M, dtype='int32, float64')
    for m in range(M):
        age_structure[m] = tuple((age[m], grad_test[m]))
    age_temp = np.argsort(age_structure, order=['f0', 'f1']).tolist()[::-1]
    selection = set([])
    len_A = 0
    for m in range(M):
        if age[m] >= A:
            len_A += 1
    if len_A >= D:
        for m in age_temp:
            if age[m] >= A:
                selection.add(m)
                age[m] = 0
            if len(selection) == D:
                break
    else:
        p_temp = copy.deepcopy(p)
        for m in age_temp:
            if age[m] >= A:
                selection.add(m)
                age[m] = 0
        if len(selection)>0:
            for m in selection:
                p_temp.remove(p[m])
        noselectiontemp = list(set(list(range(M))).difference(set(selection)))
        sel = poc(noselectiontemp, (np.array(p_temp)*M/len(noselectiontemp)).tolist(), S-len(selection))
        # sel = np.random.choice(noselectiontemp, S-len(selection), replace=False)
        for se in sel:
            selection.add(se)
            # for m in range(M):
            #     selection.add(selection_temp[0])
            #     selection_temp.pop(0)
            #     if len(selection) == D:
            #         break
    selection = list(selection)
    noselection = list(set(list(range(M))).difference(set(selection)))
    for m in noselection:
        age[m] += 1
    grad_workers= {}
    for m in selection:
        network_worker = copy.deepcopy(network)
        network_worker.train()
        optimizer_worker[m] = optim.SGD(network_worker.parameters(), lr=0.1, momentum=momentum) 
        rounds = 0
        # local_rounds = int(np.max(time_workers) / time_workers[flag])
        for aa in range(10000):
            for batch_idx, (data, target) in enumerate(loader_workers[m]):
                optimizer_worker[m].zero_grad()
                output = network_worker(data.reshape(-1, 784))
                # output = network_worker(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer_worker[m].step()
                if i <= t:
                    grad_workers[m]=optimizer_worker[m].param_groups[0]['params'] 
                    break
                if batch_idx == K - 1:
                    grad_workers[m]=optimizer_worker[m].param_groups[0]['params']
                    break
            if batch_idx == K - 1 or i <= t:
                grad_new[m] = grad_workers[m]
                break
    if i >= 0 :
        for m in selection:
            for l in range(4):
                diff[l] = (grad_new[m][l] - optimizer.param_groups[0]['params'][l]).detach().numpy().reshape(-1)
            grad_test[m] = np.linalg.norm(np.concatenate((diff[0],diff[1],diff[2],diff[3])))
    temp = [0]*4
    tt = 0
    for m in selection:
        tt += p[m]
    for l in range(4):
        for m in selection:
            temp[l] += (grad_new[m][l]- optimizer.param_groups[0]['params'][l])*p[m]/tt
        # temp[l] /= len(selection)
    for batch_idx, (data, target) in enumerate(test_loader):
        optimizer.zero_grad()
        for l in range(4):
            optimizer.param_groups[0]['params'][l].grad = -temp[l]
        optimizer.step()
        output = network(data.reshape(batch_size_test, 784))
        # output = network(data)
        loss = F.nll_loss(output, target)
        break
    # optimizer.param_groups[0]['lr'] *= 0.99
    print(loss.item(), "ADA_Wireless_modified4", i)
    print(selection)
    train_losses_ADA_Wireless_modified4.append(loss.item())
    comm_ADA_Wireless_modified4[i] += len(selection)
    comm_ADA_Wireless_modified4[i+1] = comm_ADA_Wireless_modified4[i]   
    if test_losses_ADA_Wireless_modified4[-1] >= acc:
        break
#%% ADA_Wireless_modified5
# iters = iters * 3
# iters3 = 800
c = 1.79
t = -1
S = 20
D = S
iters3 = 60000
t_ADA_Wireless_modified5 = [0]*(iters3+1)
comm_ADA_Wireless_modified5 = [0]*(iters3+1)
train_losses_ADA_Wireless_modified5 = []
test_losses_ADA_Wireless_modified5 = []
network = copy.deepcopy(network1)
# optimizer = optim.Adam(network.parameters(), lr=0.005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
optimizer = optim.SGD(network.parameters(), lr=1, momentum=momentum)
network.train()
optimizer_worker = []
for m in range(M):
    optimizer_worker.append(0)
grad_new = {}
grad_test = np.zeros(M).tolist()
age = [0] * M
diff = {}
for i in range(iters3):
    
    if i == 0:
        for batch_idx, (data, target) in enumerate(test_loader):
            output = network(data.reshape(batch_size_test, 784))
            # output = network(data)
            loss = F.nll_loss(output, target)
            break
        loss_orig = loss.item()
    if i % 10 == 0:
        network.eval()
        test_loss_ADA_Wireless_modified5 = 0
        correct_ADA_Wireless_modified5 = 0
        with torch.no_grad():
          for data, target in test_loader:
            output = network(data.reshape(batch_size_test, 784))
            # output = network(data)
            test_loss_ADA_Wireless_modified5 += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct_ADA_Wireless_modified5 += pred.eq(target.data.view_as(pred)).sum()
        test_loss_ADA_Wireless_modified5 /= len(test_loader.dataset)
        test_losses_ADA_Wireless_modified5.append(100. * correct_ADA_Wireless_modified5 / len(test_loader.dataset))

        # selection_temp = np.random.choice(range(M), M, replace=False).tolist()
    selection_temp = np.array(p).argsort()[::-1].tolist()
    age_structure = np.zeros(M, dtype='int32, float64')
    for m in range(M):
        age_structure[m] = tuple((age[m], grad_test[m]))
    age_temp = np.argsort(age_structure, order=['f0', 'f1']).tolist()[::-1]
    selection = set([])
    len_A = 0
    for m in range(M):
        if age[m] >= A:
            len_A += 1
    if len_A >= D:
        for m in age_temp:
            if age[m] >= A:
                selection.add(m)
                age[m] = 0
            if len(selection) == D:
                break
    else:
        p_temp = copy.deepcopy(p)
        for m in age_temp:
            if age[m] >= A:
                selection.add(m)
                age[m] = 0
        if len(selection)>0:
            for m in selection:
                p_temp.remove(p[m])
        noselectiontemp = list(set(list(range(M))).difference(set(selection)))
        sel = poc(noselectiontemp, (np.array(p_temp)*M/len(noselectiontemp)).tolist(), S-len(selection))
        # sel = np.random.choice(noselectiontemp, S-len(selection), replace=False)
        for se in sel:
            selection.add(se)
            # for m in range(M):
            #     selection.add(selection_temp[0])
            #     selection_temp.pop(0)
            #     if len(selection) == D:
            #         break
    selection = list(selection)
    noselection = list(set(list(range(M))).difference(set(selection)))
    for m in noselection:
        age[m] += 1
    grad_workers= {}
    for m in selection:
        network_worker = copy.deepcopy(network)
        network_worker.train()
        optimizer_worker[m] = optim.SGD(network_worker.parameters(), lr=0.1, momentum=momentum) 
        rounds = 0
        # local_rounds = int(np.max(time_workers) / time_workers[flag])
        for aa in range(10000):
            for batch_idx, (data, target) in enumerate(loader_workers[m]):
                optimizer_worker[m].zero_grad()
                output = network_worker(data.reshape(-1, 784))
                # output = network_worker(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer_worker[m].step()
                if i <= t:
                    grad_workers[m]=optimizer_worker[m].param_groups[0]['params'] 
                    break
                if batch_idx == K - 1:
                    grad_workers[m]=optimizer_worker[m].param_groups[0]['params']
                    break
            if batch_idx == K - 1 or i <= t:
                grad_new[m] = grad_workers[m]
                break
    if i >= 0 :
        for m in selection:
            for l in range(4):
                diff[l] = (grad_new[m][l] - optimizer.param_groups[0]['params'][l]).detach().numpy().reshape(-1)
            grad_test[m] = np.linalg.norm(np.concatenate((diff[0],diff[1],diff[2],diff[3])))
    temp = [0]*4
    tt = 0
    for m in selection:
        tt += p[m]
    for l in range(4):
        for m in selection:
            temp[l] += (grad_new[m][l]- optimizer.param_groups[0]['params'][l]) *p[m] / tt
        # temp[l] /= len(selection)
    for batch_idx, (data, target) in enumerate(test_loader):
        optimizer.zero_grad()
        for l in range(4):
            optimizer.param_groups[0]['params'][l].grad = -temp[l]
        optimizer.step()
        output = network(data.reshape(batch_size_test, 784))
        # output = network(data)
        loss = F.nll_loss(output, target)
        break
    # optimizer.param_groups[0]['lr'] *= 0.99
    print(loss.item(), "ADA_Wireless_modified5", i)
    print(selection)
    train_losses_ADA_Wireless_modified5.append(loss.item())
    comm_ADA_Wireless_modified5[i] += len(selection)
    comm_ADA_Wireless_modified5[i+1] = comm_ADA_Wireless_modified5[i]
    if test_losses_ADA_Wireless_modified5[-1] >= acc:
        break
#%%

figure()
plot(np.array(comm_ADA_Wireless_modified1[0:len(test_losses_ADA_Wireless_modified1)*10:10])/1, test_losses_ADA_Wireless_modified1,linestyle='-')
plot(np.array(comm_ADA_Wireless_modified2[0:len(test_losses_ADA_Wireless_modified2)*10:10])/5, test_losses_ADA_Wireless_modified2,linestyle='--',color='black')
plot(np.array(comm_ADA_Wireless_modified3[0:len(test_losses_ADA_Wireless_modified3)*10:10])/8, test_losses_ADA_Wireless_modified3,linestyle='-.')
plot(np.array(comm_ADA_Wireless_modified4[0:len(test_losses_ADA_Wireless_modified4)*10:10])/15, test_losses_ADA_Wireless_modified4,linestyle=':')
plot(np.array(comm_ADA_Wireless_modified5[0:len(test_losses_ADA_Wireless_modified5)*10:10])/20, test_losses_ADA_Wireless_modified5)
xlabel('training rounds')
ylabel('test accuracy')
# plt.legend(loc=4)
legend([r"$S=1$",r'$S=5$',r'$S=8$',r'$S=15$',r'$S=20$'])
#plot(train_losses_AMS)
# xlim([0, 30000])
grid('on')
savefig('effect_of_tau_Strain.pdf')

figure()
plot(comm_ADA_Wireless_modified1[0:len(test_losses_ADA_Wireless_modified1)*10:10], test_losses_ADA_Wireless_modified1,linestyle='-')
plot(comm_ADA_Wireless_modified2[0:len(test_losses_ADA_Wireless_modified2)*10:10], test_losses_ADA_Wireless_modified2,linestyle='--',color='black')
plot(comm_ADA_Wireless_modified3[0:len(test_losses_ADA_Wireless_modified3)*10:10], test_losses_ADA_Wireless_modified3,linestyle='-.')
plot(comm_ADA_Wireless_modified4[0:len(test_losses_ADA_Wireless_modified4)*10:10], test_losses_ADA_Wireless_modified4,linestyle=':')
plot(comm_ADA_Wireless_modified5[0:len(test_losses_ADA_Wireless_modified5)*10:10], test_losses_ADA_Wireless_modified5)
xlabel('communication cost')
ylabel('test accuracy')
# plt.legend(loc=4)
legend([r"$S=1$",r'$S=5$',r'$S=8$',r'$S=15$',r'$S=20$'])
#plot(train_losses_AMS)
# xlim([0, 30000])
grid('on')
savefig('effect_of_tau_Scomp.pdf')





