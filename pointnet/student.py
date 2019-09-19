# student model
# keep the basic structure, alter it's layer number and widths

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import torch.nn.functional as F
import time
import torch.optim as optim
from torch.nn.init import xavier_uniform, calculate_gain
import math
import copy

USE_CUDA = torch.cuda.is_available()


class STN3d(nn.Module):

    def __init__(self,
                 convs=[(3, 64, 1), (64, 128, 1), (128, 1024, 1)],
                 fcs=[(1024, 512), (512, 256), (256, 9)]):
        super(STN3d, self).__init__()
        self.k = convs[0][0]

        # convs
        self.max_width = convs[-1][1]
        self.convs = nn.ModuleList()  # maybe module dict is better?
        for i in range(len(convs)):
            self.convs.append(nn.Conv1d(*convs[i]))
            self.convs.append(nn.BatchNorm1d(convs[i][1]))
            self.convs.append(nn.ReLU())

        self.fcs = nn.ModuleList()
        for i in range(len(fcs) - 1):
            self.fcs.append(nn.Linear(*fcs[i]))
            self.fcs.append(nn.BatchNorm1d(fcs[i][1]))
            self.fcs.append(nn.ReLU())
        self.fcs.append(nn.Linear(*fcs[-1]))

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.convs(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.max_width)
        x = self.fcs(x)
        iden = torch.eye(self.k).view(1, -1).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        return (x + iden).view(-1, self.k, self.k)


class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False,
                 conv=[(3, 64, 1), (64, 128, 1), (128, 1024, 1)],
                 pointfeat=3
                 ):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(convs=[(64, 64, 1), (64, 128, 1), (128, 1024, 1)],
                         fcs=[(1024, 512), (512, 256), (256, 64 * 64)])

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(convs=[(64, 64, 1), (64, 128, 1), (128, 1024, 1)],
                              fcs=[(1024, 512), (512, 256), (256, 64 * 64)])

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(
            global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(
        torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


if __name__ == '__main__':
    sim_data = torch.rand(32, 3, 2500, requires_grad=True)
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = torch.rand(32, 64, 2500, requires_grad=True)
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k=5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    # seg = PointNetDenseCls(k=3)
    # out, _, _ = seg(sim_data)
    # print('seg', out.size())
