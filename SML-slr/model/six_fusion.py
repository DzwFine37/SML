import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from model.dropSke import DropBlock_Ske
from model.dropT import DropBlockT_1d

def conv_branch_init(conv):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal(weight, 0, math.sqrt(2. / (n * k1 * k2)))
    nn.init.constant(conv.bias, 0)

def  conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

# usage: self.mm_fusion = MFA(in_channels*2, 16)

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, num_point=25, block_size=41):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

        self.dropS = DropBlock_Ske(num_point=num_point)
        self.dropT = DropBlockT_1d(block_size=block_size)

    def forward(self, x, keep_prob, A):
        x = self.bn(self.conv(x))
        x = self.dropT(self.dropS(x, keep_prob, A), keep_prob)
        return x


class unit_tcn_skip(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn_skip, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, groups, num_point, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_point = num_point
        self.groups = groups
        self.num_subset = num_subset
        self.DecoupleA = nn.Parameter(torch.tensor(np.reshape(A.astype(np.float32), [
            3, 1, num_point, num_point]), dtype=torch.float32, requires_grad=True).repeat(1, groups, 1, 1),
                                      requires_grad=True)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn0 = nn.BatchNorm2d(out_channels * num_subset)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

        self.Linear_weight = nn.Parameter(torch.zeros(
            in_channels, out_channels * num_subset, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0, math.sqrt(
            0.5 / (out_channels * num_subset)))

        self.Linear_bias = nn.Parameter(torch.zeros(
            1, out_channels * num_subset, 1, 1, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.constant(self.Linear_bias, 1e-6)

        eye_array = []
        for i in range(out_channels):
            eye_array.append(torch.eye(num_point))
        self.eyes = nn.Parameter(torch.tensor(torch.stack(
            eye_array), requires_grad=False, device='cuda'), requires_grad=False)  # [c,25,25]

    def norm(self, A):
        b, c, h, w = A.size()
        A = A.view(c, self.num_point, self.num_point)
        D_list = torch.sum(A, 1).view(c, 1, self.num_point)
        D_list_12 = (D_list + 0.001) ** (-1)
        D_12 = self.eyes * D_list_12
        A = torch.bmm(A, D_12).view(b, c, h, w)
        return A

    def forward(self, x0):
        learn_A = self.DecoupleA.repeat(
            1, self.out_channels // self.groups, 1, 1)
        norm_learn_A = torch.cat([self.norm(learn_A[0:1, ...]), self.norm(
            learn_A[1:2, ...]), self.norm(learn_A[2:3, ...])], 0)

        x = torch.einsum(
            'nctw,cd->ndtw', (x0, self.Linear_weight)).contiguous()
        x = x + self.Linear_bias
        x = self.bn0(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.num_subset, kc // self.num_subset, t, v)
        x = torch.einsum('nkctv,kcvw->nctw', (x, norm_learn_A))

        x = self.bn(x)
        x += self.down(x0)
        x = self.relu(x)
        return x


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, groups, num_point, block_size, stride=1, residual=True,
                 attention=True):
        super(TCN_GCN_unit, self).__init__()
        num_jpts = A.shape[-1]
        self.gcn1 = unit_gcn(in_channels, out_channels, A, groups, num_point)
        self.tcn1 = unit_tcn(out_channels, out_channels,
                             stride=stride, num_point=num_point)
        self.relu = nn.ReLU()

        self.A = nn.Parameter(torch.tensor(np.sum(np.reshape(A.astype(np.float32), [
            3, num_point, num_point]), axis=0), dtype=torch.float32, requires_grad=False, device='cuda'),
                              requires_grad=False)

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn_skip(
                in_channels, out_channels, kernel_size=1, stride=stride)
        self.dropSke = DropBlock_Ske(num_point=num_point)
        self.dropT_skip = DropBlockT_1d(block_size=block_size)
        self.attention = attention
        if attention:
            print('Attention Enabled!')
            self.sigmoid = nn.Sigmoid()
            # temporal attention
            self.conv_ta = nn.Conv1d(out_channels, 1, 9, padding=4)
            nn.init.constant_(self.conv_ta.weight, 0)
            nn.init.constant_(self.conv_ta.bias, 0)
            # s attention
            ker_jpt = num_jpts - 1 if not num_jpts % 2 else num_jpts
            pad = (ker_jpt - 1) // 2
            self.conv_sa = nn.Conv1d(out_channels, 1, ker_jpt, padding=pad)
            nn.init.xavier_normal_(self.conv_sa.weight)
            nn.init.constant_(self.conv_sa.bias, 0)
            # channel attention
            rr = 2
            self.fc1c = nn.Linear(out_channels, out_channels // rr)
            self.fc2c = nn.Linear(out_channels // rr, out_channels)
            nn.init.kaiming_normal_(self.fc1c.weight)
            nn.init.constant_(self.fc1c.bias, 0)
            nn.init.constant_(self.fc2c.weight, 0)
            nn.init.constant_(self.fc2c.bias, 0)

    def forward(self, x, keep_prob):
        y = self.gcn1(x)
        if self.attention:
            # spatial attention
            se = y.mean(-2)  # N C V
            se1 = self.sigmoid(self.conv_sa(se))
            y = y * se1.unsqueeze(-2) + y
            # a1 = se1.unsqueeze(-2)

            # temporal attention
            se = y.mean(-1)
            se1 = self.sigmoid(self.conv_ta(se))
            y = y * se1.unsqueeze(-1) + y
            # a2 = se1.unsqueeze(-1)

            # channel attention
            se = y.mean(-1).mean(-1)
            se1 = self.relu(self.fc1c(se))
            se2 = self.sigmoid(self.fc2c(se1))
            y = y * se2.unsqueeze(-1).unsqueeze(-1) + y
            # a3 = se2.unsqueeze(-1).unsqueeze(-1)

        y = self.tcn1(y, keep_prob, self.A)
        x_skip = self.dropT_skip(self.dropSke(self.residual(x), keep_prob, self.A), keep_prob)
        return self.relu(y + x_skip)

class ResBlock(nn.Module):
    def __init__(self, A1, groups, num_point, block_size):
        super(ResBlock, self).__init__()
        self.w = nn.Parameter(torch.ones(2))
        self.p = TCN_GCN_unit(64, 128, A1, groups, num_point, block_size,stride=2)

    def forward(self,a, x):
        # 归一化权重
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        # 多特征融合
        x_out = self.p(a,1.0) * w1 + x * w2
        #print('w1'+str(w1)+'w2'+str(w2))
        return x_out

class ResBlock1(nn.Module):
    def __init__(self, A1, groups, num_point, block_size):
        super(ResBlock1, self).__init__()
        self.w = nn.Parameter(torch.ones(2))
        self.p = TCN_GCN_unit(128, 256, A1, groups, num_point, block_size,stride=2)

    def forward(self,a, x):
        # 归一化权重
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        # 多特征融合
        x_out = self.p(a,1.0) * w1 + x * w2
        #print('w3' + str(w1) + 'w4' + str(w2))
        return x_out

class MFA(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(MFA, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2

        self.conv_v = nn.Conv2d(self.inplanes, self.planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_k = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_q = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)

        self.conv_c = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_down = nn.Conv2d(self.planes + self.inplanes, self.inplanes, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self._init_modules()

    def _init_modules(self):
        conv_init(self.conv_v)
        conv_init(self.conv_k)
        conv_init(self.conv_q)
        conv_init(self.conv_c)
        conv_init(self.conv_down)
        bn_init(self.bn, 1)

    def forward(self, x):
        value, key, query = self.conv_v(x), self.conv_k(x), self.conv_q(x)

        batch, channel, T, V = key.size()
        # [N, C, T*V]
        key = key.view(batch, channel, T * V)

        # [N, 1, T*V]
        query = query.view(batch, 1, T * V)

        # [N, 1, T*V]
        query = self.softmax(query)

        # [N, C, 1, 1]
        interaction = torch.matmul(key, query.transpose(1, 2)).unsqueeze(-1)

        # [N, \hat{C}, 1, 1]
        interaction = self.conv_c(interaction)

        # [N, \hat{C}, 1, 1]
        attention = self.sigmoid(interaction)

        attended_emb = value * attention

        out = self.bn(self.conv_down(torch.cat([attended_emb, x], dim=1)))

        return out

class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=1, groups=8, block_size=41, graph=None, in_channels=3):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            self.graph = graph

        A = self.graph.A  # 3,25,25
        # M1_A = np.random.randn(1, 1, num_point, num_point)
        # self.M1_A = nn.Parameter(torch.from_numpy(M1_A.astype(np.float32)), requires_grad=True)
        # M2_A = np.random.randn(1, 1, num_point, num_point)
        # self.M2_A = nn.Parameter(torch.from_numpy(M2_A.astype(np.float32)), requires_grad=True)
        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * 2 * num_point)
        self.data_bn1 = nn.BatchNorm1d(num_person * in_channels * 2 * num_point)
        self.data_bn2 = nn.BatchNorm1d(num_person * in_channels  * num_point)
        self.data_bn3 = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.data_bn4 = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.data_bn5 = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.mm_fusion = MFA(in_channels * 2, 16)
        self.mm_fusion1 = MFA(in_channels * 2, 16)
       
        # self.mm_fusion = multimodal_conv(in_channels, in_channels)
        # self.mm_bn = nn.BatchNorm2d(in_channels * 2)
        # self.relu = nn.ReLU(inplace=True)
        self.jm1 = TCN_GCN_unit(in_channels * 2, 64, A, groups, num_point,
                               block_size, residual=False)
        self.jm2 = TCN_GCN_unit(64, 64, A, groups, num_point, block_size)
        self.jm3 = TCN_GCN_unit(64, 64, A, groups, num_point, block_size)
        self.jm4 = TCN_GCN_unit(64, 64, A, groups, num_point, block_size)
        self.jm5 = TCN_GCN_unit(
            64, 128, A, groups, num_point, block_size, stride=2)
        self.jm6 = TCN_GCN_unit(128, 128, A, groups, num_point, block_size)
        self.jm7 = TCN_GCN_unit(128, 128, A, groups, num_point, block_size)
        self.jm8 = TCN_GCN_unit(128, 256, A, groups,
                               num_point, block_size, stride=2)
        self.jm9 = TCN_GCN_unit(256, 256, A, groups, num_point, block_size)
        self.jm10 = TCN_GCN_unit(256, 256, A, groups, num_point, block_size)
        self.jmRes = ResBlock(A, groups, num_point, block_size)
        self.jmRes1 = ResBlock1(A, groups, num_point, block_size)

        self.bm1 = TCN_GCN_unit(in_channels * 2, 64, A, groups, num_point,
                               block_size, residual=False)
        self.bm2 = TCN_GCN_unit(64, 64, A, groups, num_point, block_size)
        self.bm3 = TCN_GCN_unit(64, 64, A, groups, num_point, block_size)
        self.bm4 = TCN_GCN_unit(64, 64, A, groups, num_point, block_size)
        self.bm5 = TCN_GCN_unit(
            64, 128, A, groups, num_point, block_size, stride=2)
        self.bm6 = TCN_GCN_unit(128, 128, A, groups, num_point, block_size)
        self.bm7 = TCN_GCN_unit(128, 128, A, groups, num_point, block_size)
        self.bm8 = TCN_GCN_unit(128, 256, A, groups,
                               num_point, block_size, stride=2)
        self.bm9 = TCN_GCN_unit(256, 256, A, groups, num_point, block_size)
        self.bm10 = TCN_GCN_unit(256, 256, A, groups, num_point, block_size)
        self.bmRes = ResBlock(A, groups, num_point, block_size)
        self.bmRes1 = ResBlock1(A, groups, num_point, block_size)

        self.j1 = TCN_GCN_unit(in_channels , 64, A, groups, num_point,
                               block_size, residual=False)
        self.j2 = TCN_GCN_unit(64, 64, A, groups, num_point, block_size)
        self.j3 = TCN_GCN_unit(64, 64, A, groups, num_point, block_size)
        self.j4 = TCN_GCN_unit(64, 64, A, groups, num_point, block_size)
        self.j5 = TCN_GCN_unit(
            64, 128, A, groups, num_point, block_size, stride=2)
        self.j6 = TCN_GCN_unit(128, 128, A, groups, num_point, block_size)
        self.j7 = TCN_GCN_unit(128, 128, A, groups, num_point, block_size)
        self.j8 = TCN_GCN_unit(128, 256, A, groups,
                               num_point, block_size, stride=2)
        self.j9 = TCN_GCN_unit(256, 256, A, groups, num_point, block_size)
        self.j10 = TCN_GCN_unit(256, 256, A, groups, num_point, block_size)
        self.jRes = ResBlock(A, groups, num_point, block_size)
        self.jRes1 = ResBlock1(A, groups, num_point, block_size)

        self.b1 = TCN_GCN_unit(in_channels, 64, A, groups, num_point,
                               block_size, residual=False)
        self.b2 = TCN_GCN_unit(64, 64, A, groups, num_point, block_size)
        self.b3 = TCN_GCN_unit(64, 64, A, groups, num_point, block_size)
        self.b4 = TCN_GCN_unit(64, 64, A, groups, num_point, block_size)
        self.b5 = TCN_GCN_unit(
            64, 128, A, groups, num_point, block_size, stride=2)
        self.b6 = TCN_GCN_unit(128, 128, A, groups, num_point, block_size)
        self.b7 = TCN_GCN_unit(128, 128, A, groups, num_point, block_size)
        self.b8 = TCN_GCN_unit(128, 256, A, groups,
                               num_point, block_size, stride=2)
        self.b9 = TCN_GCN_unit(256, 256, A, groups, num_point, block_size)
        self.b10 = TCN_GCN_unit(256, 256, A, groups, num_point, block_size)
        self.bRes = ResBlock(A, groups, num_point, block_size)
        self.bRes1 = ResBlock1(A, groups, num_point, block_size)

        self.jmotion1 = TCN_GCN_unit(in_channels, 64, A, groups, num_point,
                               block_size, residual=False)
        self.jmotion2 = TCN_GCN_unit(64, 64, A, groups, num_point, block_size)
        self.jmotion3 = TCN_GCN_unit(64, 64, A, groups, num_point, block_size)
        self.jmotion4 = TCN_GCN_unit(64, 64, A, groups, num_point, block_size)
        self.jmotion5 = TCN_GCN_unit(
            64, 128, A, groups, num_point, block_size, stride=2)
        self.jmotion6 = TCN_GCN_unit(128, 128, A, groups, num_point, block_size)
        self.jmotion7 = TCN_GCN_unit(128, 128, A, groups, num_point, block_size)
        self.jmotion8 = TCN_GCN_unit(128, 256, A, groups,
                               num_point, block_size, stride=2)
        self.jmotion9 = TCN_GCN_unit(256, 256, A, groups, num_point, block_size)
        self.jmotion10 = TCN_GCN_unit(256, 256, A, groups, num_point, block_size)
        self.jmotionRes = ResBlock(A, groups, num_point, block_size)
        self.jmotionRes1 = ResBlock1(A, groups, num_point, block_size)

        self.bmotion1 = TCN_GCN_unit(in_channels, 64, A, groups, num_point,
                               block_size, residual=False)
        self.bmotion2 = TCN_GCN_unit(64, 64, A, groups, num_point, block_size)
        self.bmotion3 = TCN_GCN_unit(64, 64, A, groups, num_point, block_size)
        self.bmotion4 = TCN_GCN_unit(64, 64, A, groups, num_point, block_size)
        self.bmotion5 = TCN_GCN_unit(
            64, 128, A, groups, num_point, block_size, stride=2)
        self.bmotion6 = TCN_GCN_unit(128, 128, A, groups, num_point, block_size)
        self.bmotion7 = TCN_GCN_unit(128, 128, A, groups, num_point, block_size)
        self.bmotion8 = TCN_GCN_unit(128, 256, A, groups,
                               num_point, block_size, stride=2)
        self.bmotion9 = TCN_GCN_unit(256, 256, A, groups, num_point, block_size)
        self.bmotion10 = TCN_GCN_unit(256, 256, A, groups, num_point, block_size)
        self.bmotionRes = ResBlock(A, groups, num_point, block_size)
        self.bmotionRes1 = ResBlock1(A, groups, num_point, block_size)

        # temporal attention
        # self.sigmoid = nn.Sigmoid()
        # self.conv_ta = nn.Conv1d(256, 1, 9, padding=4)
        # nn.init.constant_(self.conv_ta.weight, 0)
        # nn.init.constant_(self.conv_ta.bias, 0)

        self.jmfc = nn.Linear(256, num_class)
        self.bmfc = nn.Linear(256, num_class)
        self.jfc = nn.Linear(256, num_class)
        self.bfc = nn.Linear(256, num_class)
        self.jmotionfc = nn.Linear(256, num_class)
        self.bmotionfc = nn.Linear(256, num_class)
        nn.init.normal(self.jmfc.weight, 0, math.sqrt(2. / num_class))
        nn.init.normal(self.bmfc.weight, 0, math.sqrt(2. / num_class))
        nn.init.normal(self.jfc.weight, 0, math.sqrt(2. / num_class))
        nn.init.normal(self.bfc.weight, 0, math.sqrt(2. / num_class))
        nn.init.normal(self.jmotionfc.weight, 0, math.sqrt(2. / num_class))
        nn.init.normal(self.bmotionfc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        bn_init(self.data_bn1, 1)
        bn_init(self.data_bn2, 1)
        bn_init(self.data_bn3, 1)
        bn_init(self.data_bn4, 1)
        bn_init(self.data_bn5, 1)
        # bn_init(self.mm_bn, 1)
        # if drop_out:
        #     self.drop_out = nn.Dropout(drop_out)
        # else:
        #     self.drop_out = lambda x: x

    def forward(self, j,jmotion,b,bmotion,keep_prob=0.9):
        jm = torch.cat([j,jmotion],dim=1)
        bm = torch.cat([b,bmotion],dim=1)

        N, C, T, V, M = jm.size()
        jm = jm.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        jm = self.data_bn(jm)
        jm = jm.view(N, M, V, C, T).permute(
            0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        N, C, T, V, M = bm.size()
        bm = bm.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        bm = self.data_bn1(bm)
        bm = bm.view(N, M, V, C, T).permute(
            0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        N, C, T, V, M = j.size()
        j = j.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        j = self.data_bn2(j)
        j = j.view(N, M, V, C, T).permute(
            0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        N, C, T, V, M = b.size()
        b = b.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        b = self.data_bn3(b)
        b = b.view(N, M, V, C, T).permute(
            0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        N, C, T, V, M = jmotion.size()
        jmotion = jmotion.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        jmotion = self.data_bn2(jmotion)
        jmotion = jmotion.view(N, M, V, C, T).permute(
            0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        N, C, T, V, M = bmotion.size()
        bmotion = bmotion.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        bmotion = self.data_bn2(bmotion)
        bmotion = bmotion.view(N, M, V, C, T).permute(
            0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        
        jm = self.mm_fusion(jm)
        save1 = jm
        bm = self.mm_fusion1(bm)
        save2 = bm

        jm = self.jm1(jm, 1.0)
        jm = self.jm2(jm, 1.0)
        jm = self.jm3(jm, 1.0)
        a1 = jm
        save3 = a1
        jm = self.jm4(jm, 1.0)
        jm = self.jm5(jm, 1.0)
        jm = self.jm6(jm, 1.0)
        jm = self.jmRes(a1, jm)
        jm = self.jm7(jm, keep_prob)
        b1 = jm
        jm = self.jm8(jm, keep_prob)
        jm = self.jm9(jm, keep_prob)
        jm = self.jm10(jm, keep_prob)
        jm = self.jmRes1(b1,jm)
        c_new = jm.size(1)
        jm = jm.reshape(N, M, c_new, -1)
        jm = jm.mean(3).mean(1)
        jm_out = self.jmfc(jm)

        bm = self.bm1(bm, 1.0)
        bm = self.bm2(bm, 1.0)
        bm = self.bm3(bm, 1.0)
        a2 = bm
        save4 = a2
        bm = self.bm4(bm, 1.0)
        bm = self.bm5(bm, 1.0)
        bm = self.bm6(bm, 1.0)
        bm = self.bmRes(a2, bm)
        bm = self.bm7(bm, keep_prob)
        b2 = bm
        bm = self.bm8(bm, keep_prob)
        bm = self.bm9(bm, keep_prob)
        bm = self.bm10(bm, keep_prob)
        bm = self.bmRes1(b2, bm)
        c_new = bm.size(1)
        bm = bm.reshape(N, M, c_new, -1)
        bm = bm.mean(3).mean(1)
        bm_out = self.bmfc(bm)

        j = self.j1(j, 1.0)
        j = self.j2(j, 1.0)
        j = self.j3(j, 1.0)
        a3 = j
        save5 = a3
        j = self.j4(j, 1.0)
        j = self.j5(j, 1.0)
        j = self.j6(j, 1.0)
        j = self.jRes(a3, j)
        j = self.j7(j, keep_prob)
        b3 = j
        j = self.j8(j, keep_prob)
        j = self.j9(j, keep_prob)
        j = self.j10(j, keep_prob)
        j = self.jRes1(b3, j)
        c_new = j.size(1)
        j = j.reshape(N, M, c_new, -1)
        j = j.mean(3).mean(1)
        j_out = self.jfc(j)

        b = self.b1(b, 1.0)
        b = self.b2(b, 1.0)
        b = self.b3(b, 1.0)
        a4 = b
        save6 = a4
        b = self.b4(b, 1.0)
        b = self.b5(b, 1.0)
        b = self.b6(b, 1.0)
        b = self.bRes(a4, b)
        b = self.b7(b, keep_prob)
        b4 = b
        b = self.b8(b, keep_prob)
        b = self.b9(b, keep_prob)
        b = self.b10(b, keep_prob)
        b = self.bRes1(b4, b)
        c_new = b.size(1)
        b = b.reshape(N, M, c_new, -1)
        b = b.mean(3).mean(1)
        b_out = self.bfc(b)

        jmotion = self.jmotion1(jmotion, 1.0)
        jmotion = self.jmotion2(jmotion, 1.0)
        jmotion = self.jmotion3(jmotion, 1.0)
        a5 = jmotion
        save7 = a5
        jmotion = self.jmotion4(jmotion, 1.0)
        jmotion = self.jmotion5(jmotion, 1.0)
        jmotion = self.jmotion6(jmotion, 1.0)
        jmotion = self.jmotionRes(a5, jmotion)
        jmotion = self.jmotion7(jmotion, keep_prob)
        b5 = jmotion
        jmotion = self.jmotion8(jmotion, keep_prob)
        jmotion = self.jmotion9(jmotion, keep_prob)
        jmotion = self.jmotion10(jmotion, keep_prob)
        jmotion = self.jmotionRes1(b5, jmotion)
        c_new = jmotion.size(1)
        jmotion = jmotion.reshape(N, M, c_new, -1)
        jmotion = jmotion.mean(3).mean(1)
        jmotion_out = self.jmotionfc(jmotion)

        bmotion = self.bmotion1(bmotion, 1.0)
        bmotion = self.bmotion2(bmotion, 1.0)
        bmotion = self.bmotion3(bmotion, 1.0)
        a6 = bmotion
        save8 = a6
        bmotion = self.bmotion4(bmotion, 1.0)
        bmotion = self.bmotion5(bmotion, 1.0)
        bmotion = self.bmotion6(bmotion, 1.0)
        bmotion = self.bmotionRes(a6, bmotion)
        bmotion = self.bmotion7(bmotion, keep_prob)
        b6 = bmotion
        bmotion = self.bmotion8(bmotion, keep_prob)
        bmotion = self.bmotion9(bmotion, keep_prob)
        bmotion = self.bmotion10(bmotion, keep_prob)
        bmotion = self.bmotionRes1(b6, bmotion)
        c_new = bmotion.size(1)
        bmotion = bmotion.reshape(N, M, c_new, -1)
        bmotion = bmotion.mean(3).mean(1)
        bmotion_out = self.bmotionfc(bmotion)

        with open(r'/data/code/SAM-SLR-v2-main1/save_feature.pkl','wb')as f:
            dict = {'save1':save1,'save2':save2,'save3':save3,'save4':save4,'save5':save5,'save6':save6,'save7':save7,'save8':save8}
            pickle.dump(dict,f)
        print('save done')

        all = 1.0*jm_out+1.0*bm_out+1.0*j_out+0.5*jmotion_out+0.9*b_out+0.5*bmotion_out

        return all,jm_out,bm_out,j_out,b_out,jmotion_out,bmotion_out