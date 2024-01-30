import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant(conv.bias, 0)


def  conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant(bn.weight, scale)
    nn.init.constant(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
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

# I am sorry for this dull implementation for each layer, I am pretty sure there will be more elegant ways

#For layer 1 2        
class unit_gtcn_12(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gtcn_12, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))# I think this the Bk in the paper.
        nn.init.constant(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset # How many layers in each sub-Network. 

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        
        self.conv_T1 = nn.ModuleList()
        self.conv_T2 = nn.ModuleList()
        
        self.conv_ST11 = nn.ModuleList()
        self.conv_ST12 = nn.ModuleList()
        
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))# There are 3 sub-Networks in the Unit, Here means all the Kernel_size=1
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))
            
            self.conv_T1.append(nn.Conv2d(in_channels, inter_channels, (9,1), padding=(4, 0)))# To build graph from temporal infomation.
            self.conv_T2.append(nn.Conv2d(in_channels, inter_channels, (9,1), padding=(4, 0)))
            
            self.conv_ST11.append(nn.Conv2d(in_channels, inter_channels, 1))# To build graph from temporal infomation.
            self.conv_ST11.append(nn.Conv2d(in_channels, inter_channels, (9,1), padding=(4, 0)))
            self.conv_ST12.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_ST12.append(nn.Conv2d(in_channels, inter_channels, (9,1), padding=(4, 0)))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()
        
        self.A_ch3 = (4*torch.pow(self.A, 2)-self.A - 2*torch.eye(self.A.size(-1)))
        
        for m in self.modules():# return all the modules in the model
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        A_ch3 = self.A_ch3.cuda(x.get_device())
        
        #Note not include the PA during searching
        A = A_ch3+ self.PA  # Is this A the adjecent Matrix? PA is Bk?

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)#: Conv out:N, C, T, V --> N , V, C, T --> N , V, C*T
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V) #: Conv out: N, C, T, V -->  N , C*T, V
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V, and / A1.size(-1) means normalize?? # Note: A1 here is Ck in the Eq. 
            
            A_T1= self.conv_T1[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)#: Conv out:N, C, T, V --> N , V, C, T --> N , V, C*T
            A_T2 = self.conv_T2[i](x).view(N, self.inter_c * T, V) 
            A_T1 = self.soft(torch.matmul(A_T1, A_T2) / A_T1.size(-1))
            
            A_ST11= self.conv_ST11[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)#: Conv out:N, C, T, V --> N , V, C, T --> N , V, C*T
            A_ST12 = self.conv_ST12[i](x).view(N, self.inter_c * T, V) 
            A_ST11 = self.soft(torch.matmul(A_ST11, A_ST12) / A_ST11.size(-1))
            
            #A_ST21= self.conv_ST21[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)#: Conv out:N, C, T, V --> N , V, C, T --> N , V, C*T
            #A_ST22 = self.conv_ST22[i](x).view(N, self.inter_c * T, V) 
            #A_ST21 = self.soft(torch.matmul(A_ST21, A_ST22) / A_ST21.size(-1))
            
            A1 = A[i] + A1 + A_T1 + A_ST11 #+ weights[8]*A_ST21 # Means Ak+Bk+Ck+Tk in Eq(3), in line 95: A = A + B
            
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))# Means f_out in Eq(3)
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)         

#For layer 3 4
class unit_gtcn_34(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gtcn_34, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))# I think this the Bk in the paper.
        nn.init.constant(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset # How many layers in each sub-Network. 

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        
        self.conv_T1 = nn.ModuleList()
        self.conv_T2 = nn.ModuleList()
        
        self.conv_ST11 = nn.ModuleList()
        self.conv_ST12 = nn.ModuleList()
        
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))# There are 3 sub-Networks in the Unit, Here means all the Kernel_size=1
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))
            
            self.conv_T1.append(nn.Conv2d(in_channels, inter_channels, (9,1), padding=(4, 0)))# To build graph from temporal infomation.
            self.conv_T2.append(nn.Conv2d(in_channels, inter_channels, (9,1), padding=(4, 0)))
            
            self.conv_ST11.append(nn.Conv2d(in_channels, inter_channels, 1))# To build graph from temporal infomation.
            self.conv_ST11.append(nn.Conv2d(in_channels, inter_channels, (9,1), padding=(4, 0)))
            self.conv_ST12.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_ST12.append(nn.Conv2d(in_channels, inter_channels, (9,1), padding=(4, 0)))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():# return all the modules in the model
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        
        #Note not include the PA during searching
        A = self.PA  # Is this A the adjecent Matrix? PA is Bk?

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)#: Conv out:N, C, T, V --> N , V, C, T --> N , V, C*T
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V) #: Conv out: N, C, T, V -->  N , C*T, V
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V, and / A1.size(-1) means normalize?? # Note: A1 here is Ck in the Eq. 
            
            A_T1= self.conv_T1[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)#: Conv out:N, C, T, V --> N , V, C, T --> N , V, C*T
            A_T2 = self.conv_T2[i](x).view(N, self.inter_c * T, V) 
            A_T1 = self.soft(torch.matmul(A_T1, A_T2) / A_T1.size(-1))
            
            A_ST11= self.conv_ST11[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)#: Conv out:N, C, T, V --> N , V, C, T --> N , V, C*T
            A_ST12 = self.conv_ST12[i](x).view(N, self.inter_c * T, V) 
            A_ST11 = self.soft(torch.matmul(A_ST11, A_ST12) / A_ST11.size(-1))
            
            
            A1 = A[i] + A1 + A_T1 + A_ST11 #+ weights[8]*A_ST21 # Means Ak+Bk+Ck+Tk in Eq(3), in line 95: A = A + B
            
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))# Means f_out in Eq(3)
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)  


#For layer 5       
class unit_gtcn_5(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gtcn_5, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))# I think this the Bk in the paper.
        nn.init.constant(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset # How many layers in each sub-Network. 

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        
        self.conv_T1 = nn.ModuleList()
        self.conv_T2 = nn.ModuleList()

        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))# There are 3 sub-Networks in the Unit, Here means all the Kernel_size=1
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))
            
            self.conv_T1.append(nn.Conv2d(in_channels, inter_channels, (9,1), padding=(4, 0)))# To build graph from temporal infomation.
            self.conv_T2.append(nn.Conv2d(in_channels, inter_channels, (9,1), padding=(4, 0)))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()
        
        self.A_ch3 = (4*torch.pow(self.A, 2)-self.A - 2*torch.eye(self.A.size(-1)))
        
        for m in self.modules():# return all the modules in the model
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        A_ch3 = self.A_ch3.cuda(x.get_device())
        
        #Note not include the PA during searching
        A = A_ch3+ self.PA  # Is this A the adjecent Matrix? PA is Bk?

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)#: Conv out:N, C, T, V --> N , V, C, T --> N , V, C*T
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V) #: Conv out: N, C, T, V -->  N , C*T, V
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V, and / A1.size(-1) means normalize?? # Note: A1 here is Ck in the Eq. 
            
            A_T1= self.conv_T1[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)#: Conv out:N, C, T, V --> N , V, C, T --> N , V, C*T
            A_T2 = self.conv_T2[i](x).view(N, self.inter_c * T, V) 
            A_T1 = self.soft(torch.matmul(A_T1, A_T2) / A_T1.size(-1))
            
            A1 = A[i] + A1 + A_T1
            
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))# Means f_out in Eq(3)
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)    


#For layer 6 8 9        
class unit_gtcn_689(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gtcn_689, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))# I think this the Bk in the paper.
        nn.init.constant(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset # How many layers in each sub-Network. 

        self.conv_d = nn.ModuleList()
        
        self.conv_T1 = nn.ModuleList()
        self.conv_T2 = nn.ModuleList()

        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))
            
            self.conv_T1.append(nn.Conv2d(in_channels, inter_channels, (9,1), padding=(4, 0)))# To build graph from temporal infomation.
            self.conv_T2.append(nn.Conv2d(in_channels, inter_channels, (9,1), padding=(4, 0)))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()
        
        self.A_ch3 = (4*torch.pow(self.A, 2)-self.A - 2*torch.eye(self.A.size(-1)))
        
        for m in self.modules():# return all the modules in the model
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        A_ch3 = self.A_ch3.cuda(x.get_device())
        
        #Note not include the PA during searching
        A = A_ch3+ self.PA  # Is this A the adjecent Matrix? PA is Bk?

        y = None
        for i in range(self.num_subset):
            A_T1= self.conv_T1[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)#: Conv out:N, C, T, V --> N , V, C, T --> N , V, C*T
            A_T2 = self.conv_T2[i](x).view(N, self.inter_c * T, V) 
            A_T1 = self.soft(torch.matmul(A_T1, A_T2) / A_T1.size(-1))

            
            A1 = A[i] + A_T1
            
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))# Means f_out in Eq(3)
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)   

#For layer 7        
class unit_gtcn_7(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gtcn_7, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))# I think this the Bk in the paper.
        nn.init.constant(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset # How many layers in each sub-Network. 

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        
        self.conv_T1 = nn.ModuleList()
        self.conv_T2 = nn.ModuleList()
        
        self.conv_ST11 = nn.ModuleList()
        self.conv_ST12 = nn.ModuleList()
        
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))# There are 3 sub-Networks in the Unit, Here means all the Kernel_size=1
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))
            
            self.conv_T1.append(nn.Conv2d(in_channels, inter_channels, (9,1), padding=(4, 0)))# To build graph from temporal infomation.
            self.conv_T2.append(nn.Conv2d(in_channels, inter_channels, (9,1), padding=(4, 0)))
            
            self.conv_ST11.append(nn.Conv2d(in_channels, inter_channels, 1))# To build graph from temporal infomation.
            self.conv_ST11.append(nn.Conv2d(in_channels, inter_channels, (9,1), padding=(4, 0)))
            self.conv_ST12.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_ST12.append(nn.Conv2d(in_channels, inter_channels, (9,1), padding=(4, 0)))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()
        
        self.A_ch4s = self.soft((8*torch.pow(self.A, 4)- 4*torch.pow(self.A, 2)-4*self.A +torch.eye(self.A.size(-1)))/self.A.size(-1))
        self.A_ch3 = (4*torch.pow(self.A, 2)-self.A - 2*torch.eye(self.A.size(-1)))
        
        for m in self.modules():# return all the modules in the model
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        A_ch4s = self.A_ch4s.cuda(x.get_device())
        A_ch3 = self.A_ch3.cuda(x.get_device())
        
        #Note not include the PA during searching
        A = A_ch3+ A_ch4s+ self.PA  # Is this A the adjecent Matrix? PA is Bk?

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)#: Conv out:N, C, T, V --> N , V, C, T --> N , V, C*T
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V) #: Conv out: N, C, T, V -->  N , C*T, V
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V, and / A1.size(-1) means normalize?? # Note: A1 here is Ck in the Eq. 
            
            A_T1= self.conv_T1[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)#: Conv out:N, C, T, V --> N , V, C, T --> N , V, C*T
            A_T2 = self.conv_T2[i](x).view(N, self.inter_c * T, V) 
            A_T1 = self.soft(torch.matmul(A_T1, A_T2) / A_T1.size(-1))
            
            A_ST11= self.conv_ST11[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)#: Conv out:N, C, T, V --> N , V, C, T --> N , V, C*T
            A_ST12 = self.conv_ST12[i](x).view(N, self.inter_c * T, V) 
            A_ST11 = self.soft(torch.matmul(A_ST11, A_ST12) / A_ST11.size(-1))
            
            #A_ST21= self.conv_ST21[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)#: Conv out:N, C, T, V --> N , V, C, T --> N , V, C*T
            #A_ST22 = self.conv_ST22[i](x).view(N, self.inter_c * T, V) 
            #A_ST21 = self.soft(torch.matmul(A_ST21, A_ST22) / A_ST21.size(-1))
            
            A1 = A[i] + A1 + A_T1 + A_ST11 #+ weights[8]*A_ST21 # Means Ak+Bk+Ck+Tk in Eq(3), in line 95: A = A + B
            
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))# Means f_out in Eq(3)
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)   

#For layer 10        
class unit_gtcn_10(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gtcn_10, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))# I think this the Bk in the paper.
        nn.init.constant(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset # How many layers in each sub-Network. 

        self.conv_d = nn.ModuleList()
        
        self.conv_T1 = nn.ModuleList()
        self.conv_T2 = nn.ModuleList()

        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))
            
            self.conv_T1.append(nn.Conv2d(in_channels, inter_channels, (9,1), padding=(4, 0)))# To build graph from temporal infomation.
            self.conv_T2.append(nn.Conv2d(in_channels, inter_channels, (9,1), padding=(4, 0)))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()
        
        for m in self.modules():# return all the modules in the model
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        
        #Note not include the PA during searching
        A = self.PA  # Is this A the adjecent Matrix? PA is Bk?

        y = None
        for i in range(self.num_subset):
            A_T1= self.conv_T1[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)#: Conv out:N, C, T, V --> N , V, C, T --> N , V, C*T
            A_T2 = self.conv_T2[i](x).view(N, self.inter_c * T, V) 
            A_T1 = self.soft(torch.matmul(A_T1, A_T2) / A_T1.size(-1))

            A1 = A[i] + A_T1
            
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))# Means f_out in Eq(3)
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)  
        
        
class TCN_GCN_unit_12(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit_12, self).__init__()
        self.gcn1 = unit_gtcn_12(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)

        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)
        
class TCN_GCN_unit_34(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit_34, self).__init__()
        #self.gcn1 = unit_gtcn(in_channels, out_channels, A)
        self.gcn1 = unit_gtcn_34(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        #self.tcn1 = unit_tcn_G(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)
        
class TCN_GCN_unit_5(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit_5, self).__init__()
        #self.gcn1 = unit_gtcn(in_channels, out_channels, A)
        self.gcn1 = unit_gtcn_5(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        #self.tcn1 = unit_tcn_G(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)
        
class TCN_GCN_unit_689(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit_689, self).__init__()
        #self.gcn1 = unit_gtcn(in_channels, out_channels, A)
        self.gcn1 = unit_gtcn_689(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        #self.tcn1 = unit_tcn_G(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)        

class TCN_GCN_unit_7(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit_7, self).__init__()
        self.gcn1 = unit_gtcn_7(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)

        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)

class TCN_GCN_unit_10(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit_10, self).__init__()
        #self.gcn1 = unit_gtcn(in_channels, out_channels, A)
        self.gcn1 = unit_gtcn_10(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        #self.tcn1 = unit_tcn_G(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x) 


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



class ResBlock(nn.Module):
    def __init__(self, A):
        super(ResBlock, self).__init__()
        self.w = nn.Parameter(torch.ones(2))
        self.res_gcn = unit_gtcn_5(64, 128, A)
        self.res_tcn = unit_tcn(128, 128, stride=2)

    def forward(self,x1, x):
        # 归一化权重
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        # 多特征融合
        x1 = self.res_gcn(x1)
        x1 = self.res_tcn(x1)
        x_out = x1 * w1 + x * w2
        #print('w1'+str(w1)+'w2'+str(w2))
        return x_out

class ResBlock1(nn.Module):
    def __init__(self, A1):
        super(ResBlock1, self).__init__()
        self.w = nn.Parameter(torch.ones(2))
        self.res_gcn1 = unit_gtcn_5(128, 256, A1)
        self.res_tcn1 = unit_tcn(256, 256, stride=2)

    def forward(self,x1, x):
        # 归一化权重
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        # 多特征融合
        x1 = self.res_gcn1(x1)
        x1 = self.res_tcn1(x1)
        x_out = x1 * w1 + x * w2
        #print('w3' + str(w1) + 'w4' + str(w2))
        return x_out
  


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, in_channels=3):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            self.graph = graph

        A = self.graph.A
        # self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.data_bn = nn.BatchNorm1d(num_person * in_channels *2* num_point)

        # self.l1 = TCN_GCN_unit_12(3, 64, A, residual=False)
        self.l1 = TCN_GCN_unit_12(6, 64, A, residual=False)
        self.l2 = TCN_GCN_unit_12(64, 64, A)
        self.l3 = TCN_GCN_unit_34(64, 64, A)
        self.l4 = TCN_GCN_unit_34(64, 64, A)
        self.l5 = TCN_GCN_unit_5(64, 128, A, stride=2)
        self.l6 = TCN_GCN_unit_689(128, 128, A)
        self.l7 = TCN_GCN_unit_7(128, 128, A)
        self.l8 = TCN_GCN_unit_689(128, 256, A, stride=2)
        self.l9 = TCN_GCN_unit_689(256, 256, A)
        self.l10 = TCN_GCN_unit_10(256, 256, A)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        self.res1 = ResBlock(A)
        self.res2 = ResBlock1(A)
        self.mm_fusion = MFA(in_channels * 2, 16)
        # self.mm_fusion1 = MFA(in_channels * 2, 16)

    def forward(self, x,x1):
        x = torch.cat([x,x1],dim=1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.mm_fusion(x)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        # a = x
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        # x = self.res1(a,x)
        # b = x
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        # x = self.res2(b,x)
        

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return self.fc(x)