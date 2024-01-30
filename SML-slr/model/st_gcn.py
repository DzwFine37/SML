# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class GraphConvolution(nn.Module):
# 	def __init__(self, input_dim, output_dim, num_pointetex, act=F.relu, dropout=0.5, bias=True):
# 		super(GraphConvolution, self).__init__()

# 		self.alpha = 1.

# 		self.act = act
# 		self.dropout = nn.Dropout(dropout)
# 		self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
# 		if bias:
# 			self.bias = nn.Parameter(torch.randn(output_dim))
# 		else:
# 			self.bias = None

# 		for w in [self.weight]:
# 			nn.init.xavier_normal_(w)

# 	def normalize(self, m):
# 		rowsum = torch.sum(m, 0)
# 		r_inv = torch.pow(rowsum, -0.5)
# 		r_mat_inv = torch.diag(r_inv).float()

# 		m_norm = torch.mm(r_mat_inv, m)
# 		m_norm = torch.mm(m_norm, r_mat_inv)

# 		return m_norm

# 	def forward(self, graph, x):

# 		x = self.dropout(x)

# 		# K-ordered Chebyshev polynomial
# 		graph_norm = self.normalize(graph)
# 		sqr_norm = self.normalize(torch.mm(graph,graph))
# 		m_norm = self.alpha*graph_norm + (1.-self.alpha)*sqr_norm

# 		x_tmp = torch.einsum('abcd,de->abce', x, self.weight)
# 		x_out = torch.einsum('ij,abid->abjd', m_norm, x_tmp)
# 		if self.bias is not None:
# 			x_out += self.bias

# 		x_out = self.act(x_out)
		
# 		return x_out
		

# class StandConvolution(nn.Module):
# 	def __init__(self, dims, num_class, dropout):
# 		super(StandConvolution, self).__init__()

# 		self.dropout = nn.Dropout(dropout)
# 		self.conv = nn.Sequential(
# 								   nn.Conv2d(dims[0], dims[1], kernel_size=5, stride=2),
# 								   nn.InstanceNorm2d(dims[1]),
# 								   nn.ReLU(inplace=True),
# 								   #nn.AvgPool2d(3, stride=2),
# 								   nn.Conv2d(dims[1], dims[2], kernel_size=5, stride=2),
# 								   nn.InstanceNorm2d(dims[2]),
# 								   nn.ReLU(inplace=True),
# 								   #nn.AvgPool2d(3, stride=2),
# 								   nn.Conv2d(dims[2], dims[3], kernel_size=5, stride=2),
# 								   nn.InstanceNorm2d(dims[3]),
# 								   nn.ReLU(inplace=True),
# 								   #nn.AvgPool2d(3, stride=2)
# 								   )

# 		self.fc = nn.Linear(dims[3]*3, num_class)

# 	def forward(self, x):
# 		x = self.dropout(x.permute(0,3,1,2))
# 		x_tmp = self.conv(x)
# 		x_out = self.fc(x_tmp.view(x.size(0), -1))

# 		return x_out


# class StandRecurrent(nn.Module):
# 	def __init__(self, dims, num_class, dropout):
# 		super(StandRecurrent, self).__init__()

# 		self.lstm = nn.LSTM(dims[0]*45, dims[1], batch_first=True,
# 							dropout=0)
# 		self.fc = nn.Linear(dims[1], num_class)

# 	def forward(self, x):
# 		x_tmp,_ = self.lstm(x.contiguous().view(x.size(0), x.size(1), -1))
# 		x_out = self.fc(x_tmp[:,-1])

# 		return x_out


# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import torch
# # import numpy as np

# # from layer import GraphConvolution, StandConvolution

# class GGCN(nn.Module):
# 	def __init__(self, graph, num_point, num_class, gc_dims, sc_dims, feat_dims, dropout=0.5):
# 		super(GGCN, self).__init__()

# 		gc_dims = [num_point, num_point * 3]
# 		sc_dims = [num_point * 3, 16,32,64]
# 		feat_dims = 13

# 		terminal_cnt = 5
# 		actor_cnt = 1
# 		graph = graph + torch.eye(graph.size(0)).to(graph).detach()
# 		ident = torch.eye(graph.size(0)).to(graph)
# 		zeros = torch.zeros(graph.size(0), graph.size(1)).to(graph)
# 		self.graph = torch.cat([torch.cat([graph, ident, zeros], 1),
# 							  torch.cat([ident, graph, ident], 1),
# 							  torch.cat([zeros, ident, graph], 1)], 0).float()
# 		self.terminal = nn.Parameter(torch.randn(terminal_cnt, actor_cnt, feat_dims))

# 		self.gcl = GraphConvolution(gc_dims[0]+feat_dims, gc_dims[1], num_point, dropout=dropout)
# 		self.conv= StandConvolution(sc_dims, num_class, dropout=dropout)

# 		nn.init.xavier_normal_(self.terminal)

# 	def forward(self, x):
# 		head_la = F.interpolate(torch.stack([self.terminal[0],self.terminal[1]],2), 6)
# 		head_ra = F.interpolate(torch.stack([self.terminal[0],self.terminal[2]],2), 6)
# 		lw_ra = F.interpolate(torch.stack([self.terminal[3],self.terminal[4]],2), 6)
# 		node_features = torch.cat([
# 								   (head_la[:,:,:3] + head_ra[:,:,:3])/2,
# 								   torch.stack((lw_ra[:,:,2], lw_ra[:,:,1], lw_ra[:,:,0]), 2),
# 								   lw_ra[:,:,3:], head_la[:,:,3:], head_ra[:,:,3:]], 2).to(x)
# 		x = torch.cat((x, node_features.permute(0,2,1).unsqueeze(1).repeat(1,32,1,1)), 3)

# 		concat_seq = torch.cat([x[:,:-2], x[:,1:-1], x[:,2:]], 2) # 1, 30, 45, 3
# 		multi_conv = self.gcl(self.graph, concat_seq)
# 		logit = self.conv(multi_conv)
		
# 		return logit
		



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.net_utils.tgcn import ConvTemporalGraphical
# from net.utils.graph import Graph

def  conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

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
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph,
                 edge_importance_weighting = True, **kwargs):
        super().__init__()

        # load graph
        self.graph = graph
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1)*2)
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels*2, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 2, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
        ))
        self.mm_fusion = MFA(in_channels * 2, 16)

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x,x1):
        x = torch.cat([x,x1],dim=1)
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        x = self.mm_fusion(x)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A
