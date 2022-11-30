import torch
import torch.nn as nn
import logging

from decomposition import tucker
from vbmf import rank_selection

_logger = logging.getLogger('tucker')
_logger.addHandler(logging.StreamHandler())
_logger.setLevel(logging.DEBUG)


class TuckerLayer(nn.Module):

    def __init__(self, in_channels, r2, r1, out_channels, kernel_size, stride, padding,
                 weight=None, bias=None):
        """
        init tucker layer
        Parameters
        ----------
        in_channels: number of input channels
        r2: reduced number of input channels
        r1: reduced number of output channels
        out_channels: number of output channels
        kernel_size: size of the convolution kernel
        stride: stride of the convolution
        padding: padding of the convolution
        weight: if provided, load the weights into the three convolution layers. Default: None
        bias: if provided, load the bias into the last convolution layer. Default: None
        """
        super().__init__()
        conv1 = nn.Conv2d(in_channels=in_channels, out_channels=r2, kernel_size=1,
                          stride=1, padding=0, bias=False)
        conv2 = nn.Conv2d(in_channels=r2, out_channels=r1, kernel_size=kernel_size,
                          stride=stride, padding=padding, bias=False)
        conv3 = nn.Conv2d(in_channels=r1, out_channels=out_channels, kernel_size=1,
                          stride=1, padding=0, bias=True)

        if weight is not None:
            conv1.weight.data, conv2.weight.data, conv3.weight.data = weight
        if bias is not None:
            conv3.bias.data = bias

        self.tucker_conv = nn.Sequential(conv1, conv2, conv3)

    @classmethod
    def from_Conv2D(cls, conv: nn.Conv2d, rank=None, method='HOOI'):
        """
        return a tucker layer decomposed from the input convolutional layer
        Parameters
        ----------
        conv: input convolutional layer
        rank: ranks of the factor matrices
        method: algorithm of tucker decomposition - 'HOSVD' or 'HOOI'

        Returns
        -------

        """
        if rank is None:
            original_weight = conv.weight.cpu()
            rank = rank_selection(original_weight.data.detach().numpy())
            rank = max(rank[0], 1), max(rank[1], 1)
            _logger.info('Tucker Decompose with rank {}'.format(rank))
        core, factors = tucker(tensor=conv.weight.data, rank=rank, modes=(0, 1), method=method)
        weight1 = torch.transpose(factors[1], 0, 1).unsqueeze(2).unsqueeze(3)
        weight2 = core
        weight3 = factors[0].unsqueeze(2).unsqueeze(3)
        weight = (weight1, weight2, weight3)

        return cls(conv.in_channels, rank[1], rank[0], conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                   weight, conv.bias)

    def forward(self, x):
        return self.tucker_conv(x)

    def factor_matrix(self):
        """
        get the two factor matrices of the tucker layer
        Returns
        -------

        """
        weight1 = torch.transpose(self.tucker_conv[0].weight.squeeze(3).squeeze(2), 0, 1)
        weight3 = self.tucker_conv[2].weight.squeeze(3).squeeze(2)
        return weight1, weight3

    def orthogonal_error(self, ord='fro'):
        """
        get the orthogonal error |U^TU-I| + |V^TV-I|
        Parameters
        ----------
        ord: chosen matrix norm, see https://pytorch.org/docs/stable/generated/torch.linalg.matrix_norm.html
             two common practice: ‘fro’: frobenius norm, default; 2: largest singular value
        Returns
        -------

        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        weight1, weight3 = self.factor_matrix()
        error1 = torch.linalg.matrix_norm(torch.matmul(weight1.T, weight1) - torch.eye(weight1.shape[1]).to(device), ord=ord)
        error3 = torch.linalg.matrix_norm(torch.matmul(weight3.T, weight3) - torch.eye(weight3.shape[1]).to(device), ord=ord)
        return error1 + error3
