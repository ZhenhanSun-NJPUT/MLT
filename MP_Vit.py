import torch
import torch.nn as nn
from functools import partial
import math
from itertools import repeat
from torch._six import container_abcs
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, kaiming_init)
import warnings

#from .helpers import load_pretrained
#from .layers import DropPath, to_2tuple, trunc_normal_

from ..builder import BACKBONES
from .base_backbone import BaseBackbone

#import scipy.misc
#import imageio
#from tensorboardX import SummaryWriter
#import torchvision.utils as vutils

#BatchNorm2d = nn.SyncBatchNorm
BatchNorm2d = nn.BatchNorm2d


class MultiheadAttention(nn.Module):
    """A warpper for torch.nn.MultiheadAttention.

    This module implements MultiheadAttention with residual connection.
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads. Same as
            `nn.MultiheadAttention`.
        attn_drop (float): A Dropout layer on attn_output_weights. Default 0.0.
        proj_drop (float): The drop out rate after attention. Default 0.0.
    """

    def __init__(self, embed_dims, num_heads, attn_drop=0.0, proj_drop=0.0):
        super(MultiheadAttention, self).__init__()
        assert embed_dims % num_heads == 0, 'embed_dims must be ' \
            f'divisible by num_heads. got {embed_dims} and {num_heads}.'
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop)
        self.dropout = nn.Dropout(proj_drop)

    def forward(self,
                x,
                key=None,
                value=None,
                residual=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None):
        """Forward function for `MultiheadAttention`.

        Args:
            x (Tensor): The input query with shape [num_query, bs,
                embed_dims]. Same in `nn.MultiheadAttention.forward`.
            key (Tensor): The key tensor with shape [num_key, bs,
                embed_dims]. Same in `nn.MultiheadAttention.forward`.
                Default None. If None, the `query` will be used.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Default None.
                If None, the `key` will be used.
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. Default None. If not None, it will
                be added to `x` before forward function.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Default None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`.
            attn_mask (Tensor): ByteTensor mask with shape [num_query,
                num_key]. Same in `nn.MultiheadAttention.forward`.
                Default None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_key].
                Same in `nn.MultiheadAttention.forward`. Default None.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        query = x
        if key is None:
            key = query
        if value is None:
            value = key
        if residual is None:
            residual = x
        if key_pos is None:
            if query_pos is not None and key is not None:
                if query_pos.shape == key.shape:
                    key_pos = query_pos
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        out = self.attn(
            query,
            key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]

        return residual + self.dropout(out)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'first_conv': '', 'classifier': 'head',
        **kwargs
    }
    
class FFN(nn.Module):
    """Implements feed-forward networks (FFNs) with residual connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Defaluts to 2.
        act_cfg (dict, optional): The activation config for FFNs.
        dropout (float, optional): Probability of an element to be
            zeroed. Default 0.0.
        add_residual (bool, optional): Add resudual connection.
            Defaults to False.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 num_fcs=2,
                 act_cfg=dict(type='GELU'),
                 dropout=0.0,
                 add_residual=True):
        super(FFN, self).__init__()
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        layers = nn.ModuleList()
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels),
                    self.activate, nn.Dropout(dropout)))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        self.layers = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.add_residual = add_residual
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # xavier_init(m, distribution='uniform')

                # Bias init is different from our API
                # therefore initialize them separately
                # The initialization is sync with ClassyVision
                nn.init.xavier_normal_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x, residual=None):
        """Forward function for `FFN`."""
        b,c,h,w = x.shape
        x = x.reshape(b , c, h * w).permute(0,2,1).contiguous()
        #print(self.embed_dims)
        out = self.layers(x)
        if not self.add_residual:
            return out
        if residual is None:
            residual = x
        
        x = residual + self.dropout(out)
        
        x = x.reshape(b , h , w , c).permute(0,3,1,2).contiguous()
        #print('type(x): ',type(x))
        return x

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(embed_dims={self.embed_dims}, '
        repr_str += f'feedforward_channels={self.feedforward_channels}, '
        repr_str += f'num_fcs={self.num_fcs}, '
        repr_str += f'act_cfg={self.act_cfg}, '
        repr_str += f'dropout={self.dropout}, '
        repr_str += f'add_residual={self.add_residual})'
        return repr_str

class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None, expansion=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, int(inplanes*expansion), kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(int(inplanes*expansion))
        self.conv2 = nn.Conv2d(int(inplanes*expansion), int(inplanes*expansion), kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=int(inplanes*expansion))
        self.bn2 = BatchNorm2d(int(inplanes*expansion))
        self.conv3 = nn.Conv2d(int(inplanes*expansion), planes, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes)
        self.relu = nn.Hardswish(inplace=True)
        #self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        if self.downsample is not None:
          self.downsample = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out
        
def ConvBNReLU(in_channels,out_channels,kernel_size,stride=1,padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,padding=padding),
        BatchNorm2d(out_channels),
        nn.Hardswish(inplace=True)
    )
def ConvBNReLU_group(in_channels,out_channels,kernel_size,stride=1,padding=0,group=4):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,padding=padding,groups=group),
        BatchNorm2d(out_channels),
        nn.Hardswish(inplace=True)
    )
def ConvBNReLUFactorization(in_channels,out_channels,kernel_sizes,paddings):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_sizes, stride=1,padding=paddings),
        BatchNorm2d(out_channels),
        nn.Hardswish(inplace=True)
    )

class InceptionV3Module_2(nn.Module):
    def __init__(self, in_channels):
        super(InceptionV3Module_2, self).__init__()
        self.branch0 = ConvBNReLU_group(in_channels=in_channels,out_channels=in_channels,kernel_size=1)
        
        self.branch1 = ConvBNReLU(in_channels=in_channels//4,out_channels=in_channels//4,kernel_size=1)

        self.branch2 = nn.Sequential(
           
            ConvBNReLU_group(in_channels=in_channels//4, out_channels=in_channels//4, kernel_size=5, padding=2,group = in_channels//4),
            ConvBNReLU(in_channels=in_channels//4, out_channels=in_channels//4, kernel_size=1),
         )#                                                                                     5          2

        self.branch3 = nn.Sequential(
           
            ConvBNReLU_group(in_channels=in_channels//4, out_channels=in_channels//4, kernel_size=5, padding=1,group = in_channels//4),
            ConvBNReLU_group(in_channels=in_channels//4, out_channels=in_channels//4, kernel_size=5, padding=1,group = in_channels//4),
            ConvBNReLU(in_channels=in_channels//4,out_channels=in_channels//4,kernel_size=1),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels=in_channels//4, out_channels=in_channels//4, kernel_size=1),
        )

    def forward(self, x):
        b,c,h,w = x.shape
        out = self.branch0(x)
        out = out.reshape(b , 4 , c//4 , h , w).permute(1, 0, 2, 3,4).contiguous()
        
        x0 = out[0]
        x1 = out[1]
        x2 = out[2]
        x3 = out[3]
        
        out1 = self.branch1(x0)
        out2 = self.branch2(x1)
        out3 = self.branch3(x2)
        out4 = self.branch4(x3)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out
        
class InceptionV3Module(nn.Module):
    def __init__(self, in_channels):
        super(InceptionV3Module, self).__init__()
        self.branch0 = ConvBNReLU_group(in_channels=in_channels,out_channels=in_channels*4,kernel_size=1,group = 4)
        
        self.branch1 = ConvBNReLU(in_channels=in_channels,out_channels=in_channels//4,kernel_size=1)

        self.branch2 = nn.Sequential(
           
            ConvBNReLU_group(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1,group = in_channels),
            ConvBNReLU(in_channels=in_channels, out_channels=in_channels//4, kernel_size=1),
         )#                                                                                     5          2

        self.branch3 = nn.Sequential(
           
            ConvBNReLU_group(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1,group = in_channels),
            ConvBNReLU_group(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1,group = in_channels),
            ConvBNReLU(in_channels=in_channels,out_channels=in_channels//4,kernel_size=1),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels=in_channels, out_channels=in_channels//4, kernel_size=1),
        )

    def forward(self, x):
        b,c,h,w = x.shape
        out = self.branch0(x)
        out = out.reshape(b , 4 , c , h , w).permute(1, 0, 2, 3,4).contiguous()
        
        x0 = out[0]
        x1 = out[1]
        x2 = out[2]
        x3 = out[3]
        
        out1 = self.branch1(x0)
        out2 = self.branch2(x1)
        out3 = self.branch3(x2)
        out4 = self.branch4(x3)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


class InceptionV3Module_1(nn.Module):
    def __init__(self, in_channels):
        super(InceptionV3Module_1, self).__init__()
        self.branch0 = ConvBNReLU_group(in_channels=in_channels,out_channels=in_channels,kernel_size=1)
        
        self.branch1 = ConvBNReLU(in_channels=in_channels,out_channels=in_channels//4,kernel_size=1)

        self.branch2 = nn.Sequential(
           
            ConvBNReLU_group(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1,group = in_channels),
            ConvBNReLU(in_channels=in_channels, out_channels=in_channels//4, kernel_size=1),
         )#                                                                                     5          2

        self.branch3 = nn.Sequential(
           
            ConvBNReLU_group(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1,group = in_channels),
            ConvBNReLU_group(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1,group = in_channels),
            ConvBNReLU(in_channels=in_channels,out_channels=in_channels//4,kernel_size=1),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels=in_channels, out_channels=in_channels//4, kernel_size=1),
        )

    def forward(self, x):
        b,c,h,w = x.shape
        out = self.branch0(x)
        #out = out.reshape(b , 4 , c , h , w).permute(1, 0, 2, 3,4).contiguous()
        
        x0 = out
        x1 = out
        x2 = out
        x3 = out
        
        out1 = self.branch1(x0)
        out2 = self.branch2(x1)
        out3 = self.branch3(x2)
        out4 = self.branch4(x3)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out        
        
def to_2tuple(x):
    if isinstance(x, container_abcs.Iterable):
        return x
    return tuple(repeat(x, 2))


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + \
            torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class MPSM(nn.Module):
    def __init__(self,H,W,dim,stage):
        super().__init__()
        self.H = H
        self.W = W
        self.dim = dim
        self.stage = stage
        self.norm = nn.LayerNorm(self.dim)
        if stage == 1:
          self.pooling_1_1 = nn.AdaptiveAvgPool2d(1)
          self.pooling_2_2 = nn.AdaptiveAvgPool2d(2)#1 4 16 64 256
          self.pooling_4_4 = nn.AdaptiveAvgPool2d(4)
          self.pooling_8_8 = nn.AdaptiveAvgPool2d(8)
          self.pooling_16_16 = nn.AdaptiveAvgPool2d(16)
        if stage == 2:
          self.pooling_1_1 = nn.AdaptiveAvgPool2d(1)
          self.pooling_2_2 = nn.AdaptiveAvgPool2d(2)
          self.pooling_4_4 = nn.AdaptiveAvgPool2d(4)#3
          self.pooling_8_8 = nn.AdaptiveAvgPool2d(8)
          self.pooling_16_16 = nn.AdaptiveAvgPool2d(16)
        if stage == 3:
          self.pooling_1_1 = nn.AdaptiveAvgPool2d(1)
          self.pooling_2_2 = nn.AdaptiveAvgPool2d(2)
          self.pooling_4_4 = nn.AdaptiveAvgPool2d(4)
          self.pooling_8_8 = nn.AdaptiveAvgPool2d(8)
          self.pooling_16_16 = nn.AdaptiveAvgPool2d(16)
        if stage == 4:
          self.pooling_1_1 = nn.AdaptiveAvgPool2d(1)
          self.pooling_2_2 = nn.AdaptiveAvgPool2d(2)
          self.pooling_4_4 = nn.AdaptiveAvgPool2d(3)
          self.pooling_8_8 = nn.AdaptiveAvgPool2d(4)
          #self.pooling_16_16 = nn.AdaptiveAvgPool2d(4)
        
        self.proj1 = nn.Conv2d(dim, dim, kernel_size=3, stride = 1, groups = self.dim, padding = 1)
        self.proj2 = nn.Conv2d(dim, dim, kernel_size=3, stride = 1, groups = self.dim, padding = 1)
        self.proj3 = nn.Conv2d(dim, dim, kernel_size=3, stride = 1, groups = self.dim, padding = 1)
        self.proj4 = nn.Conv2d(dim, dim, kernel_size=3, stride = 1, groups = self.dim, padding = 1)
        self.proj5 = nn.Conv2d(dim, dim, kernel_size=3, stride = 1, groups = self.dim, padding = 1)        
        
        self.flatten = nn.Flatten(2,3)

    def forward(self, x):
        
        B,C,H,W = x.shape
        if self.stage == 4:
          x2 = self.pooling_8_8(x) + self.proj1(self.pooling_8_8(x))
          x3 = self.pooling_4_4(x) + self.proj2(self.pooling_4_4(x))
          x4 = self.pooling_2_2(x) + self.proj3(self.pooling_2_2(x))
          x5 = self.pooling_1_1(x) + self.proj4(self.pooling_1_1(x))
          
          x2 = x2.reshape(B,C,-1)
          x3 = x3.reshape(B,C,-1)
          x4 = x4.reshape(B,C,-1)
          x5 = x5.reshape(B,C,-1)
          
          x = torch.cat((x2,x3,x4,x5),2)
        
          x = x.permute(0, 2, 1).contiguous()
        
          x = self.norm(x)
        else:  
          x1 = self.pooling_16_16(x) + self.proj1(self.pooling_16_16(x))
          x2 = self.pooling_8_8(x) + self.proj2(self.pooling_8_8(x))
          x3 = self.pooling_4_4(x) + self.proj3(self.pooling_4_4(x))
          x4 = self.pooling_2_2(x) + self.proj4(self.pooling_2_2(x))
          x5 = self.pooling_1_1(x) + self.proj5(self.pooling_1_1(x))      
        
          x1 = x1.reshape(B,C,-1)
          x2 = x2.reshape(B,C,-1)
          x3 = x3.reshape(B,C,-1)
          x4 = x4.reshape(B,C,-1)
          x5 = x5.reshape(B,C,-1)          
        
          x = torch.cat((x1,x2,x3,x4,x5),2)
        
          x = x.permute(0, 2, 1).contiguous()
        
          x = self.norm(x)
        
        
        return x
        
class Attention(nn.Module):
    #i = 0
    #j = 0
    #k = 0
    def __init__(self, dim, q_bias=False, kv_bias=False , qk_scale=384, attn_drop=0., proj_drop=0., H = 48, W = 48, reduce_dim = 2,stage = 1):
        super().__init__()
        
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale ** -0.5
        self.kv = nn.Linear(dim//reduce_dim,2*dim//reduce_dim, bias = kv_bias)
        self.q = nn.Linear(dim//reduce_dim, dim//reduce_dim, bias=q_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.H = H
        self.W = W 
        self.act_layer = nn.Sigmoid()
        if stage != 1:
          self.proj_reduce_dim = nn.Conv2d(dim, dim//reduce_dim, kernel_size=1, stride = 1)
        #self.proj_reduce_dim = nn.Linear(dim, dim//2, bias = True)
          self.proj = nn.Conv2d(dim//reduce_dim, dim, kernel_size=1, stride = 1)
        #self.proj = nn.Linear(dim//reduce_dim, dim, bias=q_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.MPSM = MPSM(H , W, dim//reduce_dim,stage)
        self.reduce_dim = reduce_dim 
        self.stage = stage
        ######self.R = [0 , 3069, 1364, 341, 30]
        #######self.relative_position_bias_table = nn.Parameter(torch.zeros(H * W, self.R[stage]))  # 2*Wh-1 * 2*Ww-1, nH
        #print(self.relative_position_bias_table.shape)
        # get pair-wise relative position index for each token inside the window
        
             
    def forward(self, x):
        if self.stage != 1:
          x = self.proj_reduce_dim(x)
        
        B,C1,H,W = x.shape
        
        x1 = self.MPSM(x)
        B1,N1,b = x1.shape 
        
        k, v = self.kv(x1).reshape(B1, N1, 2, C1).permute(2, 0, 1, 3).contiguous()
        q = self.q(x.reshape(B,C1,self.H*self.W).permute(0,2,1).contiguous())
        
        #q1 = self.q1(x.reshape(B,C1,self.H*self.W).permute(0,2,1).contiguous())#
        #q1 = q1 *  self.scale#
        #q1 = q1.softmax(dim=1)#
        #q1 = q1.transpose(-2, -1)@q#
        #print(q1.shape)
        #q = q1.repeat(1,H*W,1)#
        #print(q.shape) 
            
        ##############################
        ##attn = (q @ k.transpose(-2, -1)).contiguous() * self.scale
        ##relative_position_bias = self.relative_position_bias_table.unsqueeze(0)
        #relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ##attn = attn + relative_position_bias
        #attn = self.softmax(attn)
        ###########################################################
        ##########
        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        #############
        #if self.stage != 4:
        #  attn1 = self.q2(attn)#
        #else:
        #  attn1 = self.q3(attn)#
        #attn1 = attn1 *  self.scale#
        #attn1 = attn1.softmax(dim=1)#
        #attn1 = attn1.transpose(-2, -1)@attn#
        #attn = attn1.repeat(1,H*W,1)#
       
        attn = attn.softmax(dim=-1)#################
       
        
        
        ##if Attention.k == 1 : 
          ##Attention.j = Attention.j + 1
          ##attn1 = attn.reshape(B,H,W,N1).permute(0,3,1,2).contiguous()
          ##attn1 = (attn1-attn1.min())/((attn1.max()-attn1.min()))
          #attn1 = self.act_layer(attn1)
          #attn1 = self.m(attn1)
          
          ##for bidx in range(attn1.size(0)):
            ##for channel in range(attn1.size(1)):
              ##PATH = "/data1/szh/work_dir3/2021_7_32_MSPM_6222/feature_map/{}_{}_{}_1_{}_{}.jpg".format(Attention.i,Attention.j,bidx,Attention.j%12,channel)
              ##imageio.imsave(PATH, (attn1[bidx,channel,:,:].cpu().detach().numpy()*255))
         
        attn = self.attn_drop(attn)
       
        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, H*W, C1)
        x = x.reshape(B, H ,W , C1).permute(0,3,1,2).contiguous()
        if  self.stage != 1:
          x = self.proj(x)##
        x = self.proj_drop(x)
        return x

 
class Block(nn.Module):

    def __init__(self, dim, mlp_ratio=0.25, q_bias=False, kv_bias=False ,qk_scale=384, drop=0., attn_drop=0.,drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, H=48, W=48,stage = 1,norm_cfg=dict(type='LN'),act_cfg=dict(type='GELU')):
        super().__init__()
        ##########################################
        #SHSA
        self.act_layer = nn.GELU()
        self.H = H
        self.W = W
        #self.norm3 = BatchNorm2d(dim)
        if dim <=128 :
           self.norm4 = BatchNorm2d(dim)# norm_layer(dim//2)
           self.attn = Attention(dim, q_bias=q_bias, kv_bias=kv_bias , qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,H = self.H ,W = self.W,reduce_dim = 1,stage = stage ) 
        elif dim <= 256:
           self.norm4 =  BatchNorm2d(dim)#norm_layer(dim//2)
           self.attn = Attention(dim, q_bias=q_bias, kv_bias=kv_bias , qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,H = self.H ,W = self.W ,reduce_dim = 2,stage = stage) 
           
        elif dim <= 512 :
           self.norm4 =  BatchNorm2d(dim)#norm_layer(dim//4)
           self.attn = Attention(dim, q_bias=q_bias, kv_bias=kv_bias , qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,H = self.H ,W = self.W ,reduce_dim = 4,stage = stage) 
           
        elif dim <= 1024 :
           self.norm4 =   BatchNorm2d(dim)#norm_layer(dim//8)
           self.attn = Attention(dim, q_bias=q_bias, kv_bias=kv_bias , qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,H = self.H ,W = self.W ,reduce_dim = 8,stage = stage) 
           
        self.dim = dim
#        
#            
#        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = BatchNorm2d(dim)#norm_layer(dim)
        
        ######################
        #CIM
        #self.InceptionV3 = InceptionV3Module(in_channels=dim)
        ######################
        ######################
        #MLP
        self.MLP = FFN(dim, 4*dim, 2, dict(type='GELU'), drop_path)
        ######################
        #######################################
        #######################################
        #MHSA
#        self.norm1_name, norm1 = build_norm_layer(
#            dict(type='LN'), dim, postfix=1)
#        self.add_module(self.norm1_name, norm1)
#        if stage != 3 :
#            self.attn = MultiheadAttention(
#            dim,
#            num_heads=stage*2,
#            attn_drop=attn_drop,
#            proj_drop=drop)
#        elif stage == 3 :
#            self.attn = MultiheadAttention(
#            dim,
#            num_heads= 8,
#            attn_drop=attn_drop,
#            proj_drop=drop)
#        self.norm2_name, norm2 = build_norm_layer(
#            dict(type='LN'), dim, postfix=2)
#        self.add_module(self.norm2_name, norm2)
        #self.mlp = FFN(dim, 4*dim, 2, dict(type='GELU'),
        #               drop)
        #########################################
#    @property
#    def norm1(self):
#        return getattr(self, self.norm1_name)
#
#    @property
#    def norm2(self):
#        return getattr(self, self.norm2_name)
#        
        
    def forward(self, x):
    
        #print('type(x): ',type(x))
        #print(x.shape)
        ############################
        #MHSA
        #print(x.shape)
#        b,c,h,w = x.shape
#        x = x.reshape(b,c,h*w).permute(0, 2, 1)
#        norm_x = self.norm1(x)
#        
#        x = x.permute(1, 0, 2)
#        norm_x = norm_x.permute(1, 0, 2)
#        x = self.attn(norm_x, residual=x)
#        
#        x = x.permute(1, 0, 2)
#        x = x.reshape(b,h,w,c).permute(0, 3 , 1 , 2)
        #MLP
        #x = self.mlp(self.norm2(x), residual=x)
        #CIM
        #x = x + self.drop_path(self.InceptionV3(self.norm3(x)))
        #print(x.shape)
        ############################
        ############################
        #SHSA
        x = x + self.drop_path(self.attn(self.norm2(x)))
        #CIM
        #x = x + self.drop_path(self.InceptionV3(self.norm4(x)))
        #MLP
        x = self.MLP(self.norm4(x))
        ###############################
        return x 
              
class BasicStem(nn.Module):
    def __init__(self, in_ch=3, out_ch=64, with_pos=False):
        super(BasicStem, self).__init__()
        hidden_ch = out_ch // 2
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = BatchNorm2d(hidden_ch)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = BatchNorm2d(hidden_ch)
        self.conv3 = nn.Conv2d(hidden_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)

        #self.act = nn.ReLU(inplace=True)
        self.act = nn.Hardswish(inplace=True)
        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)

        x = self.conv3(x)
        if self.with_pos:
            x = self.pos(x)
        return x       
  
        
class PA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))




class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, with_pos=False):
        super().__init__()
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = BatchNorm2d(embed_dim)
        
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x)
        x = self.norm(x)
        if self.with_pos:
            x = self.pos(x)
        #x = x.flatten(2).transpose(1, 2)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x,(H, W)

@BACKBONES.register_module()
class MP_VIT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=384, in_chans=3, embed_dims=[128, 256, 512,1024], depths=[2, 2, 6, 2], num_classes=19, mlp_ratios=[0.25, 0.25, 0.25, 0.25], q_bias=True, kv_bias=True, qk_scale=[ 64, 128, 256,512], drop_rate=0.1, attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(BatchNorm2d, eps=1e-6), norm_cfg=None,out_features=['stage1','stage2','stage3','stage4'],norm_eval=False, **kwargs):
        super(MP_VIT, self).__init__(**kwargs)
        self.img_size = img_size
        self.in_chans = in_chans
        self.embed_dim = embed_dims
        self.depths = depths
        self.mlp_ratio = mlp_ratios
        self.q_bias = q_bias
        self.kv_bias = kv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.norm_cfg = norm_cfg
        self.num_stages = self.depths
        #self.out_indices = tuple(range(self.num_stages))
        
        self.num_layers = len(depths)
        self._out_features = out_features
        self.norm_eval=norm_eval

        
        
        self.stem = BasicStem(in_ch=in_chans, out_ch=embed_dims[0], with_pos=True)


        self.patch_embed_2 = PatchEmbed(img_size = self.img_size//4, patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1], with_pos=True)
        self.patch_embed_3 = PatchEmbed(img_size = self.img_size//8, patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2], with_pos=True)
        self.patch_embed_4 = PatchEmbed(img_size = self.img_size//16,patch_size=2, in_chans=embed_dims[2], embed_dim=embed_dims[3], with_pos=True)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.stage1 = nn.ModuleList([
            Block(embed_dims[0], mlp_ratios[0], q_bias, kv_bias,qk_scale[0], drop_rate, attn_drop_rate,
                  drop_path=dpr[cur+i], norm_layer=norm_layer, H = self.img_size//4,W = self.img_size//4,stage = 1)
            for i in range(self.depths[0])])

        cur += depths[0]
        self.stage2 = nn.ModuleList([
            Block(embed_dims[1], mlp_ratios[1], q_bias, kv_bias, qk_scale[1], drop_rate, attn_drop_rate,
                  drop_path=dpr[cur+i], norm_layer=norm_layer, H = self.img_size//8, W = self.img_size//8,stage = 2)
            for i in range(self.depths[1])])

        cur += depths[1]
        self.stage3 = nn.ModuleList([
              Block(embed_dims[2], mlp_ratios[2], q_bias, kv_bias, qk_scale[2], drop_rate, attn_drop_rate,
                  drop_path=dpr[cur+i], norm_layer=norm_layer, H = self.img_size//16, W = self.img_size//16,stage = 3)
            for i in range(self.depths[2])])

        cur += depths[2]
        self.stage4 = nn.ModuleList([
              Block(embed_dims[3], mlp_ratios[3], q_bias, kv_bias, qk_scale[3], drop_rate, attn_drop_rate,
                  drop_path=dpr[cur+i], norm_layer=norm_layer, H = self.img_size//32, W = self.img_size//32,stage = 4)
            for i in range(self.depths[3])])
            
######################################################################
        # add a norm layer for each output
        for i in range(self.num_layers-1):
            stage = f'stage{i+1}'
            if stage in self._out_features:
                layer = norm_layer(embed_dims[i])
                layer_name = f'norm{i+1}'
                self.add_module(layer_name, layer)

        self.norm = norm_layer(embed_dims[3])
     

        #trunc_normal_(self.pos_embed, std=.02)
        #trunc_normal_(self.cls_token, std=.02)
        #self.apply(self._init_weights)

    def init_weights(self,pretrained=None):
        # nn.init.normal_(self.pos_embed, std=0.02)
        # nn.init.zeros_(self.cls_token)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
                
    @property
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def _conv_filter(self, state_dict, patch_size=16):
        """ convert patch embedding weight from manual patchify + linear proj to conv"""
        out_dict = {}
        for k, v in state_dict.items():
            if 'patch_embed.proj.weight' in k:
                v = v.reshape((v.shape[0], 3, patch_size, patch_size))
            out_dict[k] = v
        return out_dict

    def to_2D(self, x):
        n, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        x = x.transpose(1, 2).reshape(n, c, h, w)
        return x

    def to_1D(self, x):
        n, c, h, w = x.shape
        x = x.reshape(n, c, -1).transpose(1, 2)
        return x

    def forward(self, x):
       

    
        #Attention.i += 1
        #if (not Attention.i%50) : 
          #x1 = x
          #Attention.k = 1
          #i = Attention.i
          #for bidx in range(x.size(0)):
              #PATH = "/data1/szh/work_dir3/2021_7_32_MSPM_6222/feature_map/{}_X_{}.jpg".format(i,bidx)
              #imageio.imsave(PATH, x1[bidx].permute(1,2,0).contiguous().cpu().detach().numpy())
              
        #outputs = []
        x = self.stem(x)
        B, _, H, W = x.shape
        #x = x.flatten(2).permute(0, 2, 1)

        # stage 1
        for blk in self.stage1:
            x = blk(x)
        #if "stage1" in self._out_features:
        #    x = self.norm1(x)
        #    x = x#.permute(0, 2, 1).reshape(B, -1, H, W)
        #    outputs.append(x)
        #else:
        #    x = x#.permute(0, 2, 1).reshape(B, -1, H, W)

        # stage 2
        #print(3)
        x, (H, W) = self.patch_embed_2(x)
        #print(4)
        for blk in self.stage2:
            x = blk(x)
        #if "stage2" in self._out_features:
        #    x = self.norm2(x)
        #    x = x#.permute(0, 2, 1).reshape(B, -1, H, W)
        #    outputs.append(x)
        #else:
        #    x = x#.permute(0, 2, 1).reshape(B, -1, H, W)

        # stage 3
        x, (H, W) = self.patch_embed_3(x)
        for blk in self.stage3:
            x = blk(x)
        #if "stage3" in self._out_features:
            #x = self.norm3(x)
            #x = x#.permute(0, 2, 1).reshape(B, -1, H, W)
            #outputs.append(x)
        #else:
            #x = x#.permute(0, 2, 1).reshape(B, -1, H, W)
        #print(2)
        # stage 4
        x, (H, W) = self.patch_embed_4(x)
        for blk in self.stage4:
            x = blk(x)
        x = self.norm(x)
        x = x.reshape(B, -1, H*W)
        #if "stage4" in self._out_features:
            #outputs.append(x)
        #Attention.j = 0
        #Attention.k = 0
        #long running
        #print(1)
        return x#outputs
    def train(self, mode=True):
        super(MP_VIT, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    m.eval()