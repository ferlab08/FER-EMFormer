import torch
from einops import rearrange
from torch import nn, einsum
from typing import Tuple

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class StarReLU(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

def blc_to_bchw(x: torch.Tensor, x_size: Tuple) -> torch.Tensor:
    """Rearrange a tensor from the shape (B, L, C) to (B, C, H, W)."""
    B, L, C = x.shape
    return x.transpose(1, 2).view(B, C, *x_size)


class HybConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.conv_3 = nn.Conv2d(
            in_channels // 2,
            out_channels // 2,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=bias,
        )

        self.conv_5 = nn.Conv2d(
            in_channels - in_channels // 2,
            out_channels - out_channels // 2,
            kernel_size + 2,
            stride=stride,
            padding=padding + 1,
            dilation=dilation,
            groups=1,
            bias=bias,
        )

    def forward(self, x) :
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.conv_3(x1)
        x2 = self.conv_5(x2)
        x = torch.cat([x1, x2], dim=1)
        return x

class MAFN(nn.Module):

    def __init__(self, in_channels, act_ratio=0.25, act_fn=StarReLU, gate_act=nn.Sigmoid):
        super().__init__()
        reduce_channels = int(in_channels * act_ratio)
        self.norm = nn.LayerNorm(in_channels)
        self.g_reduce = nn.Conv2d(in_channels, reduce_channels, 1, bias=False)
        self.l_reduce = nn.Conv2d(in_channels, reduce_channels, 1, bias=False)
        self.act_fn = act_fn()
        self.channel_dim = nn.Linear(reduce_channels, in_channels)
        self.spatial_dim = nn.Linear(reduce_channels * 3, 1)
        self.gate_act = gate_act()
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = HybConv2d(
            reduce_channels,
            reduce_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=reduce_channels,
            dilation=1,
            bias=True,
        )

    def forward(self, x):
        x_orig = x
        x_size = (7, 7)
        x = blc_to_bchw(x, x_size).contiguous()
        x_g1 = self.max_pool(x)
        x_g2 = self.avg_pool(x)
        x_g1 = self.act_fn(self.g_reduce(x_g1))
        x_g2 = self.act_fn(self.g_reduce(x_g2))
        b_l, c, h, w = x_g1.shape
        x1 = x_g1.reshape((b_l, c, h * w)).permute(0, 2, 1)
        b_l, c, h, w = x_g2.shape
        x2 = x_g2.reshape((b_l, c, h * w)).permute(0, 2, 1)
        x_l = self.l_reduce(x)
        x_l = self.act_fn(self.conv(x_l))
        b_l, c, h, w = x_l.shape
        x_l = x_l.reshape((b_l, c, h * w)).permute(0, 2, 1)
        c_aggr1 = self.channel_dim(x1)
        c_aggr2 = self.channel_dim(x2)
        c_aggr = c_aggr1 + c_aggr2
        c_aggr = self.gate_act(c_aggr)
        s_aggr = torch.cat([x_l, x1.expand(-1, x_orig.shape[1], -1), x2.expand(-1, x_orig.shape[1], -1)], dim=-1)
        s_aggr = self.spatial_dim(s_aggr)
        s_aggr = self.gate_act(s_aggr)
        aggr = c_aggr * s_aggr
        return x_orig * aggr

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = MAFN(dim)

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim),
                                    nn.Dropout(dropout)) if project_out else nn.Identity()
        self.aggr_drop = nn.Dropout(dropout)

    def forward(self, x):

        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        aggr = dots.softmax(dim=-1)
        aggr = self.aggr_drop(aggr)#
        out = einsum('b h i j, b h j d -> b h i d', aggr, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class ETransformer(nn.Module):
    def __init__(self, dim, depth, depths, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                                              Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))]))

    def forward(self, x):
        b_l, c, h, w = x.shape
        x = x.reshape((b_l, c, h*w))
        x = x.permute(0, 2, 1)
        for aggr, ff in self.layers:
            x = aggr(x)
            x = ff(x)
        return x

class MAT(nn.Module):
    def __init__(self,  dim=512, depth=3, depths=1, heads=8, mlp_dim=1024, dim_head=64, dropout=0.0):
        super().__init__()

        self.transformer = ETransformer(dim, depth, depths, heads, dim_head, mlp_dim, dropout)
    def forward(self, x):
        x = self.transformer(x)
        return x

def enhanced_transformer():
    return MAT()

# if __name__ == '__main__':
#     img = torch.randn((1, 16, 3, 224, 224))#224 224
#     model = enhanced_transformer()
#     model(img)
