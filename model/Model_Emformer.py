import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from models.MA_Former import enhanced_transformer
import torch.nn.functional as F
from torch.nn import MultiheadAttention as MHSA
from mamba_ssm import Mamba
import torch.fft as fft

def blc_to_bchw1(x: torch.Tensor, x_size: tuple) -> torch.Tensor:
    """Rearrange a tensor from the shape (B, L, C) to (B, C, H, W)."""
    B, L, C = x.shape
    return x.transpose(1, 2).view(B, C, *x_size)

class Complex_FFT(nn.Module):
    def __init__(self):
        super(Complex_FFT, self).__init__()

    def forward(self, x):
        # Complex FFT Transform
        x_fft = fft.fft2(x, dim=(-2, -1)) 
        real = x_fft.real  
        imag = x_fft.imag  
        return real, imag


class Complex_IFFT(nn.Module):
    def __init__(self):
        super(Complex_IFFT, self).__init__()

    def forward(self, real, imag):       
        x_complex = torch.complex(real, imag)
        x_ifft = fft.ifft2(x_complex, dim=(-2, -1))
        return x_ifft.real


class Conv1x1(nn.Module):
    def __init__(self, in_channels):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, padding=0,groups=in_channels*2)

    def forward(self, x):
        return self.conv(x)


class E_fft(nn.Module):
    def __init__(self, in_channels):
        super(E_fft, self).__init__()
        self.complex_fft = Complex_FFT()
        self.conv1x1 = Conv1x1(in_channels)
        self.complex_ifft = Complex_IFFT()

    def forward(self, x):
        real, imag = self.complex_fft(x)
        assemble = torch.cat([real, imag], dim=1)
        c_out = self.conv1x1(assemble)
        channel_out = c_out.shape[1] // 2
        e_real = c_out[:, :channel_out, :, :]
        e_imag = c_out[:, channel_out:, :, :]
        output = self.complex_ifft(e_real, e_imag)

        return output

class MTF(nn.Module):
    def __init__(self, hidden_size, d_state=64, d_conv=4, expand=4, mha_head=8):
        super(MTF, self).__init__()
        self.mamba = Mamba(
            d_model=hidden_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.mhsa = MHSA(hidden_size, mha_head, )
        self.l_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size * 4), nn.GELU(),
                                 nn.Linear(hidden_size * 4, hidden_size))

    def forward(self, x):
        y = self.l_norm(x)
        y, _ = self.mhsa(y, y, y)
        x = x + y
        y = self.l_norm(x)
        y = self.mamba(y)
        x = x + y
        y = self.l_norm(x)
        y = self.mlp(y)
        x = x + y
        return x


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, drop_path=0.):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, droout=0.,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.mlp = PreNorm(planes, FeedForward(planes, 2*planes, dropout=dropout))
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        out1 = out
        out = self.mlp(out) + out1

        return out


class strip_att(nn.Module):
    def __init__(self, dim, group, kernel) -> None:
        super().__init__()
        self.H_spatial_att = SSA(dim, group=group, kernel=kernel)
        self.W_spatial_att = SSA(dim, group=group, kernel=kernel, H=False)
        self.gamma = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        out = self.H_spatial_att(x)
        out = self.W_spatial_att(out)
        return self.gamma * out + x * self.beta

class SSA(nn.Module):
    def __init__(self, dim, kernel=5, group=2, H=True) -> None:
        super().__init__()
        self.k = kernel
        pad = kernel // 2
        self.kernel = (1, kernel) if H else (kernel, 1)
        self.padding = (kernel // 2, 1) if H else (1, kernel // 2)
        self.group = group
        self.pad = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))
        self.conv = nn.Conv2d(dim, group * kernel, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.f_act = nn.Sigmoid()

    def forward(self, x):
        f = self.pool(x)
        f = self.conv(f)
        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel).reshape(n, self.group, c // self.group, self.k, h * w
        n, c1, p, q = f.shape
        f = f.reshape(n, c1 // self.k, self.k, p * q).unsqueeze(2)
        f = self.f_act(f)
        out = torch.sum(x * f, dim=3).reshape(n, c, h, w)
        return out
        
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, dim=512,
                  groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, drop_path = 0.):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])                         
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.e_former = enhanced_transformer()
        self.fc = nn.Linear(512 * block.expansion, num_classes)#
        self.mtf= MTF(64)
        self.mtf1 = MTF(128)
        self.mtf2 = MTF(256)
        self.mtf3 = MTF(512)
        self.sa = strip_att(64,group=1,kernel=5)
        self.sa01 = strip_att(64, group=1, kernel=3)
        self.sa1 = strip_att(128, group=1, kernel=5)
        self.sa11 = strip_att(128, group=1, kernel=3)
        self.sa2 = strip_att(256, group=1, kernel=5)
        self.sa21 = strip_att(256, group=1, kernel=3)
        self.e_fft = E_fft(in_channels=64)
        self.e_fft1 = E_fft(in_channels=128)
        self.e_fft2 = E_fft(in_channels=256)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x1 = self.sa(x)
        x11 = self.sa01(x)
        x1 = self.e_fft(x1+x11)
        x2 = x.flatten(2).permute(0, 2, 1)
        x2 = self.mtf(x2)
        x2 = blc_to_bchw1(x2, (56, 56)).contiguous()
        x = x1 + x2
        x = self.layer2(x)
        x3 = self.sa1(x)
        x31 =self.sa11(x)
        x3 = self.e_fft1(x3+x31)
        x4 = x.flatten(2).permute(0, 2, 1)
        x4 = self.mtf1(x4)
        x4 = blc_to_bchw1(x4, (28, 28)).contiguous()
        x = x3 + x4
        x = self.layer3(x)
        x5 = self.sa2(x)
        x51= self.sa21(x)
        x5 = self.e_fft2(x5+x51)
        x6 = x.flatten(2).permute(0, 2, 1)
        x6 =self.mtf2(x6)
        x6 = blc_to_bchw1(x6, (14,14)).contiguous()
        x = x5 + x6
        x = self.layer4(x)
        b_l,c, h,w = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        x = self.mtf3(x)
        x = x.permute(0, 2, 1).view(b_l, c, h, w).contiguous()
        x = self.e_former(x)
        f = x.mean(dim=1)
        out = self.fc(f)
        return out

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if not pretrained == '':
        print(f'[!] initializing model with "{pretrained}" weights ...')
        if pretrained == 'imagenet':
            state_dict = load_state_dict_from_url(model_urls[arch],
                                                  progress=progress)
        elif pretrained == 'msceleb':
            msceleb_model = torch.load('./models/resnet18_msceleb.pth')
            state_dict = msceleb_model['state_dict']
        else:
            raise NotImplementedError('wrong pretrained model!')
        model.load_state_dict(state_dict, strict=False)
    return model

def FER_EMFormer(pretrained=False, progress=True, **kwargs):
    return _resnet('FER_EMFormer', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)




