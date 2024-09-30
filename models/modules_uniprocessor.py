import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from torch import _assert


# LayerNorm
class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        

# LayerNorm2d
class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


# Input Projection
class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, bias=True):
        super().__init__()
        self.proj = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


# Output Projection
class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, bias=True):
        super().__init__()
        self.proj = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


# Downsample Block
class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        
        self.body = nn.Sequential(nn.Conv2d(in_channel, out_channel//4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        out = self.body(x)
        return out


# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(in_channel, out_channel*4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))
        
    def forward(self, x):
        out = self.body(x)
        return out
    

# Global Response Normalization Block
class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


# Global Response Normalization Block
class GRN_channel_first(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2,3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


# Gate Block
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
class GeluGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return F.gelu(x1) * x2
    

# ########################################
# Main Blocks
# ########################################


# Channel Attention Block
class ChannelAttention(nn.Module):
    def __init__(self, dim, channel_expansion_ratio=1, bias=True, gate='simple', norm_layer=nn.LayerNorm):
        super(ChannelAttention, self).__init__()
        hidden_features = int(dim*channel_expansion_ratio)
        self.norm = norm_layer(dim)
        self.conv1 = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.conv2 = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.conv3 = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=hidden_features, out_channels=hidden_features, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        # SimpleGate
        if gate == 'gelu':
            self.sg = GeluGate()
        else:
            self.sg = SimpleGate()

    def forward(self, x):
        input = x   # (N, C, H, W)
        x = self.norm(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = input + x
        return x


# Conv Spatial Attention Block
class ConvBlock(nn.Module):
    def __init__(self, dim, expansion_factor=2, bias=True, norm_layer=nn.LayerNorm):
        super(ConvBlock, self).__init__()
        hidden_features = int(dim*expansion_factor)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = norm_layer(dim)
        self.pwconv1 = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.grn = GRN_channel_first(hidden_features)
        self.pwconv2 = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        input = x   # (N, C, H, W)
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = input + x
        return x


# Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=True, gate='gelu', norm_layer=nn.LayerNorm):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.norm = norm_layer(dim)
        self.conv1 = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.conv2 = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.conv3 = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        # SimpleGate
        if gate == 'gelu':
            self.sg = GeluGate()
        else:
            self.sg = SimpleGate()

    def forward(self, x):
        input = x   # (N, C, H, W)
        x = self.norm(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.conv3(x)
        x = input + x
        return x


# ConvFormer Block
class ConvFormerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.channel_attn = ChannelAttention(dim=dim, norm_layer=norm_layer)#, channel_expansion_ratio=mlp_ratio)
        self.spatial_attn = ConvBlock(dim, norm_layer=norm_layer)
        self.ffn = FeedForward(dim, mlp_ratio, norm_layer=norm_layer)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        x = self.ffn(x)
        return x


# Basic ConvFormer Block
class BasicLayerConvFormer(nn.Module):
    def __init__(self, dim, depth, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        self.blocks = nn.ModuleList([
            ConvFormerBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop, attn_drop=attn_drop,
                                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                norm_layer=norm_layer)
            for i in range(depth)])
        
    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


# Spatial Attention Block
class Attention(nn.Module):
    def __init__(self, dim, num_heads, head_dim=None, qkv_bias=True, attn_drop=0., proj_drop=0., use_dwconv=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.norm = norm_layer(dim)
        self.qkv = nn.Conv2d(dim, attn_dim * 3, 1, bias=qkv_bias)
        self.use_dwconv = use_dwconv
        if use_dwconv:
            self.qkv_dwconv = nn.Conv2d(attn_dim*3, attn_dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(attn_dim, dim, 1, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape

        input = x   # (N, C, H, W)
        x = self.norm(x)
        if self.use_dwconv:
            q, k, v = self.qkv_dwconv(self.qkv(x)).view(B, self.num_heads, self.head_dim * 3, -1).chunk(3, dim=2)
        else:
            q, k, v = self.qkv(x).view(B, self.num_heads, self.head_dim * 3, -1).chunk(3, dim=2)

        attn = (q.transpose(-2, -1) @ k) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (v @ attn.transpose(-2, -1)).view(B, -1, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = input + x
        return x


# TransFormer Block
class TransFormerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.channel_attn = ChannelAttention(dim=dim, norm_layer=norm_layer)#, channel_expansion_ratio=mlp_ratio)
        self.spatial_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, norm_layer=norm_layer)
        self.ffn = FeedForward(dim, mlp_ratio, norm_layer=norm_layer)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        x = self.ffn(x)
        return x


# Basic TransFormer Block
class BasicLayerTransFormer(nn.Module):
    def __init__(self, dim, depth, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            TransFormerBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop, attn_drop=attn_drop,
                                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x
