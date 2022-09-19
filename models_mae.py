# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

# from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed
from timm.models.layers import DropPath, to_2tuple

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from parallel_experts import MoE

from moe import MoE as MMoE
from moe import cvMoE
# from mixture_of_experts import MoE as newMoE
from oldmoe import MoE as oldMoE

from parallel_experts import RandomMoE, TaskMoE

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 16, p2 = 16),
        # )
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        # x = self.to_patch_embedding(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, head_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        if head_dim is None:
            head_dim = dim // num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))

        # For rare cases, the attention weights are inf due to the mix-precision training.
        # We clamp the tensor to the max values of the current data type
        # This is different from MAE training as we don't observe such cases on image-only MAE.
        if torch.isinf(attn).any():
            clamp_value = torch.finfo(attn.dtype).max-1000
            attn = torch.clamp(attn, min=-clamp_value, max=clamp_value)

        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, head_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None):
        super().__init__()
        self.num_heads = num_heads
        if head_dim is None:
            head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None, k=None, v=None):
        B, N, C = x.shape
        N_k = k.shape[1]
        N_v = v.shape[1]

        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            v_bias = self.v_bias

        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = q.reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)    # (B, N_head, N_q, dim)

        k = F.linear(input=k, weight=self.k.weight, bias=k_bias)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)   
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))      # (B, N_head, N_q, N_k)

        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class DecoderBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, head_dim=None, init_values=None, window_size=None):
        super().__init__()
        self.norm1_q = norm_layer(dim)
        self.norm1_k = norm_layer(dim)
        self.norm1_v = norm_layer(dim)

        self.attn =  CrossAttention(
            dim, num_heads=num_heads, head_dim=head_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size)

        # # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2_cross = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x_q, x_kv, pos_q, pos_k, mask=None):
        x = x_q + self.drop_path(
            self.attn(self.norm1_q(x_q + pos_q), mask, k=self.norm1_k(x_kv + pos_k), v=self.norm1_v(x_kv)))
        x = self.norm2_cross(x)
        x = x + self.drop_path(self.mlp(x))

        return x

# MLP hidden/4 topk=4
class MoEAttention(nn.Module):
    def __init__(self, dim, num_experts=24, num_heads=8, head_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
        sample_topk=2, cvloss=0, switchloss=0.01 * 10, zloss=0.001 * 1, moe_type='normal'):
        super().__init__()
        self.num_experts = num_experts
        self.sample_topk = sample_topk

        self.num_heads = num_heads
        if head_dim is None:
            head_dim = dim // num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.moe_type = moe_type

        if moe_type == 'random':
            self.q_proj = RandomMoE(dim, head_dim, num_experts, num_heads, cvloss=cvloss, switchloss=switchloss, zloss=zloss)
        elif moe_type == 'FLOP': # use this to evaluate FLOPs
            self.att_experts = [
                nn.Sequential(
                    nn.Linear(dim, head_dim),
                )
                for _ in range(num_experts)
            ]
            self.q_proj = MMoE(dim, self.att_experts, num_heads, dropout=0., concat=True)
            self.out_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(head_dim, dim),
                    nn.Dropout(0.)
                )
                for _ in range(num_experts)
            ])
        else:
            self.q_proj = MoE(dim, head_dim, num_experts, num_heads, cvloss=cvloss, switchloss=switchloss, zloss=zloss)

        self.kv_proj = nn.Sequential(
            nn.Linear(dim, head_dim * 2),
        )

        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):

        B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        if self.moe_type == 'FLOP':
            q, aux_loss = self.q_proj(x, multiply_by_gates=False, sample_topk=self.sample_topk)
        else:
            q, aux_loss = self.q_proj.map(x, sample_topk=self.sample_topk)
        k, v = self.kv_proj(x).chunk(2, dim=-1)

        q = q.reshape(B, N, self.num_heads, self.head_dim)
        k = k.reshape(B, N, self.head_dim)
        v = v.reshape(B, N, self.head_dim)

        attn = torch.einsum('bihd,bjd->bhij', q, k) * self.scale
        # attn = attn.premute(0,3,1,2) # b, h, i, j

        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))

        # For rare cases, the attention weights are inf due to the mix-precision training.
        # We clamp the tensor to the max values of the current data type
        # This is different from MAE training as we don't observe such cases on image-only MAE.
        if torch.isinf(attn).any():
            clamp_value = torch.finfo(attn.dtype).max-1000
            attn = torch.clamp(attn, min=-clamp_value, max=clamp_value)

        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        attn = torch.einsum('bhij,bjd->bihd', attn, v)

        if self.moe_type == 'FLOP':
            x = self.q_proj.dispatch(
                    attn.reshape(B, N, self.num_heads, self.head_dim).contiguous(), 
                    self.out_proj
                )
        else:
            x = self.q_proj.reduce(attn)
        x = self.proj_drop(x)
        return x, aux_loss

class MoAttention(Attention):
    def __init__(self, dim, num_heads=8, z_weight=0.000, temperature=1.0, *args, **kwargs):
        super().__init__(dim, num_heads=num_heads, *args, **kwargs)
        self.gate = nn.Linear(dim, num_heads, bias=False)
        self.gate.requires_grad=False
        self.num_heads = num_heads
        self.z_weight = z_weight
        self.temperature = temperature

    def compute_zloss(self, logits):
        zloss = torch.mean(torch.log(torch.exp(logits).sum(dim=-1)) ** 2)
        return zloss

    def forward(self, x, mask=None):
        gate_layer = self.gate(x)  # b n h
        gates = torch.softmax(gate_layer/self.temperature, dim=-1).transpose(1, 2) # * self.num_heads # b h n

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) B, H, N, dim
        attn = (q @ k.transpose(-2, -1)) * self.scale # B, H, N, N

        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))

        # For rare cases, the attention weights are inf due to the mix-precision training.
        # We clamp the tensor to the max values of the current data type
        # This is different from MAE training as we don't observe such cases on image-only MAE.
        if torch.isinf(attn).any():
            clamp_value = torch.finfo(attn.dtype).max-1000
            attn = torch.clamp(attn, min=-clamp_value, max=clamp_value)

        attn = attn.softmax(dim=-1)

        attn = attn * gates[:, :, :, None]

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x,  self.z_weight * self.compute_zloss(gate_layer)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MoEMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., topk=2, num_ffd=32,
        cvloss=0, switchloss=0.01, zloss=0.001, sample_topk=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # class Experts(nn.Module):
        #     def __init__(self, in_features, hidden_features, out_features,num_experts = 16):
        #         super().__init__()
        #         self.w1 = nn.Parameter(torch.randn(num_experts, in_features, hidden_features))
        #         self.w2 = nn.Parameter(torch.randn(num_experts, hidden_features, out_features))
        #         # self.w3 = nn.Parameter(torch.randn(num_experts, dim * 4, dim))
        #         self.act = nn.GELU()

        #     def forward(self, x):
        #         hidden1 = self.act(torch.einsum('end,edh->enh', x, self.w1))
        #         out = torch.einsum('end,edh->enh', hidden1, self.w2)
        #         return out

        # experts = Experts(in_features, hidden_features, out_features, num_experts = 32)
        # self.net = newMoE(dim = in_features, num_experts = 32, experts = experts)
        # ffd_exports = [
        #         nn.Sequential(
        #         nn.Linear(in_features, hidden_features),
        #         act_layer(),
        #         # nn.Dropout(drop),
        #         nn.Linear(hidden_features, out_features),
        #         # nn.Dropout(drop)
        #     )
        #         for _ in range(num_ffd)
        #     ]
        # self.net = cvMoE(in_features, ffd_exports, topk, cvloss=cvloss, switchloss=switchloss, zloss=zloss)
        self.net = oldMoE(input_size=in_features, output_size=out_features, num_experts=10, hidden_size=hidden_features, noisy_gating=True, k=4)
        self.sample_topk = sample_topk

    def forward(self, x):
        b, l, d = x.shape[0], x.shape[1], x.shape[2]
        x = x.view(-1,d)
        x, loss = self.net(x)
        x = x.reshape(b, l, -1)
        return x, loss
        # return self.net(x)

class MoEMlpBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6), head_dim=None, init_values=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, head_dim=head_dim, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MoEMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
        y, z_loss = self.mlp(self.norm2(x))
        x = x + self.drop_path(y)
        return x, z_loss

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, head_dim=None, init_values=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, head_dim=head_dim, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class mlpBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6), head_dim=None, init_values=None):
        super().__init__()
        # self.norm1 = norm_layer(dim)
        # self.attn = Attention(
        #     dim, num_heads=num_heads, head_dim=head_dim, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        # x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class BBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, head_dim=None, init_values=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, head_dim=head_dim, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, 0

class MoABlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, head_dim=None, init_values=None, z_weight=0.000):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MoAttention(
            dim, num_heads=num_heads, head_dim=head_dim, z_weight=z_weight, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        assert z_weight == 0

    def forward(self, x, mask=None):
        y, z_loss = self.attn(self.norm1(x), mask=mask)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class MoEBlock(nn.Module):

    def __init__(self, dim, num_heads, num_attn_experts=24, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, head_dim=None, init_values=None, z_weight=0.000,
                 cvloss=0, switchloss=0.01 * 1, zloss=0.001 * 1, sample_topk=0, moe_type='normal'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MoEAttention(
            dim, num_heads=num_heads, num_experts=num_attn_experts, head_dim=head_dim, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            cvloss=cvloss, switchloss=switchloss, zloss=zloss, sample_topk=sample_topk, moe_type=moe_type)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        assert z_weight == 0

    def forward(self, x, mask=None):
        y, z_loss = self.attn(self.norm1(x), mask=mask)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, z_loss

class MoEnhanceBlock(nn.Module):

    def __init__(self, dim, num_heads, num_attn_experts=24, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 num_ffd_experts=16, ffd_heads=2, ffd_noise=True,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, head_dim=None, init_values=None, z_weight=0.000,
                 post_layer_norm=False,
                 cvloss=0, switchloss=0.01 * 1, zloss=0.001 * 1, sample_topk=0, moe_type='normal'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MoEAttention(
            dim, num_heads=num_heads, num_experts=num_attn_experts, head_dim=head_dim, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            cvloss=cvloss, switchloss=switchloss, zloss=zloss, sample_topk=sample_topk, moe_type=moe_type)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        if moe_type == 'FLOP':
            ffd_exports = [
                    nn.Sequential(
                    # nn.LayerNorm(dim),
                    nn.Linear(dim, mlp_hidden_dim // ffd_heads),
                    nn.GELU(),
                    # nn.Dropout(dropout),
                    nn.Linear(mlp_hidden_dim // ffd_heads, dim),
                    # nn.Dropout(dropout)
                    # nn.LayerNorm(dim),
                    )
                    for _ in range(num_ffd_experts)
                ]
            self.mlp = MMoE(dim, ffd_exports, ffd_heads, 0.)
        else:
            self.mlp = MoE(dim,
                    mlp_hidden_dim // ffd_heads, num_ffd_experts, ffd_heads,
                    cvloss=cvloss,
                    switchloss=switchloss,
                    zloss=zloss,
                    activation=nn.Sequential(
                        nn.GELU(),
                        # self.dropout_module Remove dropout for now
                    ),
                    noisy_gating=ffd_noise
                )
        self.post_layer_norm = post_layer_norm
        assert z_weight == 0

    def forward(self, x, mask=None):
        if self.post_layer_norm:
            y, z_loss = self.attn(x, mask=mask)
            x = x + self.drop_path(y)
            x = self.norm1(x)

            y, aux_loss = self.mlp(x)
            x = x + self.drop_path(y)
            x = self.norm2(x)
            return x, z_loss + aux_loss
        else:
            y, z_loss = self.attn(self.norm1(x), mask=mask)
            x = x + self.drop_path(y)

            y, aux_loss = self.mlp(self.norm2(x))
            x = x + self.drop_path(y)
            return x, z_loss + aux_loss

class MoETaskAttention(nn.Module):
    def __init__(self, dim, task_num=9, num_experts=24, num_heads=8, head_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
        sample_topk=2, cvloss=0, switchloss=0.01 * 10, zloss=0.001 * 1, moe_type='normal'):
        super().__init__()
        self.task_num = task_num
        self.num_experts = num_experts
        self.sample_topk = sample_topk

        self.num_heads = num_heads
        if head_dim is None:
            head_dim = dim // num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.moe_type = moe_type

        self.q_proj = TaskMoE(dim, head_dim, num_experts, num_heads, acc_aux_loss=True, task_num=task_num, cvloss=cvloss, switchloss=switchloss, zloss=zloss)

        self.kv_proj = nn.Sequential(
            nn.Linear(dim, head_dim * 2),
        )

        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, task_bh, mask=None):

        B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        q, aux_loss = self.q_proj.map(x, task_bh, sample_topk=self.sample_topk)
        k, v = self.kv_proj(x).chunk(2, dim=-1)

        q = q.reshape(B, N, self.num_heads, self.head_dim)
        k = k.reshape(B, N, self.head_dim)
        v = v.reshape(B, N, self.head_dim)

        attn = torch.einsum('bihd,bjd->bhij', q, k) * self.scale
        # attn = attn.premute(0,3,1,2) # b, h, i, j

        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))

        # For rare cases, the attention weights are inf due to the mix-precision training.
        # We clamp the tensor to the max values of the current data type
        # This is different from MAE training as we don't observe such cases on image-only MAE.
        if torch.isinf(attn).any():
            clamp_value = torch.finfo(attn.dtype).max-1000
            attn = torch.clamp(attn, min=-clamp_value, max=clamp_value)

        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        attn = torch.einsum('bhij,bjd->bihd', attn, v)

        if self.moe_type == 'FLOP':
            x = self.q_proj.dispatch(
                    attn.reshape(B, N, self.num_heads, self.head_dim).contiguous(), 
                    self.out_proj
                )
        else:
            x = self.q_proj.reduce(attn)
        x = self.proj_drop(x)
        return x, aux_loss

class MoEnhanceTaskBlock(nn.Module):

    def __init__(self, dim, num_heads, num_attn_experts=24, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 num_ffd_experts=16, ffd_heads=2, ffd_noise=True,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, head_dim=None, init_values=None, z_weight=0.000,
                 post_layer_norm=False,
                 task_num=9,
                 cvloss=0, switchloss=0.01 * 1, zloss=0.001 * 1, sample_topk=0, moe_type='normal'):
        super().__init__()
        self.task_num = task_num
        self.norm1 = norm_layer(dim)
        self.attn = MoETaskAttention(
            dim, task_num=task_num, num_heads=num_heads, num_experts=num_attn_experts, head_dim=head_dim, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            cvloss=cvloss, switchloss=switchloss, zloss=zloss, sample_topk=sample_topk, moe_type=moe_type)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = TaskMoE(dim,
                mlp_hidden_dim // ffd_heads, num_ffd_experts, ffd_heads,
                acc_aux_loss=True, 
                cvloss=cvloss,
                switchloss=switchloss,
                zloss=zloss,
                task_num=task_num,
                activation=nn.Sequential(
                    nn.GELU(),
                    # self.dropout_module Remove dropout for now
                ),
                noisy_gating=ffd_noise
            )
        assert z_weight == 0

    def forward(self, x, task_bh, mask=None):
        y, z_loss = self.attn(self.norm1(x), task_bh, mask=mask)
        x = x + self.drop_path(y)

        y, aux_loss = self.mlp(self.norm2(x), task_bh)
        x = x + self.drop_path(y)
        return x, z_loss + aux_loss

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16, head_dim=None,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 has_z_loss=False):
        super().__init__()
        self.has_z_loss = has_z_loss

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, head_dim=head_dim, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        z_loss = 0

        # apply Transformer blocks
        for blk in self.blocks:
            if self.has_z_loss:
                x, aux_loss = blk(x)
                z_loss = z_loss + aux_loss
            else:
                x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore, z_loss

    def forward_decoder(self, x, ids_restore, mask=None):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x, 0

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore, z_loss = self.forward_encoder(imgs, mask_ratio)
        pred, aux_loss = self.forward_decoder(latent, ids_restore, mask=mask)  # [N, L, p*p*3]
        z_loss = z_loss + aux_loss
        loss = self.forward_loss(imgs, pred, mask)

        ### For visualization
        img_token = self.patchify(imgs)
        mask_img = img_token * (1-mask[:,:,None])
        mask_img = self.unpatchify(mask_img)

        if self.norm_pix_loss: # unnormalize
            mean = img_token.mean(dim=-1, keepdim=True)
            var = img_token.var(dim=-1, keepdim=True)
            pred = pred * ((var + 1.e-6)**.5) + mean

        pred = pred * mask[:,:,None] + img_token * (1-mask[:,:,None])
        pred = self.unpatchify(pred)

        return loss, pred, mask, z_loss


class MaskedAutoencoderViTMoA(MaskedAutoencoderViT):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, 
                 embed_dim=1024, depth=24, num_heads=16, head_dim=None,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, z_weight=0.000,
                 *args, **kwargs):
        super().__init__(embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                 mlp_ratio=mlp_ratio, norm_layer=norm_layer, 
                 *args, **kwargs)

        self.blocks = nn.ModuleList([
            MoABlock(embed_dim, num_heads, mlp_ratio, head_dim=head_dim, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])

        self.initialize_weights()

        for blk in self.blocks:
            blk.attn.gate.apply(self.moa_init_weight)

    def moa_init_weight(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.fill_(0.00)
            # module.weight.data.normal_(mean=1.0, std=0.002)
            # torch.nn.init.xavier_uniform_(module.weight)

import random
class MaskedAutoencoderViTMoE(MaskedAutoencoderViT):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, 
                 embed_dim=1024, depth=24, num_heads=16, head_dim=None,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, z_weight=0.000,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 num_attn_experts=24, 
                 has_z_loss=True,
                 cvloss=0, switchloss=0.01 * 10, zloss=0.000 * 1, sample_topk=2,
                 moe_type='normal',
                 *args, **kwargs):
        super().__init__(embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                 mlp_ratio=mlp_ratio, norm_layer=norm_layer, has_z_loss=has_z_loss,
                 decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads,
                 *args, **kwargs)

        self.blocks = nn.ModuleList([
            MoEBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, num_attn_experts=num_attn_experts, head_dim=head_dim, qkv_bias=True, norm_layer=norm_layer,
                cvloss=cvloss, switchloss=switchloss, zloss=zloss, sample_topk=sample_topk, moe_type=moe_type)
            for i in range(depth)])

        self.initialize_weights()

        self.decoder_blocks = nn.ModuleList([
            MoEBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio=mlp_ratio, num_attn_experts=num_attn_experts, head_dim=head_dim, qkv_bias=True, norm_layer=norm_layer,
                cvloss=cvloss, switchloss=switchloss, zloss=zloss, sample_topk=sample_topk)
            for i in range(decoder_depth)])

        for blk in self.blocks:
            blk.attn.q_proj.f_gate.apply(self.moa_init_weight)

        for blk in self.decoder_blocks:
            blk.attn.q_proj.f_gate.apply(self.moa_init_weight)

        self.gate_num = torch.zeros(48).float()
        self.gate_num.requires_grad=False

        self.gate_num_top1 = torch.zeros(48).float()
        self.gate_num_top1.requires_grad=False

    def moa_init_weight(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.fill_(0.00)
            # torch.nn.init.normal_(module.weight.data, std=.002)

    def clear(self):
        self.gate_num = torch.zeros(48).float()
        self.gate_num.requires_grad=False
        self.gate_num_top1 = torch.zeros(48).float()
        self.gate_num_top1.requires_grad=False

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        z_loss = 0

        # apply Transformer blocks
        now = 0
        vis = 0

        for blk in self.blocks:
            now = now + 1
            if now == 1:
                # with torch.no_grad():
                vis = blk.attn.q_proj.f_gate(x) # B, N, expert
                choose = torch.argmax(vis, dim=-1) # B, N
                for i in range(0,48):
                    self.gate_num_top1[i] = self.gate_num_top1[i] + (choose==i).sum().item()

                value, choose = torch.topk(vis, 12, dim=-1)
                if random.random()<0.01:
                    value = value.mean(0).mean(0)
                    # print('value: ', value* 100000)
                    # print('vis: ', vis.mean()* 100000,vis.min()* 100000,vis.max()* 100000)

                for i in range(0,48):
                    self.gate_num[i] = self.gate_num[i] + (choose==i).sum().item()
                vis = vis.sum()

            x, aux_loss = blk(x)
            z_loss = z_loss + aux_loss + vis * 0
            
        x = self.norm(x)

        return x, mask, ids_restore, z_loss

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        z_loss = 0
        for blk in self.decoder_blocks:
            x, aux_loss = blk(x)
            z_loss = z_loss + aux_loss
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x, z_loss



class MaskedAutoencoderViTMoEAll(MaskedAutoencoderViT):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, 
                 embed_dim=1024, depth=24, num_heads=16, 
                 mlp_ratio=4., norm_layer=nn.LayerNorm, z_weight=0.000,
                 has_z_loss=True,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 num_attn_experts=48, head_dim=None,
                 num_ffd_experts=16, ffd_heads=2, ffd_noise=True,
                 *args, **kwargs):
        super().__init__(embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                 mlp_ratio=mlp_ratio, norm_layer=norm_layer, has_z_loss=has_z_loss,
                 *args, **kwargs)

        self.blocks = nn.Sequential(*[
            MoEnhanceBlock(
                num_attn_experts=num_attn_experts, head_dim=head_dim,
                num_ffd_experts=num_ffd_experts, ffd_heads=ffd_heads, ffd_noise=ffd_noise,
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True,
                drop=0., attn_drop=0., drop_path=0., norm_layer=norm_layer,
                moe_type='normal',
                )
            for i in range(depth)])

        self.initialize_weights()

        for idx, blk in enumerate(self.blocks):
            blk.mlp.f_gate.data.fill_(0.00)
            blk.attn.q_proj.f_gate.data.fill_(0.00)

        self.gate_num = torch.zeros(32).float()
        self.gate_num.requires_grad=False

        self.gate_num_top1 = torch.zeros(32).float()
        self.gate_num_top1.requires_grad=False


    def moa_init_weight(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.fill_(0.00)

    def clear(self):
        self.gate_num = torch.zeros(32).float()
        self.gate_num.requires_grad=False
        self.gate_num_top1 = torch.zeros(32).float()
        self.gate_num_top1.requires_grad=False

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        z_loss = 0

        # apply Transformer blocks
        now = 0
        vis = 0

        for blk in self.blocks:
            x, aux_loss = blk(x)
            z_loss = z_loss + aux_loss
            now = now + 1
            
        x = self.norm(x)

        return x, mask, ids_restore, z_loss


class MaskedAutoencoderViTMoEMlp(MaskedAutoencoderViT):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, 
                 embed_dim=1024, depth=24, num_heads=16, head_dim=None,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, z_weight=0.000,
                 has_z_loss=True,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 *args, **kwargs):
        super().__init__(embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                 mlp_ratio=mlp_ratio, norm_layer=norm_layer, has_z_loss=has_z_loss,
                 *args, **kwargs)

        A = []
        for i in range(depth):
            # A.append(MoEMlpBlock(embed_dim, num_heszads, mlp_ratio=mlp_ratio, head_dim=head_dim, qkv_bias=True, norm_layer=norm_layer))
            if i%2==1:
                A.append(MoEMlpBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, head_dim=head_dim, qkv_bias=True, norm_layer=norm_layer))
            else:
                A.append(BBlock(embed_dim, num_heads, mlp_ratio, head_dim=head_dim, qkv_bias=True, norm_layer=norm_layer))
        self.blocks = nn.ModuleList(A)
        # self.blocks = nn.ModuleList([
        #     MoEMlpBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, head_dim=head_dim, qkv_bias=True, norm_layer=norm_layer)
        #     for i in range(depth)])

        # self.decoder_blocks = nn.ModuleList([
        #     MoEMlpBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio=mlp_ratio, head_dim=head_dim, qkv_bias=True, norm_layer=norm_layer)
        #     for i in range(decoder_depth)])

        self.initialize_weights()

        # for idx, blk in enumerate(self.blocks):
        #     if idx % 2 == 1:
        #         blk.mlp.net.f_gate.apply(self.moa_init_weight)

        self.gate_num = torch.zeros(32).float()
        self.gate_num.requires_grad=False

        self.gate_num_top1 = torch.zeros(32).float()
        self.gate_num_top1.requires_grad=False


    def moa_init_weight(self, module):
        if isinstance(module, (nn.Linear)):
            # module.weight.data.fill_(0.00)
            module.weight.data.fill_(0.00)

    def clear(self):
        self.gate_num = torch.zeros(32).float()
        self.gate_num.requires_grad=False
        self.gate_num_top1 = torch.zeros(32).float()
        self.gate_num_top1.requires_grad=False

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        z_loss = 0

        # apply Transformer blocks
        now = 0
        vis = 0

        for blk in self.blocks:
            # if now == 0:
            #     # with torch.no_grad():
            #     # xx = x + blk.drop_path(blk.attn(blk.norm1(x), mask=None))
            #     xx = x
            #     vis = blk.mlp.net.f_gate(blk.norm2(xx)) # B, N, expert

            #     # noise = torch.normal(torch.zeros_like(vis).float(), torch.ones_like(vis).float()/32).cuda()

            #     choose = torch.argmax(vis, dim=-1) # B, N
            #     for i in range(0,32):
            #         self.gate_num_top1[i] = self.gate_num_top1[i] + (choose==i).sum().item()

            #     # _, choose = torch.topk(vis, 2, dim=-1)
            #     # for i in range(0,32):
            #     #     self.gate_num_top1[i] = self.gate_num_top1[i] + (choose==i).sum().item()

            #     # vis = vis + noise
            #     value, choose = torch.topk(vis, 2, dim=-1)

            #     # print(torch.mean(value[:,:,0]), torch.mean(value[:,:,1])) # 2.4, 1.8
            #     for i in range(0,32):
            #         self.gate_num[i] = self.gate_num[i] + (choose==i).sum().item()
            #     vis = vis.sum()
            #     z_loss = z_loss + vis * 0

            x, aux_loss = blk(x)
            z_loss = z_loss + aux_loss
            now = now + 1
            
        x = self.norm(x)

        return x, mask, ids_restore, z_loss

def mae_vit_moemlp_base_patch16(**kwargs):
    model = MaskedAutoencoderViTMoEMlp(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, head_dim=768//12 * 2,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# this setting have unbalanced issues.
def mae_vit_moe_base_patch16(**kwargs):
    model = MaskedAutoencoderViTMoE(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, head_dim=768//12 * 3,
        num_attn_experts=48,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        cvloss=0, switchloss=0.01, zloss=0.000, sample_topk=0,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_moa_base_patch16(**kwargs):
    model = MaskedAutoencoderViTMoA(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, head_dim=768//12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_one_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=1, num_heads=12, head_dim=768//12,
        decoder_embed_dim=512, decoder_depth=1, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_one_moemlp_patch16(**kwargs):
    model = MaskedAutoencoderViTMoEMlp(
        patch_size=16, embed_dim=768, depth=1, num_heads=12, head_dim=768//12 * 2,
        decoder_embed_dim=512, decoder_depth=1, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_one_moe_patch16(**kwargs):
    model = MaskedAutoencoderViTMoE(
        patch_size=16, embed_dim=768, depth=1, num_heads=12, head_dim=768//12 * 2,
        decoder_embed_dim=512, decoder_depth=1, decoder_num_heads=16,
        num_attn_experts=48,
        cvloss=0, switchloss=0.01, zloss=0.001, sample_topk=0,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_tiny_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, head_dim=192//3,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_enmoe_tiny(**kwargs):
    model = VisionTransformerMoEAll(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, 
        num_attn_experts=3*8, head_dim=192//3 * 2,
        num_ffd_experts=12, ffd_heads=2, ffd_noise=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_moe_tiny(**kwargs):
    model = VisionTransformerMoE(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, 
        num_attn_experts=3*8, head_dim=192//3 * 2,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        norm_layer=partial(nn.LayewrNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_enmoe_tiny(**kwargs):
    model = MaskedAutoencoderViTMoEAll(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, 
        num_attn_experts=3*8, head_dim=768//12 * 2,
        num_ffd_experts=12, ffd_heads=2, ffd_noise=True,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, head_dim=768//12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

## CrossViT part

class MaskedAutoencoderCrossViT(MaskedAutoencoderViT):
    def __init__(self, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 *args, **kwargs):
        super().__init__(decoder_embed_dim=decoder_embed_dim, decoder_num_heads=decoder_num_heads,
            decoder_depth=decoder_depth, mlp_ratio=mlp_ratio, norm_layer=norm_layer, 
            *args, **kwargs)
        del self.decoder_blocks
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.initialize_weights()

    def forward_decoder(self, x, ids_restore, mask=None):
        # embed tokens
        x = self.decoder_embed(x)
        N, _, D = x.shape[0], x.shape[1], x.shape[2]

        mask_token = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)

        x_ = torch.cat([x[:, 1:, :], mask_token], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        bool_mask = mask.bool()

        x_masked = x_[bool_mask, :].reshape(N, -1, D) 
        x_unmasked = x_[~bool_mask, :].reshape(N, -1, D) 

        pos_embed_masked = self.decoder_pos_embed[:,1:].repeat(N,1,1)[bool_mask, :].reshape(N, -1, D)
        pos_embed_unmasked = self.decoder_pos_embed[:,1:].repeat(N,1,1)[~bool_mask, :].reshape(N, -1, D)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            # x_masked = blk(x_masked, torch.cat([x_unmasked, x_masked], dim=1), 
            #     pos_embed_masked, torch.cat([pos_embed_unmasked, pos_embed_masked], dim=1))
            x_masked = blk(x_masked, x_unmasked, 
                pos_embed_masked, pos_embed_unmasked)

        x = torch.zeros([x_masked.shape[0], x_masked.shape[1] + x_unmasked.shape[1], x_masked.shape[2]], device=x_masked.device)
        x = x.reshape(-1, D)
        x_masked = x_masked.reshape(-1, D)
        x_unmasked = x_unmasked.reshape(-1, D)

        x[bool_mask.view(-1),:] = x[bool_mask.view(-1),:] + x_masked
        x[~bool_mask.view(-1),:] = x[~bool_mask.view(-1),:] + x_unmasked

        x = x.reshape(N, -1, D)

        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x, 0

class MaskedAutoencoderCrossViTMoEAll(MaskedAutoencoderCrossViT):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, 
                 embed_dim=1024, depth=24, num_heads=16, 
                 mlp_ratio=4., norm_layer=nn.LayerNorm, z_weight=0.000,
                 has_z_loss=True,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 num_attn_experts=48, head_dim=None,
                 num_ffd_experts=16, ffd_heads=2, ffd_noise=True,
                 *args, **kwargs):
        super().__init__(embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                 mlp_ratio=mlp_ratio, norm_layer=norm_layer, has_z_loss=has_z_loss,
                 *args, **kwargs)

        self.blocks = nn.Sequential(*[
            MoEnhanceBlock(
                num_attn_experts=num_attn_experts, head_dim=head_dim,
                num_ffd_experts=num_ffd_experts, ffd_heads=ffd_heads, ffd_noise=ffd_noise,
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True,
                drop=0., attn_drop=0., drop_path=0., norm_layer=norm_layer,
                moe_type='normal',
                )
            for i in range(depth)])

        self.initialize_weights()

        for idx, blk in enumerate(self.blocks):
            blk.mlp.f_gate.data.fill_(0.00)
            blk.attn.q_proj.f_gate.data.fill_(0.00)

    def moa_init_weight(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.fill_(0.00)

def mae_cross_vit_base(**kwargs):
    model = MaskedAutoencoderCrossViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, head_dim=768//12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_cross_vit_tiny(**kwargs):
    model = MaskedAutoencoderCrossViT(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, head_dim=192//3,
        decoder_embed_dim=512, decoder_depth=1, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_cross_pretrain_tiny(**kwargs):
    model = MaskedAutoencoderCrossViTMoEAll(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, 
        num_attn_experts=3, head_dim=192//3 * 2,
        num_ffd_experts=2, ffd_heads=2, ffd_noise=True,
        decoder_embed_dim=512, decoder_depth=1, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_cross_vit_enmoe_tiny(**kwargs):
    model = MaskedAutoencoderCrossViTMoEAll(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, 
        num_attn_experts=3*8, head_dim=192//3 * 2,
        num_ffd_experts=12, ffd_heads=2, ffd_noise=True,
        decoder_embed_dim=512, decoder_depth=1, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

