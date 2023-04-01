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

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
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
                    bias=True,
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
    def __init__(self, dim, noisy_gating=True, task_num=9, num_experts=24, num_heads=8, head_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
        sample_topk=2, cvloss=0, switchloss=0.01 * 10, zloss=0.001 * 1, w_topk_loss=0.1, w_MI=0., limit_k=0, moe_type='normal'):
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

        self.q_proj = TaskMoE(dim, head_dim, num_experts, num_heads, noisy_gating=noisy_gating, w_MI=w_MI, acc_aux_loss=True, task_num=task_num, cvloss=cvloss, switchloss=switchloss, zloss=zloss, w_topk_loss=w_topk_loss, limit_k=limit_k)

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
                 noisy_gating=True,
                 att_w_topk_loss=0.0, att_limit_k=0, 
                 cvloss=0, switchloss=0.01 * 1, zloss=0.001 * 1, w_topk_loss=0.0, limit_k=0, 
                 w_MI = 0.,
                 use_moe_mlp=True,
                 use_moe_attn=True,
                 sample_topk=0, moe_type='normal'):
        super().__init__()
        self.task_num = task_num
        self.norm1 = norm_layer(dim)
        self.use_moe_attn = use_moe_attn
        if use_moe_attn:
            self.attn = MoETaskAttention(
                dim, task_num=task_num, noisy_gating=noisy_gating, num_heads=num_heads, num_experts=num_attn_experts, head_dim=head_dim, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                cvloss=cvloss, switchloss=switchloss, zloss=zloss, w_MI=w_MI, w_topk_loss=att_w_topk_loss, limit_k=att_limit_k, sample_topk=sample_topk, moe_type=moe_type)
        else:
            self.attn = Attention(dim, num_heads=num_heads, head_dim=head_dim, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.use_moe_mlp = use_moe_mlp
        if use_moe_mlp:
            self.mlp = TaskMoE(dim,
                    mlp_hidden_dim // ffd_heads, num_ffd_experts, ffd_heads,
                    bias=True,
                    acc_aux_loss=True, 
                    cvloss=cvloss,
                    switchloss=switchloss,
                    zloss=zloss,
                    w_topk_loss=w_topk_loss,
                    w_MI=w_MI,
                    limit_k=limit_k,
                    task_num=task_num,
                    activation=nn.Sequential(
                        nn.GELU(),
                        # self.dropout_module Remove dropout for now
                    ),
                    noisy_gating=ffd_noise
                )
        else:
            self.mlp = Mlp(dim, hidden_features=dim * 4, drop=drop)
            # Mlp(nn.Module): def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        assert z_weight == 0

    def forward(self, x, task_bh, mask=None):
        if self.use_moe_attn:
            y, z_loss = self.attn(self.norm1(x), task_bh, mask=mask)
            x = x + self.drop_path(y)
        else:
            z_loss = 0.0
            y = self.attn(self.norm1(x), mask=mask)
            x = x + self.drop_path(y)

        if self.use_moe_mlp:
            y, aux_loss = self.mlp(self.norm2(x), task_bh)
            x = x + self.drop_path(y)
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            aux_loss = 0
        return x, z_loss + aux_loss
