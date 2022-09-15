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

from models_mae import PatchEmbed, Attention, MoAttention, Mlp, Block, MoABlock, MoEBlock, MoEMlpBlock, MoEMlp, MoEnhanceBlock

import timm.models.vision_transformer
class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # apply Transformer blocks

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome, 0

    def forward(self, x):
        x, z_loss = self.forward_features(x)
        x = self.head(x)
        return x, z_loss

class RegressorBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1_q = norm_layer(dim)
        self.norm1_k = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.norm2_cross = norm_layer(dim)
        self.cross_attn =  CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        self.mlp_cross = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1_cross = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2_cross = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1_cross = nn.Parameter(torch.ones((dim)),requires_grad=False)
            self.gamma_2_cross = nn.Parameter(torch.ones((dim)),requires_grad=False)

    def forward(self, x_q, x_kv, pos_q, pos_k, bool_masked_pos):
        x = x_q + self.drop_path(self.gamma_1_cross * self.cross_attn(self.norm1_q(x_q + pos_q),
         bool_masked_pos, k=self.norm1_k(x_kv + pos_k), v=self.norm1_v(x_kv)))
        x = self.norm2_cross(x)
        x = x + self.drop_path(self.gamma_2_cross * self.mlp_cross(x))

        return x

class VisionTransformerMoA(VisionTransformer):
    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None, 
                 **kwargs):
        super(VisionTransformerMoA, self).__init__(
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,  
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer, 
            **kwargs)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] 
        self.blocks = nn.Sequential(*[
            MoABlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.apply(self._init_weights)

class VisionTransformerMoE(VisionTransformer):
    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None, 
                 num_attn_experts=48, head_dim=None,
                 moe_type='normal', 
                 **kwargs):
        super(VisionTransformerMoE, self).__init__(
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,  
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer, 
            **kwargs)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] 
        self.blocks = nn.Sequential(*[
            MoEBlock(
                num_attn_experts=num_attn_experts, head_dim=head_dim,
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                moe_type=moe_type,
                )
            for i in range(depth)])

        self.apply(self._init_weights)
        self.gate_num = torch.zeros(48).float()
        self.gate_num.requires_grad=False

        for blk in self.blocks:
            if moe_type != 'random':
                blk.attn.q_proj.f_gate.data.fill_(0.00)

    def moa_init_weight(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.fill_(0.00)

    def clear(self):
        self.gate_num = torch.zeros(48).float()
        self.gate_num.requires_grad=False

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        z_loss = 0

        now = 0
        vis = 0

        for blk in self.blocks:
            x, aux_loss = blk(x)
            z_loss = z_loss + aux_loss + vis * 0

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome, z_loss

class VisionTransformerFixMoE(VisionTransformer):
    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None, 
                 num_attn_experts=48, head_dim=None,
                 moe_type='normal', 
                 **kwargs):
        super(VisionTransformerFixMoE, self).__init__(
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,  
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer, 
            **kwargs)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] 
        self.blocks = nn.Sequential(*[
            MoEBlock(
                num_attn_experts=num_attn_experts, head_dim=head_dim,
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                moe_type=moe_type,
                )
            for i in range(depth)])

        self.apply(self._init_weights)
        self.gate_num = torch.zeros(48).float()
        self.gate_num.requires_grad=False

        for blk in self.blocks:
            if moe_type != 'random':
                # blk.attn.q_proj.f_gate.data.fill_(0.00)
                torch.nn.init.xavier_uniform_(blk.attn.q_proj.f_gate.data)
                blk.attn.q_proj.f_gate.requires_grad = False

    def moa_init_weight(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.fill_(0.00)

    def clear(self):
        self.gate_num = torch.zeros(48).float()
        self.gate_num.requires_grad=False

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        z_loss = 0

        now = 0
        vis = 0

        for blk in self.blocks:
            x, aux_loss = blk(x)
            # z_loss = z_loss + aux_loss + vis * 0

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome, z_loss

class VisionTransformerMoEAll(VisionTransformer):
    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None, 
                 num_attn_experts=48, head_dim=None,
                 num_ffd_experts=16, ffd_heads=2, ffd_noise=True,
                 moe_type='normal',
                 **kwargs):
        super(VisionTransformerMoEAll, self).__init__(
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,  
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer, 
            **kwargs)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] 
        self.blocks = nn.Sequential(*[
            MoEnhanceBlock(
                num_attn_experts=num_attn_experts, head_dim=head_dim,
                num_ffd_experts=num_ffd_experts, ffd_heads=ffd_heads, ffd_noise=ffd_noise,
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                moe_type=moe_type,
                )
            for i in range(depth)])

        self.apply(self._init_weights)
        self.gate_num = torch.zeros(48).float()
        self.gate_num.requires_grad=False

        for blk in self.blocks:
            if moe_type != 'random':
                blk.attn.q_proj.f_gate.data.fill_(0.00)
                blk.mlp.f_gate.data.fill_(0.00)

    def moa_init_weight(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.fill_(0.00)

    def clear(self):
        self.gate_num = torch.zeros(48).float()
        self.gate_num.requires_grad=False

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        z_loss = 0

        now = 0
        vis = 0

        for blk in self.blocks:
            x, aux_loss = blk(x)
            z_loss = z_loss + aux_loss + vis * 0

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome, z_loss

class VisionTransformerMoEMlp(VisionTransformer):
    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None, 
                 num_attn_experts=48, head_dim=None,
                 **kwargs):
        super(VisionTransformerMoEMlp, self).__init__(
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,  
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer, 
            **kwargs)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] 

        self.blocks = nn.ModuleList([
            MoEMlpBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, head_dim=head_dim, qkv_bias=True, norm_layer=norm_layer,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],)
            for i in range(depth)])

        self.apply(self._init_weights)
        
        for blk in self.blocks:
            blk.mlp.net.f_gate.apply(self.moa_init_weight)

        self.gate_num = torch.zeros(16).float()
        self.gate_num.requires_grad=False

        self.gate_num_top1 = torch.zeros(16).float()
        self.gate_num_top1.requires_grad=False


    def moa_init_weight(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.fill_(0.00)

    def clear(self):
        self.gate_num = torch.zeros(48).float()
        self.gate_num.requires_grad=False
        self.gate_num_top1 = torch.zeros(16).float()
        self.gate_num_top1.requires_grad=False

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        z_loss = 0

        now = 0
        vis = 0

        for blk in self.blocks:
            now = now + 1
            if now == 4:
                xx = x + blk.drop_path(blk.attn(blk.norm1(x), mask=None))
                vis = blk.mlp.net.f_gate(blk.norm2(xx)) # B, N, expert
                choose = torch.argmax(vis, dim=-1) # B, N
                for i in range(0,16):
                    self.gate_num_top1[i] = self.gate_num_top1[i] + (choose==i).sum().item()

                _, choose = torch.topk(vis, 2, dim=-1)
                for i in range(0,16):
                    self.gate_num[i] = self.gate_num[i] + (choose==i).sum().item()
                vis = vis.sum()
                z_loss = z_loss + vis * 0

            x, aux_loss = blk(x)
            z_loss = z_loss + aux_loss

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome, z_loss

def vit_moemlp_base_patch16(**kwargs):
    model = VisionTransformerMoEMlp(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, head_dim=768//12 * 2, qkv_bias=True,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_moe_base_patch16(**kwargs):
    model = VisionTransformerMoE(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=12*8, head_dim=768//12 * 2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_moa_base_patch16(**kwargs):
    model = VisionTransformerMoA(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_tiny(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_moa_tiny(**kwargs):
    model = VisionTransformerMoA(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_moe_tiny(**kwargs):
    model = VisionTransformerMoE(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=3*8, head_dim=192//3 * 2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_enmoe_tiny(**kwargs):
    model = VisionTransformerMoEAll(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=3*8, head_dim=192//3 * 2,
        num_ffd_experts=12, ffd_heads=2, ffd_noise=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_pretrain_tiny(**kwargs):
    model = VisionTransformerMoEAll(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=3, head_dim=192//3 * 2,
        num_ffd_experts=2, ffd_heads=2, ffd_noise=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_pretrain_small(**kwargs):
    model = VisionTransformerMoEAll(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=6, head_dim=384//6 * 2,
        num_ffd_experts=2, ffd_heads=2, ffd_noise=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_pretrain_base(**kwargs):
    model = VisionTransformerMoEAll(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=12, head_dim=768//12 * 2,
        num_ffd_experts=2, ffd_heads=2, ffd_noise=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_task0_tiny(**kwargs):
    model = VisionTransformerMoEAll(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=3, head_dim=192//3 * 2,
        num_ffd_experts=2, ffd_heads=2, ffd_noise=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_task1_tiny(**kwargs):
    model = VisionTransformerMoEAll(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=3 * 2, head_dim=192//3 * 2,
        num_ffd_experts=2 * 2, ffd_heads=2, ffd_noise=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_task0_small(**kwargs):
    model = VisionTransformerMoEAll(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=6, head_dim=384//6 * 2,
        num_ffd_experts=2, ffd_heads=2, ffd_noise=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_task1_small(**kwargs):
    model = VisionTransformerMoEAll(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=6 + 3, head_dim=384//6 * 2,
        num_ffd_experts=2 + 2, ffd_heads=2, ffd_noise=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_task2_small(**kwargs):
    model = VisionTransformerMoEAll(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=6 + 3 * 2, head_dim=384//6 * 2,
        num_ffd_experts=2 + 2 * 2, ffd_heads=2, ffd_noise=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_task3_small(**kwargs):
    model = VisionTransformerMoEAll(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=6 + 3 * 3, head_dim=384//6 * 2,
        num_ffd_experts=2 + 2 * 3, ffd_heads=2, ffd_noise=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_task4_small(**kwargs):
    model = VisionTransformerMoEAll(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=6 + 3 * 4, head_dim=384//6 * 2,
        num_ffd_experts=2 + 2 * 4, ffd_heads=2, ffd_noise=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
    
def vit_task1_base(**kwargs):
    model = VisionTransformerMoEAll(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=12 + 3 * 1, head_dim=768//12 * 2,
        num_ffd_experts=2 * 2, ffd_heads=2, ffd_noise=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_task4_base(**kwargs):
    model = VisionTransformerMoEAll(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=12 + 3 * 2, head_dim=768//12 * 2,
        num_ffd_experts=2 * 5, ffd_heads=2, ffd_noise=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_task1_little_small(**kwargs):
    model = VisionTransformerMoEAll(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=6 + 3, head_dim=384//6 * 2,
        num_ffd_experts=2 + 2, ffd_heads=2, ffd_noise=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_task2_tiny(**kwargs):
    model = VisionTransformerMoEAll(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=3 * 3, head_dim=192//3 * 2,
        num_ffd_experts=2 * 3, ffd_heads=2, ffd_noise=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_task3_tiny(**kwargs):
    model = VisionTransformerMoEAll(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=3 * 4, head_dim=192//3 * 2,
        num_ffd_experts=2 * 4, ffd_heads=2, ffd_noise=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_task4_tiny(**kwargs):
    model = VisionTransformerMoEAll(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=3 * 5, head_dim=192//3 * 2,
        num_ffd_experts=2 * 5, ffd_heads=2, ffd_noise=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_task5_tiny(**kwargs):
    model = VisionTransformerMoEAll(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=3 * 6, head_dim=192//3 * 2,
        num_ffd_experts=2 * 6, ffd_heads=2, ffd_noise=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_task6_tiny(**kwargs):
    model = VisionTransformerMoEAll(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=3 * 7, head_dim=192//3 * 2,
        num_ffd_experts=2 * 7, ffd_heads=2, ffd_noise=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_hugemoe_tiny(**kwargs):
    model = VisionTransformerMoE(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=3*16, head_dim=192//3 * 2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_rmoe_tiny(**kwargs):
    model = VisionTransformerMoE(
        moe_type='random',
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=3*8, head_dim=192//3 * 2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_moa_small(**kwargs):
    model = VisionTransformerMoA(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_moe_small(**kwargs):
    model = VisionTransformerMoE(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=6*8, head_dim=384//6 * 2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_fix_moe_small(**kwargs):
    model = VisionTransformerFixMoE(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=6*8, head_dim=384//6 * 2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_enmoe_small(**kwargs):
    model = VisionTransformerMoEAll(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=6*8, head_dim=384//6 * 2, 
        num_ffd_experts=16, ffd_heads=2, ffd_noise=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_enmoe_12_small(**kwargs):
    model = VisionTransformerMoEAll(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=6*8, head_dim=384//6 * 2,
        num_ffd_experts=12, ffd_heads=2, ffd_noise=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_hugemoe_small(**kwargs):
    model = VisionTransformerMoE(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=6*16, head_dim=384//6 * 2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_rmoe_small(**kwargs):
    model = VisionTransformerMoE(
        moe_type='random',
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=6*4, head_dim=384//6 * 2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model