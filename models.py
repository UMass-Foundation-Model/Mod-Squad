from models_mae import MaskedAutoencoderViT

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

    def forward_decoder(self, x, ids_restore, mask):
        # embed tokens
        x = self.decoder_embed(x)

        mask_token = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)

        x_ = torch.cat([x[:, 1:, :], mask_token], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        x_masked = x_[:, mask]
        x_unmasked = x_[:, ~mask]

        pos_embed_masked = self.decoder_pos_embed[:,1:][mask].reshape(batch_size, -1, dim)
        pos_embed_unmasked = self.decoder_pos_embed[:,1:][~mask].reshape(batch_size, -1, dim)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x_masked = blk(x_masked, torch.cat([x_unmasked, x_masked], dim=1), pos_embed_masked, torch.cat([pos_embed_unmasked, pos_embed_masked], dim=1))

        x = torch.zeros([x_masked.shape[0], x_masked.shape[1] + x_unmasked.shape[1], x_masked.shape[2]], device=x_masked.device)
        x[:, mask] = x[:, mask] + x_masked
        x[:, ~mask] = x[:, +mask] + x_unmasked

        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x, 0

def mae_cross_vit_base(**kwargs):
    model = MaskedAutoencoderCrossViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, head_dim=768//12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_cross_vit_tiny(**kwargs):
    model = MaskedAutoencoderCrossViT(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, head_dim=192//3,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model