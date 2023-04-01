import torch
import torch.nn as nn

from torchvision.utils import make_grid, save_image
from PIL import Image
import functools
from torch import nn

# Note: I remove Positional Embedding here
class WordEmbedding(nn.Module):
    r"""
    A :class:`~torch.nn.Module` for learned word embeddings and position
    embeddings for input tokens. Each token is mapped to a fixed dimensional
    word embedding; and corresponding positional embedding based on its index.
    These are summed together followed by layer normalization and an optional
    dropout.

    Args:
        vocab_size: Size of token vocabulary.
        hidden_size: Size of token embedding vectors.
        dropout: Probability for final dropout applied after layer normalization.
        max_caption_length: Maximum length of input captions; this is used to create a
            fixed positional embedding lookup table.
        padding_idx: Token index of ``[PAD]`` token, word embedding for these tokens
            will be a vector of zeroes (and not trainable).
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        dropout: float = 0.0,
        max_caption_length: int = 30,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx

        self.words = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)

        # We provide no "padding index" for positional embeddings. We zero out
        # the positional embeddings of padded positions as a post-processing.
        # self.positions = nn.Embedding(max_caption_length, hidden_size)
        self.layer_norm = nn.LayerNorm(
            hidden_size, eps=1e-8, elementwise_affine=True
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        r"""
        Get combined word and positional embeddings for input tokens.

        Args:
            tokens: A tensor of shape ``(batch_size, max_caption_length)``
                containing a batch of caption tokens, values in ``[0, vocab_size)``.

        Returns:
            A tensor of shape ``(batch_size, max_caption_length, hidden_size)``
            containing corresponding token embeddings.
        """
        # position_indices = self._create_position_indices(tokens)

        # shape: (batch_size, max_caption_length, hidden_size)
        word_embeddings = self.words(tokens)
        # position_embeddings = self.positions(position_indices)

        # shape: (batch_size, max_caption_length, hidden_size)
        # embeddings = self.layer_norm(word_embeddings + position_embeddings)
        embeddings = self.layer_norm(word_embeddings)
        embeddings = self.dropout(embeddings)

        # Zero-out embeddings for positions which have padding tokens.
        # shape: (batch_size, max_caption_length, 1)
        token_mask = (tokens != self.padding_idx).unsqueeze(-1)

        # shape: (batch_size, max_caption_length, hidden_size)
        embeddings = embeddings * token_mask.type(embeddings.dtype)
        return embeddings

    @functools.lru_cache(maxsize=128)
    def _create_position_indices(self, tokens: torch.Tensor):

        # Create position indices of the same size as token indices.
        batch_size, max_caption_length = tokens.size()
        positions = torch.arange(
            max_caption_length, dtype=tokens.dtype, device=tokens.device
        )
        # shape: (batch_size, max_caption_length)
        positions = positions.unsqueeze(0).expand(batch_size, max_caption_length)
        return positions


def model_save(fn, model, optimizer, scheduler, epoch):
    with open(fn, 'wb') as f:
        torch.save([model, optimizer, scheduler, epoch], f)


def model_load(fn, model, optimizer, scheduler, epoch):
    with open(fn, 'rb') as f:
        model, optimizer, scheduler, epoch = torch.load(f)
    return model, optimizer, scheduler, epoch

def create_d_vae(weight_path, d_vae_type, image_size, device):
    if d_vae_type == "dall-e":
        return get_dalle_vae(weight_path, image_size, device)
    elif d_vae_type == "customized":
        assert False
        # return get_d_vae(weight_path, image_size, device)
    else:
        raise NotImplementedError()

# logit_laplace_eps: float = 0.1
# def map_pixels(x: torch.Tensor) -> torch.Tensor:
#     if x.dtype != torch.float:
#         raise ValueError('expected input to have type float')

#     return (1 - 2 * logit_laplace_eps) * x + logit_laplace_eps

def get_dalle_vae(weight_path, image_size, device):
    vae = Dalle_VAE(image_size)
    vae.load_model(model_dir=weight_path, device=device)
    return vae

import io, requests
def load_model(path: str, device: torch.device = None) -> nn.Module:
    if path.startswith('http://') or path.startswith('https://'):
        resp = requests.get(path)
        resp.raise_for_status()
            
        with io.BytesIO(resp.content) as buf:
            return torch.load(buf, map_location=device)
    else:
        with open(path, 'rb') as f:
            return torch.load(f, map_location=device)

def images2grid(images, **grid_kwargs):
    # images should be (N, C, H, W)
    grid = make_grid(images, **grid_kwargs)
    out = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return out

# Note: images show be unnormalize if norm_pixel - True
def log_image_grid(writer, images, logging_name, itr, num=36):
    mean = torch.FloatTensor([ 0.485, 0.456, 0.406 ]).to(images.device)
    std = torch.FloatTensor([ 0.229, 0.224, 0.225 ]).to(images.device)
    img = images[:num]
    img = img * (std[None, :, None, None] + 0.0000001) + mean[None, :, None, None]
    nrow = max(1, int(img.size(0) ** 0.5))
    ndarr = images2grid(img, return_as_PIL=True, nrow=nrow, padding=2, pad_value=0, normalize=False, range=(0,1))
    writer.add_image(logging_name, ndarr, itr, dataformats='HWC')
