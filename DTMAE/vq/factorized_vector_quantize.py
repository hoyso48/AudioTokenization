from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# from torch.nn.utils import weight_norm
# from torch.nn.utils.parametrizations import weight_norm

class FactorizedVectorQuantize(nn.Module):
    def __init__(self, dim, codebook_size, codebook_dim, commitment, **kwargs):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment = commitment
        
        if dim != self.codebook_dim:
            self.in_proj = nn.Linear(dim, self.codebook_dim)
            self.out_proj = nn.Linear(self.codebook_dim, dim)
        else:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()
        self._codebook = nn.Embedding(codebook_size, self.codebook_dim)
    
    @property
    def codebook(self):
        return self._codebook

    def forward(self, z):
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z :
            - Dense path: Tensor[B x T x D]
            - Varlen path: Tensor[total_tokens x D]

        Returns
        -------
        Tensor[B x T x D]
            Quantized continuous representation of input (dense path)
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[1]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        Tensor[B x T x D]
            Projected latents (continuous representation of input before quantization, dense path)
        """
        # Dense vs varlen handling (based on tensor rank)
        is_varlen = (z.dim() == 2)

        if is_varlen:
            # Expect z: [total_tokens, D]
            # Prepare (b,t,d) as (1,total,D) for linear
            z_bt_d = z.unsqueeze(0)
            z_e_bt_d = self.in_proj(z_bt_d)  # (1, total, D)
            z_e = rearrange(z_e_bt_d, "b t d -> b d t")  # (1, D, total)
            z_q, indices = self.decode_latents(z_e)  # z_q: (1, D, total); indices: (1, total)
        else:
            # Expect z: [B, T, D]
            z_e_bt_d = self.in_proj(z)  # (B, T, D)
            z_e = rearrange(z_e_bt_d, "b t d -> b d t")  # (B, D, T)
            z_q, indices = self.decode_latents(z_e)
        

        if self.training:
            commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction='none') * self.commitment
            codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction='none')
            commit_loss = commitment_loss + codebook_loss
        else:
            commit_loss = torch.zeros(z_e.shape, device = z.device)

        z_q = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        # Project back to model dim
        z_q_bt_d = rearrange(z_q, "b d t -> b t d")
        z_q_bt_d = self.out_proj(z_q_bt_d)
        # For dense, return [B, T, D]; for varlen, handled below
        z_q_out_bt_d = z_q_bt_d

        if is_varlen:
            # Return varlen shape: [total_tokens, D]
            z_q_var = z_q_out_bt_d.squeeze(0)  # (total, D)
            indices_var = indices.squeeze(0)  # (total,)
            return z_q_var, indices_var, commit_loss
        else:
            return z_q_out_bt_d, indices, commit_loss

    def vq2emb(self, vq, proj=True):
        emb = self.embed_code(vq)
        if proj:
            emb = self.out_proj(emb)
        return emb

    def get_emb(self):
        return self.codebook.weight

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices