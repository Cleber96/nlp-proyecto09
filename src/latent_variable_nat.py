import torch
from torch import nn
from .model_nat_base import NATBase

class LatentNAT(NATBase):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_encoder = nn.Linear(768, latent_dim)
        self.latent_decoder = nn.Linear(latent_dim, 768)

    def sample_latent(self, src_embedding):
        z = self.latent_encoder(src_embedding.mean(dim=1))
        return self.latent_decoder(z).unsqueeze(1).repeat(1, src_embedding.size(1), 1)

    def forward(self, src_embeddings):
        latent_context = self.sample_latent(src_embeddings)
        return super().forward(latent_context, tgt_mask=None)