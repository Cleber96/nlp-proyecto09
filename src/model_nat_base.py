import torch.nn as nn
from transformers import BertModel

class NATBase(nn.Module):
    def __init__(self, encoder_name="bert-base-uncased", vocab_size=30522):
        super().__init__()
        self.encoder = BertModel.from_pretrained(encoder_name)
        self.decoder = nn.TransformerDecoderLayer(d_model=768, nhead=8)
        self.out_proj = nn.Linear(768, vocab_size)

    def forward(self, src_embeddings, tgt_mask):
        hidden = self.decoder(src_embeddings, src_embeddings, tgt_mask=tgt_mask)
        return self.out_proj(hidden)