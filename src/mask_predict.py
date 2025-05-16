import torch
import torch.nn.functional as F
from .model_nat_base import NATBase

class MaskPredictNAT(NATBase):
    def __init__(self, tokenizer, num_iterations=10):
        super().__init__()
        self.tokenizer = tokenizer
        self.num_iterations = num_iterations

    def decode(self, src_embeddings):
        seq_len = src_embeddings.size(1)
        output = torch.full((src_embeddings.size(0), seq_len), self.tokenizer.mask_token_id).to(src_embeddings.device)
        for _ in range(self.num_iterations):
            logits = self.forward(src_embeddings, tgt_mask=None)
            probs = F.softmax(logits, dim=-1)
            top_preds = torch.argmax(probs, dim=-1)
            output = torch.where(output == self.tokenizer.mask_token_id, top_preds, output)
        return output