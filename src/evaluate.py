from nltk.translate.bleu_score import corpus_bleu
import time
import torch
from torch.utils.data import DataLoader

def evaluate(model, dataloader, tokenizer):
    model.eval()
    predictions, references = [], []
    start = time.time()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"]
            labels = batch["labels"]

            input_ids = input_ids.to(next(model.parameters()).device)
            labels = labels.to(next(model.parameters()).device)

            pred_ids = model.decode(input_ids) 
            preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

            predictions.extend(preds)
            references.extend([[r] for r in refs])

    latency = (time.time() - start) / len(predictions)
    bleu = corpus_bleu(references, predictions)
    return bleu, latency
