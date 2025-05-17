import torch

def train_model(model, dataloader, optimizer, criterion):
    model.train()
    for batch in dataloader:
        inputs, targets = batch["input_ids"], batch["labels"]
        optimizer.zero_grad()
        logits = model(inputs, tgt_mask=None)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()