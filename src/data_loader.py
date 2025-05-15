import datasets

def load_dataset(name="wmt14", lang_pair=("en", "de")):
    dataset = datasets.load_dataset(name, lang_pair)
    return dataset["train"], dataset["validation"]

def preprocess(example, tokenizer, max_length=128):
    source = tokenizer(example["translation"]["en"], truncation=True, padding="max_length", max_length=max_length)
    target = tokenizer(example["translation"]["de"], truncation=True, padding="max_length", max_length=max_length)
    return {"input_ids": source.input_ids, "labels": target.input_ids}