import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class JsonlDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.offsets = []

        with open(path, "r", encoding="utf-8") as f:
            pos = 0
            for line in f:
                self.offsets.append(pos)
                pos += len(line.encode("utf-8"))

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        with open(self.path, "r", encoding="utf-8") as f:
            f.seek(self.offsets[idx])
            line = f.readline()
            sample = json.loads(line)

        return sample  # {"words": ..., "tags": ...}

def create_data_loader(
    data_path,
    tokenizer,
    label_to_index,
    batch_size=32,
    num_workers=4,
    max_length=128,
    shuffle=True,
):
    dataset = JsonlDataset(data_path)

    def collate_fn(batch):
        words_batch = [item['words'] for item in batch]
        labels_batch = [
            [label_to_index[tag] for tag in item['tags']]
            for item in batch
        ]

        return batch_tokenization(
            words_batch,
            labels_batch,
            tokenizer=tokenizer,
            max_length=max_length
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

def batch_tokenization(words_batch, labels_batch, tokenizer, max_length):
    
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    for words, labels in zip(words_batch, labels_batch):
        
        # Initialize with BOS token (<s>)
        input_ids = [tokenizer.bos_token_id]
        aligned_labels = [-100] 

        for word, label in zip(words, labels):
            sub_tokens = tokenizer.tokenize(word)
            
            if not sub_tokens:
                continue
                
            sub_ids = tokenizer.convert_tokens_to_ids(sub_tokens)
            input_ids.extend(sub_ids)

            # Label assignment: 
            aligned_labels.append(label)
            aligned_labels.extend([-100] * (len(sub_tokens) - 1))

        # Add EOS token (</s>)
        input_ids.append(tokenizer.eos_token_id)
        aligned_labels.append(-100)

        pad_length = max_length - len(input_ids)
        if pad_length < 0:
            raise
        
        attention_mask = [1] * len(input_ids) + [0] * pad_length
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_length
        aligned_labels = aligned_labels + [-100] * pad_length

        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(aligned_labels)

    # Convert lists to PyTorch Tensors
    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long)
    }