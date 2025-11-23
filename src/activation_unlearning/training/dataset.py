import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class SFTDataset(Dataset):
    """
    Supervised Fine-Tuning dataset for Qwen2.5 / LLaMA-style causal LMs.
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        add_eos: bool = True,
    ):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        if "prompt" not in self.df.columns:
            raise ValueError("CSV must contain a 'prompt' column.")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_eos = add_eos

        # Ensure pad token is valid
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pad_token_id = self.tokenizer.pad_token_id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        text = str(self.df.iloc[idx]["prompt"]).strip()

        if self.add_eos:
            text = text + self.tokenizer.eos_token

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
        attn_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long)
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
            "text": text,
            "pad_token_id": self.pad_token_id,
        }

    @staticmethod
    def collate_fn(batch):
        max_len = max(len(x["input_ids"]) for x in batch)

        pad_token_id = batch[0]["pad_token_id"]

        input_ids_list = []
        attn_mask_list = []
        labels_list = []

        for sample in batch:
            ids = sample["input_ids"]
            mask = sample["attention_mask"]
            labels = sample["labels"]

            pad_len = max_len - len(ids)

            ids = torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])

            input_ids_list.append(ids)
            attn_mask_list.append(mask)
            labels_list.append(labels)

        return {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attn_mask_list),
            "labels": torch.stack(labels_list),
        }
