import os
import pytest
import pandas as pd
from transformers import AutoTokenizer
from activation_unlearning.training.dataset import SFTDataset


DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "src",
    "activation_unlearning",
    "data",
    "prompt_set.csv",
)

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_dataset_file_exists(model_name):
    assert os.path.exists(DATA_PATH), f"Dataset file not found: {DATA_PATH}"


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_dataset_load(model_name):
    df = pd.read_csv(DATA_PATH)
    assert not df.empty, "Dataset CSV is empty"
    assert "prompt" in df.columns, "Missing required column: prompt"


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_tokenizer_load(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    assert tok is not None
    encoded = tok("test prompt", return_tensors="pt")
    assert "input_ids" in encoded
    assert encoded["input_ids"].shape[1] > 0


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_sft_dataset(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds = SFTDataset(DATA_PATH, tokenizer)

    # dataset length
    assert len(ds) > 0, "SFTDataset returned 0 samples"

    # get one sample
    sample = ds[0]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "labels" in sample
    assert "text" in sample

    # check shapes
    L = sample["input_ids"].shape[0]
    assert L > 0, "Tokenized prompt length must be > 0"
    assert sample["labels"].shape[0] == L, "Labels must match token length"


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_batch_collation(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds = SFTDataset(DATA_PATH, tokenizer)

    # simulate a batch of 4
    batch_samples = [ds[i] for i in range(4)]
    batch = SFTDataset.collate_fn(batch_samples)

    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch

    B, T = batch["input_ids"].shape
    assert B == 4
    assert T > 0
    assert batch["labels"].shape == (B, T)
