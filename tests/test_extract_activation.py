import os
import shutil
import torch
import numpy as np
import pytest
from unittest.mock import MagicMock
import pandas as pd

from activation_unlearning.probing.extract import (
    load_latest_prompts,
    capture_activations,
)


# ----------------------------------------------------------------------
# 1. Test prompt loader (no logs directory)
# ----------------------------------------------------------------------
def test_load_latest_prompts_no_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prompts = load_latest_prompts("logs")
    assert prompts == [], "Expected empty prompt list when logs directory missing"


# ----------------------------------------------------------------------
# 2. Test prompt loader with empty logs directory
# ----------------------------------------------------------------------
def test_load_latest_prompts_empty_dir(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    prompts = load_latest_prompts(str(log_dir))
    assert prompts == [], "Expected empty prompt list when no JSON logs exist"


# ----------------------------------------------------------------------
# 3. Test prompt loader with valid fake JSON
# ----------------------------------------------------------------------
def test_load_latest_prompts_valid(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    json_path = log_dir / "recommender_20250101_120000.json"
    json_path.write_text(
        """{
            "records": [
                {"question": "Test prompt 1"},
                {"question": "Test prompt 2"}
            ]
        }"""
    )

    prompts = load_latest_prompts(str(log_dir))
    assert len(prompts) == 2
    assert prompts[0] == "Test prompt 1"
    assert prompts[1] == "Test prompt 2"

