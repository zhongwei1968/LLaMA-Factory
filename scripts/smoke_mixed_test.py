import json
from dataclasses import dataclass
from typing import Any, Optional

import os
import tempfile

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.hparams.data_args import DataArguments
from llamafactory.hparams.model_args import ModelArguments
from transformers import Seq2SeqTrainingArguments


class SimpleTokenizer:
    def __init__(self):
        self.vocab = {}
        self.inv_vocab = {}
        self._next_id = 1
        # special tokens
        self._bos_token = "<bos>"
        self._eos_token = "<eos>"
        self._pad_token = None
        self.add_bos_token = True
        # register default specials
        self._register_token(self._bos_token)
        self._register_token(self._eos_token)
        self.bos_token_id = self.vocab[self._bos_token]
        self.eos_token_id = self.vocab[self._eos_token]
        self.pad_token_id = None
        self.additional_special_tokens = []
        self.additional_special_tokens_ids = []
        self.chat_template = None

    # properties
    @property
    def bos_token(self):
        return self._bos_token

    @property
    def eos_token(self):
        return self._eos_token

    @property
    def pad_token(self):
        return self._pad_token

    @pad_token.setter
    def pad_token(self, value):
        if value is not None and value not in self.vocab:
            self._register_token(value)
        self._pad_token = value
        self.pad_token_id = None if value is None else self.vocab[value]

    # helpers
    def _register_token(self, token: str) -> int:
        if token in self.vocab:
            return self.vocab[token]
        tid = self._next_id
        self._next_id += 1
        self.vocab[token] = tid
        self.inv_vocab[tid] = token
        return tid

    # API surface used by the pipeline
    def add_special_tokens(self, special_tokens_dict: dict) -> int:
        num_added = 0
        if "eos_token" in special_tokens_dict and special_tokens_dict["eos_token"]:
            tok = special_tokens_dict["eos_token"]
            if tok != self._eos_token:
                if tok not in self.vocab:
                    self._register_token(tok)
                    num_added += 1
                self._eos_token = tok
                self.eos_token_id = self.vocab[tok]
        if "additional_special_tokens" in special_tokens_dict:
            for tok in special_tokens_dict["additional_special_tokens"]:
                if tok not in self.vocab:
                    self._register_token(tok)
                    num_added += 1
                if tok not in self.additional_special_tokens:
                    self.additional_special_tokens.append(tok)
                    self.additional_special_tokens_ids.append(self.vocab[tok])
        return num_added

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._register_token(token)

    def encode(self, text: str, add_special_tokens: bool = False):
        # very naive byte-level encoding
        ids = []
        for ch in text:
            tok = f"<c:{ord(ch)}>"
            ids.append(self._register_token(tok))
        return ids

    def decode(self, ids, skip_special_tokens: bool = False):
        s = []
        for i in ids:
            tok = self.inv_vocab.get(int(i), "")
            if skip_special_tokens and tok.startswith("<"):
                continue
            if tok.startswith("<c:") and tok.endswith(">"):
                try:
                    s.append(chr(int(tok[3:-1])))
                except Exception:
                    s.append("?")
            else:
                s.append(tok)
        return "".join(s)

    def __call__(self, texts: list[str], add_special_tokens: bool = False, truncation: bool = False, max_length: Optional[int] = None):
        input_ids = []
        attention_mask = []
        for text in texts:
            ids = self.encode(text, add_special_tokens=add_special_tokens)
            if truncation and max_length is not None:
                ids = ids[:max_length]
            input_ids.append(ids)
            attention_mask.append([1] * len(ids))
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def _write(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    # Auto-generate tiny datasets in a temp dir (no repo files required)
    with tempfile.TemporaryDirectory() as tmpdir:
        # dataset_info.json describing two local files: one SFT and one PT (unsupervised) sample
        dataset_info = {
            "smoke_sft": {"file_name": "smoke_sft.jsonl"},
            "smoke_pt": {"file_name": "smoke_pt.jsonl", "columns": {"response": None}},
        }
        _write(os.path.join(tmpdir, "dataset_info.json"), json.dumps(dataset_info))

        # SFT example: instruction + output
        _write(
            os.path.join(tmpdir, "smoke_sft.jsonl"),
            json.dumps({"instruction": "Write a short greeting.", "input": "", "output": "Hello world!"}) + "\n",
        )

        # PT example: instruction only (no response column)
        _write(
            os.path.join(tmpdir, "smoke_pt.jsonl"),
            json.dumps({"instruction": "List three fruits.", "input": ""}) + "\n",
        )

        # minimal args for dataset pipeline
        data_args = DataArguments(
            template="alpaca",
            dataset=["smoke_sft", "smoke_pt"],
            dataset_dir=tmpdir,
            cutoff_len=1024,
            val_size=0.0,
            mix_strategy="concat",
            streaming=False,
        )
        model_args = ModelArguments(model_name_or_path="dummy-model")
        training_args = Seq2SeqTrainingArguments(output_dir="scripts/_tmp_out", per_device_train_batch_size=2)

        tok = SimpleTokenizer()
        template = get_template_and_fix_tokenizer(tok, data_args)
        module = get_dataset(template, model_args, data_args, training_args, stage="sft", tokenizer=tok, processor=None)
        ds = module["train_dataset"]

        # iterate and check presence of labels for PT sample and IGNORE_INDEX patterns for SFT sample
        print(f"Total samples: {len(ds)}")
        stats = []
        for i, ex in enumerate(ds):
            labels = ex.get("labels")
            inps = ex.get("input_ids")
            # Heuristic: PT sample has labels equal to input_ids entirely (no -100)
            is_pt_like = labels is not None and len(labels) == len(inps) and (-100 not in labels)
            stats.append({"idx": i, "len": len(inps), "pt_like": is_pt_like, "has_labels": labels is not None})
            print(f"idx={i}, len={len(inps)}, pt_like={is_pt_like}, has_labels={labels is not None}")

        # Expect one sft-like (has -100) and one pt-like (labels==input_ids)
        has_pt = any(s["pt_like"] for s in stats)
        has_sft = any((s["has_labels"] and not s["pt_like"]) for s in stats)
        assert has_pt and has_sft, f"Mixed dataset not detected correctly: {stats}"
        print("Smoke test passed: mixed SFT+PT preprocessing works.")


if __name__ == "__main__":
    main()
