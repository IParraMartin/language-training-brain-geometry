"""Short adaptation runs that train a single module of a frozen causal language model.

``run_embedding_ablation`` trains only the input embedding matrix (embedding-only adaptation); it saves the
adapted model and tokenizer locally and can push them to the Hub.
"""

from __future__ import annotations

import os
from pathlib import Path

# HF_HOME has to be set before transformers and datasets are imported
_cache = str(Path(__file__).resolve().parents[3] / "cache")
os.makedirs(_cache, exist_ok=True)
os.environ.setdefault("HF_HOME", _cache)

import numpy as np
import torch
import wandb
from datasets import Dataset as HFDataset, DatasetDict
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from brain_code.training.data import _TokenChunkDataset

_RESULTS_DIR = Path("results") / "models"
_SEED = 42


def _load_texts(
    data: list[str] | str | Path | HFDataset | DatasetDict,
    text_column: str,
    split: str
) -> list[str]:
    """Resolves a list, a one-example-per-line text file, a Dataset or a DatasetDict to a list of texts."""
    if isinstance(data, (str, Path)):
        with open(data, "r", encoding="utf-8") as fh:
            return [line for line in fh.read().splitlines() if line.strip()]
    if isinstance(data, list):
        return data

    if isinstance(data, DatasetDict):
        if split not in data:
            available = list(data.keys())
            raise ValueError(f"Split '{split}' not found in DatasetDict. Available splits: {available}")
        data = data[split]

    if isinstance(data, HFDataset):
        if text_column not in data.column_names:
            raise ValueError(
                f"Column '{text_column}' not found in dataset. "
                f"Available columns: {data.column_names}"
            )
        data = [row for row in data[text_column] if row and row.strip()]
    return data


def _select_device() -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.mps.is_available()
        else "cpu"
    )


def _make_loader(
    texts: list[str],
    tokenizer,
    context_length: int,
    max_tokens: int,
    batch_size: int
) -> DataLoader:
    """Packs the texts into fixed-length chunks behind a seeded, shuffling loader."""
    dataset = _TokenChunkDataset(
        texts,
        tokenizer,
        context_length,
        max_tokens
    )
    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset has only {len(dataset)} chunks (each {context_length} tokens) "
            f"but batch_size={batch_size}. Increase max_tokens or reduce batch_size. "
            f"Need at least {batch_size * context_length:,} tokens."
        )
    generator = torch.Generator()
    generator.manual_seed(_SEED)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator
    )


def _train_steps(
    model,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    n_steps: int,
    desc: str,
    log_wandb: bool = False
) -> list[float]:
    """Runs n_steps optimizer steps, cycling through the loader, and returns the per-step losses."""
    model.train()
    step = 0
    losses: list[float] = []
    pbar = tqdm(total=n_steps, desc=desc, ncols=100)
    while step < n_steps:
        for batch in loader:
            if step >= n_steps:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            loss = model(input_ids=input_ids, labels=labels).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            pbar.update(1)
            if log_wandb:
                wandb.log({"loss": loss.item()}, step=step)

    pbar.close()
    model.eval()
    return losses


def freeze_all_but_token_embeddings(model) -> torch.nn.Embedding:
    """Freezes every parameter except the token embedding matrix and returns that embedding.

    In GPT-2 the output head is tied to this matrix, so updating it also changes the output projection;
    position embeddings, blocks and the final layer norm stay frozen.
    """
    for p in model.parameters():
        p.requires_grad = False
    emb = model.get_input_embeddings()
    emb.weight.requires_grad = True
    return emb


def run_embedding_ablation(
    model_id: str,
    english_texts: list[str] | str | Path | HFDataset | DatasetDict,
    n_steps: int = 500,
    max_tokens: int = 50000,
    lr: float = 1e-3,
    context_length: int = 1024,
    batch_size: int = 16,
    out_dir: Path | None = None,
    text_column: str = "code",
    split: str = "train",
    push_to_hub_: bool = False,
    use_wandb: bool = False
) -> str:
    """Fine-tunes only the input embedding layer of a model on English text.

    The corpus should be large enough to move the stimulus-token embeddings out of the initialization regime
    but small enough not to retrain the model body.

    Args:
        model_id: Hub model ID or local checkpoint path.
        english_texts: English text, as a list of examples, a path to a text file with one paragraph or
            sentence per line, a ``datasets.Dataset`` (``text_column`` is read) or a ``datasets.DatasetDict``
            (``split`` is selected first).
        n_steps: Number of optimizer steps.
        max_tokens: Maximum number of tokens drawn from ``english_texts``.
        lr: AdamW learning rate.
        context_length: Sequence length of the packed training chunks.
        batch_size: Training batch size.
        out_dir: Directory for the ablated model; defaults to ``results/models/ablated_model/``.
        text_column: Column read from a Dataset or DatasetDict.
        split: Split selected from a DatasetDict.
        push_to_hub_: Also push the model and tokenizer to ``IParraMartin/<base-model-name>-ft``.
        use_wandb: Start a Weights & Biases run and log the loss at every step.

    Returns:
        Absolute path to the saved ablated model directory.

    Raises:
        ValueError: If the data yields fewer chunks than one batch.
    """
    if use_wandb:
        wandb.init(
            config={
                "model_id": model_id,
                "n_steps": n_steps,
                "max_tokens": max_tokens,
                "lr": lr,
                "context_length": context_length,
                "batch_size": batch_size
            },
            project="brain-embed-ft"
        )

    out_dir = Path(out_dir or (_RESULTS_DIR / "ablated_model"))
    out_dir.mkdir(parents=True, exist_ok=True)
    english_texts = _load_texts(english_texts, text_column, split)

    device = _select_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.to(device)

    emb = freeze_all_but_token_embeddings(model)

    n_trainable = emb.weight.numel()
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"Frozen model body.  Trainable: {n_trainable:,} / {n_total:,} params "
        f"({100 * n_trainable / n_total:.2f}% — embedding layer only)"
    )
    log_wandb = use_wandb and wandb.run is not None
    if log_wandb:
        wandb.run.summary["trainable_params"] = n_trainable

    set_seed(_SEED)
    loader = _make_loader(
        english_texts,
        tokenizer,
        context_length,
        max_tokens,
        batch_size
    )
    optimizer = AdamW([emb.weight], lr=lr)
    losses = _train_steps(
        model,
        loader,
        optimizer,
        device,
        n_steps,
        desc="embedding ablation",
        log_wandb=log_wandb
    )

    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    mean_loss = float(np.mean(losses[-50:])) if losses else float("nan")

    base_name = model_id.split("/")[-1]
    hub_repo_id = f"IParraMartin/{base_name}-ft"
    if push_to_hub_:
        model.push_to_hub(hub_repo_id)
        tokenizer.push_to_hub(hub_repo_id)
        print(f"Pushed ablated model -> https://huggingface.co/{hub_repo_id}")

    print(
        f"\nAblated model saved -> {out_dir}\n"
        f"Final mean loss (last 50 steps): {mean_loss:.4f}\n"
        f"Next: recompute brain scores on this checkpoint and compare to {model_id}"
    )

    if log_wandb:
        wandb.finish()

    return str(out_dir.resolve())
