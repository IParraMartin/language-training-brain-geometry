from __future__ import annotations

import os
import json
import hashlib
import yaml
from pathlib import Path
from datetime import datetime

# HF_HOME has to be set before transformers and datasets are imported
_cache = str(Path(__file__).resolve().parents[3] / "cache")
os.makedirs(_cache, exist_ok=True)
os.environ.setdefault("HF_HOME", _cache)

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling
)
from transformers.trainer_utils import get_last_checkpoint

from datasets import DatasetDict

from brain_code.config import PATHS
from brain_code.utils.get_null_positions import null_positions


class LMTrainer():
    """Trainer wrapper for causal language model pretraining using HuggingFace Transformers.

    Initializes a model from scratch using a specified architecture config, tokenizes
    the provided dataset, and runs a full training loop via the HuggingFace `Trainer`
    API. Training metrics are logged to Weights & Biases.

    Attributes:
        cache_dir (str): Path to the directory used for caching tokenizer downloads.
        model_args (dict): Parsed model architecture arguments.
        training_args (dict): Parsed training hyperparameters.
        tokenizer (AutoTokenizer): Tokenizer loaded from the specified HuggingFace hub model.
        model (AutoModelForCausalLM): Causal language model instantiated from config.
        dataset (DatasetDict): The raw dataset containing 'train' and 'eval' splits.
        dataset_name (str): Short name for the dataset used to label the output directory.
        max_len (int): Maximum token sequence length used during tokenization.
    """

    def __init__(
        self,
        dataset: DatasetDict,
        training_args: dict | str,
        model_args: dict | str,
        tokenizer: str | None = PATHS["tokenizer"]["default"],
        dataset_name: str = "noname_data",
        null_pos: bool = False
    ) -> None:
        """Initializes LMTrainer by loading the tokenizer, building the model, and preparing config.

        The model is initialized from scratch using `AutoModelForCausalLM.from_config`,
        meaning no pretrained weights are loaded — only the architecture is borrowed from
        `model_args`. If `gradient_checkpointing` is enabled in `training_args`, it is
        activated on the model to reduce memory usage during backpropagation.

        Args:
            tokenizer (str): Uses a custom trained tokenizer if provided. Else, falls
                back to `gpt2` tokenizer.
            dataset (DatasetDict): A HuggingFace `DatasetDict` with at minimum ``'train'``
                and ``'eval'`` splits, each containing a ``'text'`` column.
            training_args (dict | str): Training hyperparameters. Either a dictionary
                of arguments compatible with `TrainingArguments`, or a path to a YAML
                file that will be parsed into one.
            model_args (dict | str): Model architecture arguments. Must include a
                ``'model_type'`` key (e.g., ``'gpt2'``). Either a dictionary or a path
                to a YAML file.
            dataset_name (str): Short name for the dataset, appended to the output
                directory alongside ``run_name`` and a timestamp so each run writes
                to a unique path (e.g., ``./out/gpt2_tinystories_20260517_143022``).
        """
        self.model_args = self.return_arguments(model_args)
        self.training_args = self.return_arguments(training_args)

        # seed before model construction, so identical seeds give
        # matched initial parameters across separate processes
        seed = int(self.training_args.get("seed", 42))
        deterministic = bool(self.training_args.pop("deterministic", False))
        from brain_code.training.reproducibility import seed_everything
        seed_everything(seed, deterministic=deterministic)
        from transformers import set_seed as _set_seed
        _set_seed(seed)

        if tokenizer is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer,
                use_fast=True,
                cache_dir=os.environ["HF_HOME"]
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "gpt2",
                use_fast=True,
                cache_dir=os.environ["HF_HOME"]
            )

        attn_impl = self.model_args.get("attn_implementation", "eager")
        self.model = AutoModelForCausalLM.from_config(
            self.get_model_config(),
            attn_implementation=attn_impl
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.training_args.get("gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable()

        if null_pos:
            self.model = null_positions(self.model)

        self.dataset = dataset
        self.dataset_name = dataset_name
        self.max_len = self.model_args["n_positions"]

        self.initialization_seed = seed
        self.deterministic_mode = deterministic

    def train(
        self,
        split_before_pack: bool = True,
        insert_eos_between_docs: bool = True,
        extra_callbacks: list | None = None,
        trainer_cls=None,
        trainer_kwargs: dict | None = None,
        timestamp_dir: bool = True,
        save_final: bool = True,
        resume: bool = False
    ) -> str:
        """Runs the full pretraining loop and finalizes the W&B run.

        Args:
            split_before_pack: If ``True`` (default), split the raw dataset into
                train/eval *before* tokenization and packing.  This prevents
                the same source document from contributing packed chunks to
                both splits.  When a built-in eval/validation split is present
                it is used directly and this flag has no effect.
            insert_eos_between_docs: If ``True`` (default), insert an EOS token
                between consecutive documents before packing.  This prevents
                cross-document attention at document boundaries.
            extra_callbacks: Optional list of additional HuggingFace
                ``TrainerCallback`` instances appended to the default
                ``EarlyStoppingCallback`` (e.g. ``TokenMilestoneCallback``).
            trainer_cls: Optional ``Trainer`` subclass to instantiate instead of
                the stock ``transformers.Trainer`` (e.g.
                ``TokenCountingTrainer`` for exact non-padding token accounting).
                Defaults to ``Trainer``.
            trainer_kwargs: Optional extra keyword arguments forwarded to
                ``trainer_cls`` (e.g. ``{"token_counter": counter}``).
            timestamp_dir: If ``True`` (default), append
                ``{run_name}_{dataset}_{timestamp}`` to ``output_dir`` so repeat
                runs never collide.  Set ``False`` when a downstream step needs a
                deterministic path (e.g. a factorial run that scores
                ``checkpoints/factorial/cond_A/seed_42`` right after training it).
            save_final: If ``True`` (default), write the final model and
                tokenizer to the resolved output directory.  Without this the
                only artifacts are periodic ``checkpoint-N/`` subdirectories, so
                the run directory itself is not loadable by ``from_pretrained``.
            resume: If ``True`` and the resolved output directory already holds
                a ``checkpoint-N/`` subdirectory, continue from its latest
                checkpoint instead of restarting from step 0. Opt-in, because a
                rerun after a config change must not inherit old optimizer state.
                Requires ``timestamp_dir=False``.

        Returns:
            str: The resolved output directory the model was written to.
        """
        seed = self.training_args.get("seed", 42)

        if "eval" in self.dataset:
            print("Using built-in eval split")
            raw_train = DatasetDict({"train": self.dataset["train"]})
            raw_eval = DatasetDict({"train": self.dataset["eval"]})
        elif "validation" in self.dataset:
            print("Using built-in validation split")
            raw_train = DatasetDict({"train": self.dataset["train"]})
            raw_eval = DatasetDict({"train": self.dataset["validation"]})
        elif split_before_pack:
            print("split_before_pack=True: splitting raw documents before tokenization...")
            raw_split = self.dataset["train"].train_test_split(test_size=0.1, seed=seed)
            n_train = len(raw_split["train"])
            n_eval = len(raw_split["test"])
            print(f"  Raw documents — train: {n_train:,}  eval: {n_eval:,}")
            raw_train = DatasetDict({"train": raw_split["train"]})
            raw_eval = DatasetDict({"train": raw_split["test"]})
        else:
            # post-pack split: one document may contribute to both train and eval
            print(
                "split_before_pack=False (seed-42 pipeline): splitting tokenized chunks.  "
                "WARNING: the same source document may contribute to both train and eval."
            )
            raw_train = self.dataset
            raw_eval = None

        print("Tokenizing train split...")
        tokenized_train = self.get_tokenized_dataset(raw_train, insert_eos_between_docs)
        train_dataset = tokenized_train["train"]

        if raw_eval is not None:
            print("Tokenizing eval split...")
            tokenized_eval = self.get_tokenized_dataset(raw_eval, insert_eos_between_docs)
            eval_dataset = tokenized_eval["train"]
        else:
            # post-pack split
            full_tokenized = self.get_tokenized_dataset(self.dataset, insert_eos_between_docs)
            split = full_tokenized["train"].train_test_split(test_size=0.1, seed=seed)
            train_dataset = split["train"]
            eval_dataset = split["test"]

        n_train_chunks = len(train_dataset)
        n_eval_chunks = len(eval_dataset)
        print(
            f"Packed chunks — train: {n_train_chunks:,}  eval: {n_eval_chunks:,}  "
            f"({n_train_chunks * self.max_len:,} train tokens, "
            f"{n_eval_chunks * self.max_len:,} eval tokens)"
        )
        print("Launching training...")

        run_name = self.training_args.get("run_name", "model")
        if timestamp_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.training_args["output_dir"] = os.path.join(
                self.training_args.get("output_dir", "./out"),
                f"{run_name}_{self.dataset_name}_{timestamp}"
            )
        out_dir = self.training_args.get("output_dir", "./out")
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        from brain_code.training.reproducibility import parameter_hash

        initial_parameter_hash = parameter_hash(self.model)
        tokenizer_payload = {
            "vocab": self.tokenizer.get_vocab(),
            "special_tokens": self.tokenizer.special_tokens_map,
            "init_kwargs": self.tokenizer.init_kwargs
        }
        tokenizer_hash = hashlib.sha256(
            json.dumps(
                tokenizer_payload,
                sort_keys=True,
                separators=(",", ":"),
                default=str
            ).encode()
        ).hexdigest()
        provenance_path = out_path / "TRAINING_PROVENANCE.json"
        provenance = {
            "initialization_seed": self.initialization_seed,
            "deterministic_mode": self.deterministic_mode,
            "initial_parameter_hash": initial_parameter_hash,
            "tokenizer_hash": tokenizer_hash,
            "model_args": self.model_args,
            "training_args": self.training_args,
            "dataset_name": self.dataset_name,
            "split_before_pack": bool(split_before_pack),
            "insert_eos_between_docs": bool(insert_eos_between_docs),
            "status": "training_started"
        }
        provenance_path.write_text(
            json.dumps(
                provenance,
                indent=2,
                sort_keys=True,
                default=str
            ) + "\n",
            encoding="utf-8"
        )

        patience = self.training_args.pop("patience", None)
        callbacks = []
        if patience is not None:
            if not self.training_args.get("load_best_model_at_end", False):
                raise ValueError(
                    "Training configuration sets 'patience' but disables "
                    "load_best_model_at_end. Hugging Face early stopping requires "
                    "best-model tracking; set patience: null to disable early "
                    "stopping or enable load_best_model_at_end."
                )
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(patience)))
        if extra_callbacks:
            callbacks.extend(extra_callbacks)
        trainer_class = trainer_cls if trainer_cls is not None else Trainer
        trainer = trainer_class(
            model=self.model,
            args=TrainingArguments(**self.training_args),
            data_collator=self.get_data_collator(),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            callbacks=callbacks,
            **(trainer_kwargs or {})
        )
        # walltime kills land mid-run, so a requeued job continues from its last checkpoint; a directory
        # without one is a normal fresh start
        resume_checkpoint = None
        if resume:
            if timestamp_dir:
                raise ValueError(
                    "resume=True requires timestamp_dir=False: a timestamped output directory is new on "
                    "every run and never holds a checkpoint to resume from."
                )
            resume_checkpoint = get_last_checkpoint(out_dir)
            if resume_checkpoint is None:
                print(f"Resume requested but no checkpoint under {out_dir}; starting fresh")
            else:
                print(f"Resuming from {resume_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_checkpoint)

        if save_final:
            # Periodic checkpoints land in `checkpoint-N/` subdirectories, which
            # leaves the run directory itself unloadable by `from_pretrained`.
            # Write the final (best, when load_best_model_at_end is set) weights
            # and the tokenizer to the run directory so downstream scoring can
            # point straight at it.
            trainer.save_model(out_dir)
            self.tokenizer.save_pretrained(out_dir)
            print(f"Final model saved to {out_dir}")

        provenance.update({
            "status": "complete",
            "final_parameter_hash": parameter_hash(trainer.model),
            "global_step": int(trainer.state.global_step),
            "best_model_checkpoint": trainer.state.best_model_checkpoint
        })
        provenance_path.write_text(
            json.dumps(
                provenance,
                indent=2,
                sort_keys=True,
                default=str
            ) + "\n",
            encoding="utf-8"
        )

        return out_dir

    def get_model_config(self) -> AutoConfig:
        """Builds an `AutoConfig` for the target model architecture.

        Reads ``'model_type'`` from `model_args` to determine the architecture, then
        passes the remaining arguments (e.g., ``n_layer``, ``n_head``, ``n_embd``) as
        keyword arguments alongside the tokenizer's vocabulary size.

        Returns:
            AutoConfig: A configuration object for the specified model type, populated
                with the provided architecture hyperparameters and the tokenizer's
                vocabulary size.

        Raises:
            ValueError: If ``'model_type'`` is not present in `model_args`.
        """
        model_type = self.model_args.get("model_type")
        if model_type is None:
            raise ValueError("'model_args' must include 'model_type' (e.g., 'gpt2', 'llama'...)")
        _skip = {"model_type", "attn_implementation"}
        config = AutoConfig.for_model(
            model_type,
            vocab_size=self.tokenizer.vocab_size,
            **{k: v for k, v in self.model_args.items() if k not in _skip}
        )
        # Align special-token ids with the (custom) tokenizer. AutoConfig.for_model
        # keeps the base model's defaults (e.g. gpt2 bos/eos = 50256), which are out
        # of range for a smaller custom vocab and disagree with the tokenizer's real
        # eos (id 0). Left unfixed this is a latent provenance / generate() bug.
        tok = self.tokenizer
        if tok.eos_token_id is not None:
            config.eos_token_id = tok.eos_token_id
        config.bos_token_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        if getattr(tok, "pad_token_id", None) is not None:
            config.pad_token_id = tok.pad_token_id
        return config

    def get_tokenized_dataset(self, dataset: DatasetDict, insert_eos_between_docs: bool = True) -> DatasetDict:
        """Tokenizes all splits in the dataset using batched processing.

        Args:
            dataset: The raw dataset to tokenize.
            insert_eos_between_docs: If ``True``, append an EOS token ID after
                every document's token sequence before concatenation.  This
                prevents cross-document attention at packing boundaries.

        Returns:
            DatasetDict with packed fixed-length chunks.
        """
        first_split = next(iter(dataset.values()))
        text_col = next((c for c in ("text", "code") if c in first_split.column_names), None)
        if text_col is None:
            raise ValueError(f"Dataset has none of the expected text columns. Got: {first_split.column_names}")
        tokenized_dataset = dataset.map(
            self.tokenize,
            batched=True,
            remove_columns=[text_col],
            desc="Tokenizing"
        )
        # pack into fixed-length blocks; remove_columns avoids row-count mismatch
        first_split = next(iter(tokenized_dataset.values()))
        group_fn = self.group_texts_with_eos if insert_eos_between_docs else self.group_texts
        tokenized_dataset = tokenized_dataset.map(
            group_fn,
            batched=True,
            remove_columns=first_split.column_names,
            desc="Grouping texts"
        )
        return tokenized_dataset

    def get_data_collator(self) -> DataCollatorForLanguageModeling:
        """Constructs a data collator for causal (autoregressive) language modeling.

        Configures `DataCollatorForLanguageModeling` with masked language modeling
        disabled (``mlm=False``), making it suitable for decoder-only, next-token
        prediction objectives. Labels are automatically shifted inside the model.

        Returns:
            DataCollatorForLanguageModeling: A collator that pads batches and returns
                PyTorch tensors, with no token masking applied.
        """
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            return_tensors="pt"
        )

    def tokenize(self, batch: dict) -> dict:
        """Tokenizes a batch of raw text examples.

        Wraps the tokenizer call with fixed padding and truncation settings.
        Sequences shorter than `max_len` are right-padded; longer sequences
        are truncated to `max_len`.

        Args:
            batch (dict): A batch dictionary with a ``'text'`` key mapping to a list
                of raw string examples, as produced by `datasets.DatasetDict.map`.

        Returns:
            dict: A dictionary containing tokenizer outputs — at minimum
                ``'input_ids'`` and ``'attention_mask'`` — each padded or truncated
                to `self.max_len`.
        """
        for col in ("text", "code", "assembly"):
            if col in batch:
                return self.tokenizer(
                    batch[col],
                    truncation=True,
                    max_length=self.max_len
                )
        raise ValueError(f"Batch has none of the expected columns. Got: {list(batch.keys())}")

    def group_texts(self, examples: dict) -> dict:
        """Concatenates all tokenized texts and rechunks into fixed-length blocks.

        Eliminates padding waste by packing documents end-to-end and slicing
        into sequences of exactly ``max_len`` tokens. Any remainder shorter
        than ``max_len`` is dropped. A ``'labels'`` key is added as a copy of
        ``'input_ids'`` to satisfy the causal LM training objective; the model
        applies the left-shift internally.

        Args:
            examples (dict): A batch of tokenized examples with ``'input_ids'``
                and ``'attention_mask'`` keys, each mapping to a list of lists
                of integer token ids, as produced by ``DatasetDict.map``.

        Returns:
            dict: Rechunked sequences of exactly ``self.max_len`` length, with
                a ``'labels'`` key copied from ``'input_ids'``.
        """
        # flatten all sequences into one long stream per key
        # only process sequence-type columns; non-list columns (e.g. metadata ints) are dropped
        seq_keys = [k for k in examples.keys() if examples[k] and isinstance(examples[k][0], list)]
        concatenated = {
            k: [tok for seq in examples[k] for tok in seq]
            for k in seq_keys
        }
        total_length = len(concatenated["input_ids"])
        # drop the remainder that doesn't fill a complete block
        total_length = (total_length // self.max_len) * self.max_len
        # split into fixed-length chunks
        result = {
            k: [concatenated[k][i : i + self.max_len] for i in range(0, total_length, self.max_len)]
            for k in concatenated.keys()
        }
        # labels are just a copy of input_ids; the model handles the shift internally
        result["labels"] = result["input_ids"].copy()
        return result

    def group_texts_with_eos(self, examples: dict) -> dict:
        """Like ``group_texts`` but inserts EOS between documents before packing.

        An EOS token is appended to each document's ``input_ids`` (and
        correspondingly to ``attention_mask``) before concatenation.  This
        prevents the model from learning cross-document dependencies at chunk
        boundaries.  The EOS token is also assigned a label so it is included
        in the loss.

        Document boundary count is logged to stdout.
        """
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError(
                "Tokenizer has no eos_token_id.  Cannot insert EOS between documents.  "
                "Set insert_eos_between_docs=False or configure a tokenizer with an EOS token."
            )
        seq_keys = [k for k in examples.keys() if examples[k] and isinstance(examples[k][0], list)]
        n_docs = len(examples[seq_keys[0]])
        concatenated: dict = {k: [] for k in seq_keys}
        for seq_idx in range(n_docs):
            for k in seq_keys:
                concatenated[k].extend(examples[k][seq_idx])
                if k == "input_ids":
                    concatenated[k].append(eos_id)
                elif k == "attention_mask":
                    concatenated[k].append(1)
                else:
                    concatenated[k].append(concatenated[k][-1] if concatenated[k] else 0)
        total_length = (len(concatenated["input_ids"]) // self.max_len) * self.max_len
        result = {
            k: [concatenated[k][i : i + self.max_len] for i in range(0, total_length, self.max_len)]
            for k in concatenated.keys()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def return_arguments(self, args: str | dict) -> dict:
        """Resolves configuration arguments from either a dict or a YAML file path.

        Allows training and model arguments to be specified inline as Python
        dictionaries or externalized to YAML config files for reproducibility.
        If a file path is provided, the file is parsed with `yaml.safe_load`.

        Args:
            args (str | dict): Either a dictionary of arguments to use directly,
                or a string path to a YAML file containing the arguments.

        Returns:
            dict: The resolved argument dictionary.

        Raises:
            ValueError: If ``args`` is a string but cannot be opened or parsed
                as a valid YAML file.
        """
        if isinstance(args, dict):
            return args
        else:
            try:
                with open(args, "r") as f:
                    return yaml.safe_load(f)
            except Exception:
                raise ValueError("Argument should be a dict or path to a YAML file.")
