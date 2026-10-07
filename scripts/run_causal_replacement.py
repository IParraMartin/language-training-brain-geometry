"""Transplants the embeddings or one transformer block from a donor model into a receiver.

Saves the patched model with the receiver's tokenizer under --out_dir (results/models/patched_model/
by default).

Example: brain-patch --patching_mode embeddings --donor_model <donor_id> --receiver_model <receiver_id>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from brain_code.experiments.causal_replacement import (
    patch_embeddings,
    patch_block,
    save_patched
)


def parse_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    """Parses the command-line arguments.

    Returns:
        The parser, kept for error reporting, and the parsed arguments.
    """
    p = argparse.ArgumentParser(
        description="Patch a target model's embeddings or layers."
    )
    p.add_argument(
        "--patching_mode",
        type=str,
        choices=["embeddings", "block"],
        required=True,
        help="Which component to transplant from donor into receiver."
    )
    p.add_argument(
        "--donor_model",
        type=str,
        required=True,
        help="HuggingFace model ID or path to copy weights FROM."
    )
    p.add_argument(
        "--receiver_model",
        type=str,
        required=True,
        help="HuggingFace model ID or path to copy weights INTO."
    )
    p.add_argument(
        "--include_position_embeddings",
        action="store_true",
        help="In 'embeddings' mode, also transplant the positional embeddings (wpe)."
    )
    p.add_argument(
        "--patch_head",
        action="store_true",
        help=(
            "In 'embeddings' mode, also sync the LM head via tie_weights() after patching.  "
            "When set, the condition modifies BOTH the input embedding matrix and the output "
            "head.  Label results as 'input-embedding-plus-output-head', not 'embedding-only'.  "
            "Default: False (input embeddings only)."
        )
    )
    p.add_argument(
        "--layer",
        type=int,
        required=False,
        help="In 'block' mode, the zero-based transformer block index to patch."
    )
    p.add_argument(
        "--out_dir",
        type=str,
        required=False,
        help="Output directory for the patched model (default: results/models/patched_model/)."
    )
    return p, p.parse_args()


def main() -> None:
    """Patches the receiver model and saves the result."""
    parser, args = parse_args()

    print(f"Patching {args.patching_mode}:")
    print(f"\t -> Donor: {args.donor_model}")
    print(f"\t -> Receiver: {args.receiver_model}")

    if args.patching_mode == "embeddings":
        model = patch_embeddings(
            donor_model=args.donor_model,
            receiver_model=args.receiver_model,
            include_position_embeddings=args.include_position_embeddings,
            sync_lm_head=args.patch_head
        )
    else:
        if args.layer is None:
            parser.error("--layer is required when --patching_mode is 'block'.")
        model = patch_block(
            donor_model=args.donor_model,
            receiver_model=args.receiver_model,
            layer=args.layer
        )

    save_patched(
        model=model,
        tokenizer_source=args.receiver_model,
        out_dir=Path(args.out_dir) if args.out_dir else None
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
