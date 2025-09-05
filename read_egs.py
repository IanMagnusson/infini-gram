#!/usr/bin/env python3
"""
Read an infini-gram tokenized.{s} file (default shard 0), interpret it as GPT-2 token IDs,
remove document separator sentinels, and decode back to text with the Hugging Face GPT-2 tokenizer.

Assumptions:
    - You built the index with --tokenizer gpt2 or dolma2 (so token IDs match the tokenizer).
    - Default token dtype is uint32 (u32) for dolma2. (Change --token-dtype if different.)
    - Document separator token is the all-0xFF value for that dtype (255 for u8, 65535 for u16, 4294967295 for u32).
Usage:
    python read_decode_gpt2.py --path tokenized.0 --max-docs 3
"""

import argparse
import numpy as np
from transformers import GPT2TokenizerFast

def detect_dtype(path, explicit):
        if explicit is not None:
                return {'u8': np.uint8, 'u16': np.uint16, 'u32': np.uint32}[explicit]
        # Default to u32 for dolma2-tokenizer
        return np.uint32

def get_separator(dtype):
        # All-bytes-FF sentinel
        return np.array([(1 << (8 * np.dtype(dtype).itemsize)) - 1], dtype=dtype)[0]

def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--path", default="tokenized.0", help="Path to tokenized.{s} file")
        ap.add_argument("--token-dtype", choices=["u8","u16","u32"], default=None,
                                        help="Override token dtype if known (otherwise assume u32 for dolma2).")
        ap.add_argument("--max-docs", type=int, default=2,
                                        help="Number of documents to decode (splits on sentinel).")
        ap.add_argument("--join-docs", action="store_true",
                                        help="If set, prints one big decoded string instead of per-document.")
        args = ap.parse_args()

        dtype = detect_dtype(args.path, args.token_dtype)
        raw = np.memmap(args.path, mode="r", dtype=np.uint8)
        tokens = raw.view(dtype)
        sep = get_separator(dtype)

        sep_positions = np.where(tokens == sep)[0]
        if len(sep_positions) == 0:
                docs = [tokens]
        else:
                docs = []
                for i, pos in enumerate(sep_positions):
                        start = pos + 1
                        end = sep_positions[i + 1] if i + 1 < len(sep_positions) else len(tokens)
                        doc_tokens = tokens[start:end]
                        if doc_tokens.size > 0:
                                docs.append(doc_tokens)
                if not docs:
                        docs = [tokens[tokens != sep]]

        tokenizer = GPT2TokenizerFast.from_pretrained("allenai/dolma2-tokenizer")

        if args.join_docs:
                concat = np.concatenate(docs) if len(docs) > 1 else docs[0]
                valid = concat[concat < tokenizer.vocab_size]
                text = tokenizer.decode(valid.tolist())
                print(text)
        else:
                for i, doc in enumerate(docs[:args.max_docs]):
                        valid = doc[doc < tokenizer.vocab_size]
                        text = tokenizer.decode(valid.tolist())
                        print(f"========== Document {i} (tokens={len(doc)}) ==========")
                        print(text)
                        print()

if __name__ == "__main__":
        main()
