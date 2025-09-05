import os, json
import numpy as np

# NEW: Select tokenizer and set token width
TOKENIZER_NAME = "allenai/dolma2-tokenizer"  # or "gpt2"
if TOKENIZER_NAME == "allenai/dolma2-tokenizer":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")
    TOKEN_DTYPE = np.uint32
    TOKEN_WIDTH = 4
else:
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    TOKEN_DTYPE = np.uint16
    TOKEN_WIDTH = 2

SHARD = 0
PREFIX = "/mnt/raid0/suffix-array-dolma2/"
TOKEN_FILE = f"{PREFIX}tokenized.{SHARD}"
SA_FILE = f"{PREFIX}table.{SHARD}"
OFF_FILE = f"{PREFIX}offset.{SHARD}"
META_FILE = f"{PREFIX}metadata.{SHARD}"
METAOFF_FILE = f"{PREFIX}metaoff.{SHARD}"

def load_token_array():
    token_size = TOKEN_WIDTH
    raw = np.memmap(TOKEN_FILE, dtype=np.uint8, mode='r')
    if raw.size % token_size != 0:
        raise ValueError(f"Token file size {raw.size} not divisible by token size {token_size}")
    tokens = raw.view(TOKEN_DTYPE)
    return raw, tokens, token_size

def load_suffix_array(t_count, token_size):
    """Load suffix array pointers.

    The Rust index stores each suffix pointer in a fixed number of bytes
    ("ratio" computed during indexing: ceil(log2(ds_size)/8)). This is often < 8,
    so the table file length is t_count * ratio. We reconstruct the integer
    offsets and auto-detect endianness by validating token alignment.
    """
    table_bytes = np.memmap(SA_FILE, dtype=np.uint8, mode='r')
    total_bytes = table_bytes.shape[0]
    if t_count == 0:
        raise ValueError("Token count is zero; empty index?")
    if total_bytes % t_count != 0:
        raise ValueError(
            f"Suffix table size {total_bytes} not divisible by token count {t_count}. "
            "Corrupt or mismatched files." )
    k = total_bytes // t_count  # bytes per pointer
    view = table_bytes.reshape(t_count, k).astype(np.uint64)

    # Little-endian reconstruction
    le_factors = np.power(256, np.arange(k, dtype=np.uint64), dtype=np.uint64)
    ptrs_le = (view * le_factors).sum(axis=1, dtype=np.uint64)
    aligned_le = np.all(ptrs_le % token_size == 0)

    if aligned_le:
        return ptrs_le

    # Big-endian reconstruction fallback
    be_factors = le_factors[::-1]
    ptrs_be = (view * be_factors).sum(axis=1, dtype=np.uint64)
    aligned_be = np.all(ptrs_be % token_size == 0)

    if aligned_be:
        return ptrs_be

    # Neither alignment worked fully; report diagnostics (sample first 10 remainders)
    rem_le = (ptrs_le[:10] % token_size).tolist()
    rem_be = (ptrs_be[:10] % token_size).tolist()
    raise ValueError(
        "Could not determine suffix array endianness: neither little nor big endian rebuilds "
        f"yield fully token-aligned pointers (token size {token_size}). Sample remainders le={rem_le} be={rem_be}." )

def load_doc_offsets():
    if not os.path.exists(OFF_FILE):
        return None
    return np.memmap(OFF_FILE, dtype=np.uint64, mode='r')

def load_metadata_structs():
    if not (os.path.exists(META_FILE) and os.path.exists(METAOFF_FILE)):
        return None, None
    metaoffs = np.memmap(METAOFF_FILE, dtype=np.uint64, mode='r')
    return metaoffs, META_FILE

def get_doc_index(byte_off, doc_offsets):
    i = np.searchsorted(doc_offsets, byte_off, side='right') - 1
    return int(i)

def read_metadata(doc_ix, metaoffs, metafile):
    with open(metafile, 'rb') as f:
        f.seek(int(metaoffs[doc_ix]))
        line = f.readline().decode('utf-8')
    return json.loads(line)

def decode_tokens(token_list):
    return tokenizer.decode(token_list)

def iter_suffixes(limit=10, max_tokens=10):
    raw_bytes, token_ids, token_size = load_token_array()
    t_count = token_ids.shape[0]
    suffix_ptrs = load_suffix_array(t_count, token_size)
    doc_offsets = load_doc_offsets()
    metaoffs, metafile = load_metadata_structs()

    sentinel_id = (1 << (8 * token_size)) - 1  # e.g. 65535 for u16, 4294967295 for u32

    for rank in range(min(limit, t_count)):
        byte_off = int(suffix_ptrs[rank])
        if byte_off % token_size != 0:
            raise ValueError(f"Suffix byte offset {byte_off} not aligned to token size {token_size}")
        token_start = byte_off // token_size
        end = token_start + max_tokens + 1
        seq = token_ids[token_start:end]
        if seq.size > 0 and int(seq[0]) == sentinel_id:
            seq_no_sep = seq[1:]
            starts_at_sep = True
        else:
            seq_no_sep = seq
            starts_at_sep = False
        if seq_no_sep.size == 0:
            trimmed = seq_no_sep
        else:
            sep_positions = np.where(seq_no_sep == sentinel_id)[0]
            cut = int(sep_positions[0]) if sep_positions.size > 0 else seq_no_sep.size
            trimmed = seq_no_sep[:cut]

        doc_ix = None
        doc_meta = None
        if doc_offsets is not None:
            doc_ix = get_doc_index(byte_off, doc_offsets)
            if metaoffs is not None:
                doc_meta = read_metadata(doc_ix, metaoffs, metafile)

        tokens_int = [int(x) for x in trimmed.tolist()]
        decoded = decode_tokens(tokens_int)

        yield {
            "rank": rank,
            "byte_offset": byte_off,
            "token_start": token_start,
            "starts_at_doc_boundary": starts_at_sep,
            "tokens": tokens_int,
            "decoded": decoded,
            "doc_index": doc_ix,
            "metadata": doc_meta
        }

if __name__ == "__main__":
    for info in iter_suffixes(limit=5, max_tokens=12):
        print(json.dumps(info, ensure_ascii=False))