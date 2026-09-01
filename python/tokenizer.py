"""Export GPT-2's byte-level BPE to a flat binary the C++ runtime can mmap.

vocab.json stores tokens as printable unicode (space -> 'Ġ') to stay text-safe;
that mapping is undone here so C++ only ever sees raw bytes. Merges are
resolved to ids at export time, turning the C++ inner loop into a lookup on a
pair of u32s rather than a string hash.

Layout:
  header       magic "GPT2TOK\0", version, n_vocab, n_merges, eot_id  (32 B)
  byte_to_id   u32[256]              single-byte token for each byte value
  merges       u32[n_merges][3]      {left, right, result}, index = rank
  offsets      u32[n_vocab + 1]      into blob
  blob         raw token bytes       for decoding
"""

from dataclasses import dataclass
from huggingface_hub import hf_hub_download
import numpy as np
import json
import mmap

MAGIC = b"GPT2TOK\0"
VERSION = 1
HEADER_BYTES = 32


def bytes_to_unicode():
    """GPT-2's byte to printable-unicode mapping, from openai/gpt-2 encoder.py."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


@dataclass
class Tokenizer:
    vocab: list          # id -> raw bytes
    merges: list         # rank -> (left_id, right_id, result_id)
    byte_to_id: list     # byte value -> id
    eot_id: int

    @classmethod
    def from_hf(cls, model="gpt2"):
        raw_vocab = json.load(open(hf_hub_download(model, "vocab.json")))
        merge_lines = open(hf_hub_download(model, "merges.txt")).read().split("\n")

        decoder = {c: b for b, c in bytes_to_unicode().items()}

        def to_bytes(tok):
            # Specials like <|endoftext|> aren't in the byte alphabet; keep utf-8.
            if all(c in decoder for c in tok):
                return bytes(decoder[c] for c in tok)
            return tok.encode("utf-8")

        vocab = [b""] * len(raw_vocab)
        for tok, i in raw_vocab.items():
            vocab[i] = to_bytes(tok)

        merges = []
        for line in merge_lines[1:]:  # line 0 is "#version: 0.2"
            if not line:
                continue
            a, b = line.split(" ")
            merges.append((raw_vocab[a], raw_vocab[b], raw_vocab[a + b]))

        byte_to_id = [raw_vocab[bytes_to_unicode()[b]] for b in range(256)]
        return cls(vocab, merges, byte_to_id, raw_vocab["<|endoftext|>"])

    def write(self, out_path="tokenizer.bin"):
        blob = b"".join(self.vocab)
        offsets = np.zeros(len(self.vocab) + 1, np.uint32)
        np.cumsum([len(v) for v in self.vocab], out=offsets[1:])

        header = bytearray(HEADER_BYTES)
        header[:8] = MAGIC
        header[8:24] = np.array(
            [VERSION, len(self.vocab), len(self.merges), self.eot_id], np.uint32
        ).tobytes()

        with open(out_path, "wb") as f:
            f.write(header)
            f.write(np.array(self.byte_to_id, np.uint32).tobytes())
            f.write(np.array(self.merges, np.uint32).tobytes())
            f.write(offsets.tobytes())
            f.write(blob)
        return out_path


if __name__ == "__main__":
    tok = Tokenizer.from_hf()
    path = tok.write()
    size = len(open(path, "rb").read())
    print(f"{path}: {size / 1e6:.1f} MB")
    print(f"  vocab={len(tok.vocab)} merges={len(tok.merges)} eot={tok.eot_id}")
    print(f"  vocab[31373]={tok.vocab[31373]!r} vocab[995]={tok.vocab[995]!r}")
