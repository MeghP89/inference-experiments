"""Greedy decode speed of HuggingFace transformers on CPU, as a baseline.

    python hf_bench.py "The capital of France is" 128 2   -> prompt, tokens, threads
"""

import resource
import sys
import time

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

prompt = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
n_new = int(sys.argv[2]) if len(sys.argv) > 2 else 128
torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 2)

start = time.perf_counter()
tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
load_s = time.perf_counter() - start

ids = tok(prompt, return_tensors="pt").input_ids
with torch.inference_mode():
    model.generate(ids, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
    start = time.perf_counter()
    out = model.generate(ids, max_new_tokens=n_new, min_new_tokens=n_new,
                         do_sample=False, pad_token_id=tok.eos_token_id)
    secs = time.perf_counter() - start

print(tok.decode(out[0]))
rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
print(f"load {load_s:.2f}s, {n_new} tokens in {secs:.2f}s ({n_new / secs:.1f} tok/s), "
      f"peak rss {rss_mb} MB", file=sys.stderr)
