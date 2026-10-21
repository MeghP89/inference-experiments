# gpt2-inference

GPT-2 inference in C++ from scratch. No BLAS, no ONNX, no libtorch: the model
is a memory-mapped file of floats and the forward pass is a few hundred lines
of loops.

Python is used only to convert the HuggingFace checkpoint into a flat binary.
Nothing in the runtime depends on it.

## Layout

```
python/
  loader.py      gpt2.safetensors  -> gpt2.bin
  tokenizer.py   vocab.json+merges -> tokenizer.bin
  encode.py      reference tokenizer, for cross-checking
cpp/
  gpt2.h         mmap loader, weight pointers
  tokenizer.h    byte-level BPE
  utils.h        layernorm, matmul, gelu, softmax, attention
  forward.h      the forward pass
  main.cpp       greedy generation
```

## Build

Export the weights once:

```sh
cd python
python loader.py       # gpt2.bin, 498 MB
python loader.py --fp16  # gpt2-f16.bin, 251 MB, optional
python tokenizer.py    # tokenizer.bin, 1.1 MB
```

Then build and run:

```sh
cd cpp
cmake -B build -S .
cmake --build build -j
./build/gpt2 "The capital of France is" 20
```

Run it from `cpp/`, or pass the two paths as extra arguments.

## File format

`gpt2.bin` is a 256-byte header (magic, version, dtype, config, `ln_eps`)
followed by every tensor in a fixed order, each aligned to 64 bytes and stored
layer-major. The reader mmaps the file and hands out pointers into it, so
loading is a few syscalls and no copying.

Two things happen at export time so the runtime does not have to:

- Weights are transposed from HuggingFace's `[in, out]` to `[out, in]`, which
  makes each output element a dot product of two contiguous rows.
- The per-layer `attn.bias` causal masks are dropped. They are 50 MB of
  constants that the attention loop bounds reproduce for free.

The tensor order is the contract — nothing on disk names the tensors, and
`Cursor` in `cpp/gpt2.h` walks the same sequence `plan()` in `loader.py` wrote.
A sidecar `gpt2.bin.json` records names and offsets for debugging.

## Correctness

Checked against HuggingFace at every stage. Logits agree to `1.1e-04` absolute
over the 50257-way vocabulary, and 64 steps of greedy generation match token
for token:

```
The capital of France is the capital of the French Republic, and the capital
of the French Republic is the capital of the French
```

The tokenizer is exact on ASCII and on ordinary prose with accents or CJK. It
diverges from the reference on Unicode whitespace and non-ASCII punctuation,
which the hand-written pre-tokenizer treats as letters; `encode.py` is the
escape hatch for input like that.

## Performance

GPT-2 small, greedy decoding, on a 12-thread Intel Core 7 150U laptop:

| Change | Decode tok/s |
|---|---|
| Scalar, single-threaded, no KV cache | 2.95 |
| KV cache | 33.6 |
| AVX2 matmul, 4 outputs per pass | 40.9 |
| Thread pool, one worker per physical core | 60-73 |
| fp16 weights (`loader.py --fp16`), F16C convert in the matmul | 100-107 |

Decoding streams every weight once per token, so it is bound by memory
bandwidth rather than compute. Storing matmul weights as fp16 halves the bytes
read per token; activations and accumulation stay fp32. Greedy output still
matches HuggingFace token for token over 64 steps, with step-0 logits within
`3.7e-02`. Set `GPT2_THREADS` to override the worker count.

### Against HuggingFace

Same prompt, 128 greedy tokens, same laptop, interleaved runs.
`python/hf_bench.py` is the baseline: transformers with torch 2.4 CPU at its
best thread count (2).

| | tok/s | peak RSS | load |
|---|---|---|---|
| transformers (fp32) | 35 | 1,236 MB | 0.40 s |
| this, fp32 | 58-65 | 493 MB | 5 ms |
| this, fp16 | 102-107 | 259 MB | 5 ms |

```sh
python python/hf_bench.py "The capital of France is" 128 2
./cpp/build/gpt2 "The capital of France is" 128 python/gpt2-f16.bin python/tokenizer.bin
```
