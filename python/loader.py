"""Export GPT-2 safetensors to a flat binary the C++ runtime can mmap.

The file is a 256-byte header followed by tensors in a fixed order, each
64-byte aligned. See cpp/gpt2.h for the reading side.
"""

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from dataclasses import dataclass, field
import numpy as np
import json
import mmap

ALIGN = 64

path = hf_hub_download("gpt2", "model.safetensors")
cfg_path = hf_hub_download("gpt2", "config.json")
cfg = json.load(open(cfg_path))
raw = load_file(path)


@dataclass
class ModelConfig:
    n_layer: int
    n_head: int
    d_model: int
    vocab: int
    ctx: int
    ln_eps: float

    @classmethod
    def from_hf(cls, cfg, raw):
        return cls(
            n_layer=cfg["n_layer"],
            n_head=cfg["n_head"],
            d_model=cfg["n_embd"],
            vocab=raw["wte.weight"].shape[0],
            ctx=cfg["n_ctx"],
            ln_eps=cfg["layer_norm_epsilon"],
        )


@dataclass
class Tensor:
    name: str
    shape: tuple
    offset: int = 0

    @property
    def nbytes(self):
        return int(np.prod(self.shape)) * 4

    def view(self, mm):
        """Zero-copy fp32 view into an mmap of the exported file."""
        buf = np.frombuffer(mm, np.float32, int(np.prod(self.shape)), self.offset)
        return buf.reshape(self.shape)


@dataclass
class Layout:
    config: ModelConfig
    tensors: list = field(default_factory=list)
    header_bytes: int = 256

    def by_name(self):
        return {t.name: t for t in self.tensors}

    @property
    def nbytes(self):
        last = self.tensors[-1]
        return last.offset + last.nbytes

    def manifest(self):
        return {
            "config": self.config.__dict__,
            "align": ALIGN,
            "tensors": [
                {"name": t.name, "shape": list(t.shape), "offset": t.offset}
                for t in self.tensors
            ],
        }


def plan(config, raw):
    """Assign each tensor a 64-byte-aligned offset, in layer-major order.

    Matmul weights are transposed to [out, in] so each output element is one
    contiguous dot product. The per-layer attn.bias causal masks are left out
    entirely; the runtime rebuilds the mask from the loop bounds.
    """
    order = ["wte.weight", "wpe.weight"]
    transposed = set()
    for i in range(config.n_layer):
        p = f"h.{i}."
        order += [
            p + "ln_1.weight", p + "ln_1.bias",
            p + "attn.c_attn.weight", p + "attn.c_attn.bias",
            p + "attn.c_proj.weight", p + "attn.c_proj.bias",
            p + "ln_2.weight", p + "ln_2.bias",
            p + "mlp.c_fc.weight", p + "mlp.c_fc.bias",
            p + "mlp.c_proj.weight", p + "mlp.c_proj.bias",
        ]
        transposed |= {
            p + "attn.c_attn.weight", p + "attn.c_proj.weight",
            p + "mlp.c_fc.weight", p + "mlp.c_proj.weight",
        }
    order += ["ln_f.weight", "ln_f.bias"]

    layout = Layout(config=config)
    off = layout.header_bytes
    for name in order:
        shape = tuple(raw[name].shape)
        if name in transposed:
            shape = shape[::-1]
        off += -off % ALIGN
        layout.tensors.append(Tensor(name, shape, off))
        off += layout.tensors[-1].nbytes
    return layout, transposed


def write(layout, transposed, raw, out_path="gpt2.bin"):
    c = layout.config
    header = bytearray(layout.header_bytes)
    header[:8] = b"GPT2CPP\0"
    header[8:36] = np.array(
        [1, 0, c.n_layer, c.n_head, c.d_model, c.ctx, c.vocab], np.int32
    ).tobytes()
    header[36:40] = np.float32(c.ln_eps).tobytes()

    with open(out_path, "wb") as f:
        f.write(header)
        for t in layout.tensors:
            f.write(b"\0" * (t.offset - f.tell()))
            w = raw[t.name]
            w = w.t() if t.name in transposed else w
            f.write(w.contiguous().numpy().tobytes())

    with open(out_path + ".json", "w") as f:
        json.dump(layout.manifest(), f, indent=2)


def load(path="gpt2.bin", manifest_path=None):
    """mmap the exported file and return (config, {name: array}, mapping)."""
    m = json.load(open(manifest_path or path + ".json"))
    config = ModelConfig(**m["config"])
    tensors = [Tensor(t["name"], tuple(t["shape"]), t["offset"]) for t in m["tensors"]]
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    return config, {t.name: t.view(mm) for t in tensors}, mm


if __name__ == "__main__":
    config = ModelConfig.from_hf(cfg, raw)
    layout, transposed = plan(config, raw)
    write(layout, transposed, raw)
    print(config)
    print(f"gpt2.bin: {layout.nbytes / 1e6:.1f} MB, {len(layout.tensors)} tensors")

    _, w, _ = load()
    assert np.allclose(w["wpe.weight"], raw["wpe.weight"].numpy())
    assert np.allclose(w["h.0.mlp.c_fc.weight"], raw["h.0.mlp.c_fc.weight"].t().numpy())
    print("roundtrip ok:", w["h.0.mlp.c_fc.weight"].shape, w["wte.weight"].shape)
