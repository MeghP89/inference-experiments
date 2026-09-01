// Loader for gpt2.bin, produced by python/loader.py.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace gpt2 {

constexpr size_t kHeaderBytes = 256;
constexpr size_t kAlign = 64;

#pragma pack(push, 1)
struct Header {
  char magic[8];  // "GPT2CPP\0"
  int32_t version;
  int32_t dtype;  // 0 = f32
  int32_t n_layer;
  int32_t n_head;
  int32_t d_model;
  int32_t ctx;
  int32_t vocab;
  float ln_eps;
};
#pragma pack(pop)
static_assert(sizeof(Header) == 40, "header field packing changed");

// Weights for one transformer block, pointing into the mapping.
// fc_* is mlp.c_fc and fc_proj_* is mlp.c_proj, renamed so the two c_proj
// weights in a block do not collide.
struct Layer {
  const float* ln_1_w;      // [d]
  const float* ln_1_b;      // [d]
  const float* c_attn_w;    // [3d, d]
  const float* c_attn_b;    // [3d]
  const float* c_proj_w;    // [d, d]
  const float* c_proj_b;    // [d]
  const float* ln_2_w;      // [d]
  const float* ln_2_b;      // [d]
  const float* fc_w;        // [4d, d]
  const float* fc_b;        // [4d]
  const float* fc_proj_w;   // [d, 4d]
  const float* fc_proj_b;   // [d]
};

struct Model {
  Header cfg;
  const float* wte;   // [vocab, d], also the tied output projection
  const float* wpe;   // [ctx, d]
  std::vector<Layer> layers;
  const float* ln_f_w;  // [d]
  const float* ln_f_b;  // [d]

  int d_model() const { return cfg.d_model; }
  int n_head() const { return cfg.n_head; }
  int head_dim() const { return cfg.d_model / cfg.n_head; }

  void* map = nullptr;
  size_t map_bytes = 0;
  void close() {
    if (map) munmap(map, map_bytes);
    map = nullptr;
  }
};

// Hands out 64-byte-aligned pointers by walking the tensor order the exporter
// wrote. The order is the format contract; nothing on disk names the tensors.
class Cursor {
 public:
  Cursor(const uint8_t* base, size_t off, size_t limit)
      : base_(base), off_(off), limit_(limit) {}

  const float* take(size_t n) {
    off_ = (off_ + kAlign - 1) & ~(kAlign - 1);
    const float* p = reinterpret_cast<const float*>(base_ + off_);
    off_ += n * sizeof(float);
    ok_ &= off_ <= limit_;
    return p;
  }
  bool ok() const { return ok_; }
  size_t offset() const { return off_; }

 private:
  const uint8_t* base_;
  size_t off_;
  size_t limit_;
  bool ok_ = true;
};

// Maps path into m. Returns false and fills err on any mismatch; on success
// the caller owns the mapping and must call m.close().
inline bool load(const char* path, Model& m, std::string& err) {
  int fd = ::open(path, O_RDONLY);
  if (fd < 0) {
    err = std::string("open failed: ") + path;
    return false;
  }
  struct stat st {};
  if (fstat(fd, &st) != 0 || (size_t)st.st_size < kHeaderBytes) {
    ::close(fd);
    err = "file too small";
    return false;
  }
  void* map = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (map == MAP_FAILED) {
    err = "mmap failed";
    return false;
  }

  const uint8_t* base = static_cast<const uint8_t*>(map);
  std::memcpy(&m.cfg, base, sizeof(Header));
  if (std::memcmp(m.cfg.magic, "GPT2CPP\0", 8) != 0) {
    munmap(map, st.st_size);
    err = "bad magic";
    return false;
  }
  if (m.cfg.version != 1 || m.cfg.dtype != 0) {
    munmap(map, st.st_size);
    err = "unsupported version/dtype";
    return false;
  }

  m.map = map;
  m.map_bytes = st.st_size;

  const int d = m.cfg.d_model;
  Cursor c(base, kHeaderBytes, st.st_size);
  m.wte = c.take((size_t)m.cfg.vocab * d);
  m.wpe = c.take((size_t)m.cfg.ctx * d);
  m.layers.resize(m.cfg.n_layer);
  for (Layer& l : m.layers) {
    l.ln_1_w = c.take(d);
    l.ln_1_b = c.take(d);
    l.c_attn_w = c.take((size_t)3 * d * d);
    l.c_attn_b = c.take(3 * d);
    l.c_proj_w = c.take((size_t)d * d);
    l.c_proj_b = c.take(d);
    l.ln_2_w = c.take(d);
    l.ln_2_b = c.take(d);
    l.fc_w = c.take((size_t)4 * d * d);
    l.fc_b = c.take(4 * d);
    l.fc_proj_w = c.take((size_t)d * 4 * d);
    l.fc_proj_b = c.take(d);
  }
  m.ln_f_w = c.take(d);
  m.ln_f_b = c.take(d);

  if (!c.ok()) {
    m.close();
    err = "file truncated: layout ran past end";
    return false;
  }
  return true;
}

}  // namespace gpt2
