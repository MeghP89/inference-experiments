#pragma once

#include <algorithm>
#include <vector>

#include "gpt2.h"
#include "utils.h"

namespace gpt2 {

// Scratch buffers and the KV cache, sized for at most max_T tokens.
// Allocated once and reused across layers and generation steps
struct Activations {
  std::vector<float> x;        // [max_T, d] residual stream
  std::vector<float> xb;       // [max_T, d] normalized input / branch output
  std::vector<float> xb2;      // [max_T, d] attention heads before projection
  std::vector<float> qkv;      // [max_T, 3d]
  std::vector<float> h;        // [max_T, 4d] MLP hidden
  std::vector<float> scores;   // [max_T]
  std::vector<float> logits;   // [vocab]
  std::vector<float> k_cache;  // [n_layer, max_T, d]
  std::vector<float> v_cache;  // [n_layer, max_T, d]
  int max_T;
  int n_past = 0;

  Activations(const Model& m, int max_T)
      : x(max_T * m.cfg.d_model),
        xb(max_T * m.cfg.d_model),
        xb2(max_T * m.cfg.d_model),
        qkv(max_T * 3 * m.cfg.d_model),
        h(max_T * 4 * m.cfg.d_model),
        scores(max_T),
        logits(m.cfg.vocab),
        k_cache((size_t)m.cfg.n_layer * max_T * m.cfg.d_model),
        v_cache((size_t)m.cfg.n_layer * max_T * m.cfg.d_model),
        max_T(max_T) {}
};

// Runs the tokens of ids not yet in the cache and leaves the final position's
// logits in a.logits. A shorter ids than last call starts a new sequence
inline void forward(const Model& m, Activations& a, const std::vector<int>& ids) {
  const int d = m.cfg.d_model;
  const int T = static_cast<int>(ids.size());
  const float eps = m.cfg.ln_eps;

  if (T <= a.n_past) a.n_past = 0;
  const int start = a.n_past;
  const int n = T - start;

  for (int i = 0; i < n; ++i)
    for (int j = 0; j < d; ++j)
      a.x[i * d + j] = m.wte[ids[start + i] * d + j] + m.wpe[(start + i) * d + j];

  for (size_t li = 0; li < m.layers.size(); ++li) {
    const Layer& l = m.layers[li];
    float* k_cache = a.k_cache.data() + li * a.max_T * d;
    float* v_cache = a.v_cache.data() + li * a.max_T * d;

    layernorm(a.xb.data(), a.x.data(), l.ln_1_w, l.ln_1_b, n, d, eps);
    matmul(a.qkv.data(), a.xb.data(), l.c_attn_w, l.c_attn_b, n, d, 3 * d);
    for (int i = 0; i < n; ++i) {
      const float* row = a.qkv.data() + i * 3 * d;
      std::copy(row + d, row + 2 * d, k_cache + (start + i) * d);
      std::copy(row + 2 * d, row + 3 * d, v_cache + (start + i) * d);
    }
    attention(a.xb2.data(), a.qkv.data(), k_cache, v_cache, a.scores.data(), n,
              start, d, m.cfg.n_head);
    matmul(a.xb.data(), a.xb2.data(), l.c_proj_w, l.c_proj_b, n, d, d);
    for (int i = 0; i < n * d; ++i) a.x[i] += a.xb[i];

    layernorm(a.xb.data(), a.x.data(), l.ln_2_w, l.ln_2_b, n, d, eps);
    matmul(a.h.data(), a.xb.data(), l.fc_w, l.fc_b, n, d, 4 * d);
    gelu(a.h.data(), n * 4 * d);
    matmul(a.xb.data(), a.h.data(), l.fc_proj_w, l.fc_proj_b, n, 4 * d, d);
    for (int i = 0; i < n * d; ++i) a.x[i] += a.xb[i];
  }
  a.n_past = T;

  // Only the last position feeds the logits
  const float* last = a.x.data() + (n - 1) * d;
  layernorm(a.xb.data(), last, m.ln_f_w, m.ln_f_b, 1, d, eps);
  // The output projection is the tied token embedding, and carries no bias
  matmul(a.logits.data(), a.xb.data(), m.wte, nullptr, 1, d, m.cfg.vocab);
}

inline int argmax(const std::vector<float>& v) {
  int best = 0;
  for (size_t i = 1; i < v.size(); ++i)
    if (v[i] > v[best]) best = static_cast<int>(i);
  return best;
}

}  // namespace gpt2
