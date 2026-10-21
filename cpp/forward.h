#pragma once

#include <vector>

#include "gpt2.h"
#include "utils.h"

namespace gpt2 {

// Scratch buffers for one forward pass, sized for at most max_T tokens.
// Allocated once and reused across layers and generation steps.
struct Activations {
  std::vector<float> x;       // [max_T, d] residual stream
  std::vector<float> xb;      // [max_T, d] normalized input / branch output
  std::vector<float> xb2;     // [max_T, d] attention heads before projection
  std::vector<float> qkv;     // [max_T, 3d]
  std::vector<float> h;       // [max_T, 4d] MLP hidden
  std::vector<float> scores;  // [max_T]
  std::vector<float> logits;  // [vocab]

  Activations(const Model& m, int max_T)
      : x(max_T * m.cfg.d_model),
        xb(max_T * m.cfg.d_model),
        xb2(max_T * m.cfg.d_model),
        qkv(max_T * 3 * m.cfg.d_model),
        h(max_T * 4 * m.cfg.d_model),
        scores(max_T),
        logits(m.cfg.vocab) {}
};

// Runs the model over ids and leaves the final position's logits in a.logits.
inline void forward(const Model& m, Activations& a, const std::vector<int>& ids) {
  const int d = m.cfg.d_model;
  const int T = static_cast<int>(ids.size());
  const float eps = m.cfg.ln_eps;

  for (int t = 0; t < T; ++t)
    for (int j = 0; j < d; ++j)
      a.x[t * d + j] = m.wte[ids[t] * d + j] + m.wpe[t * d + j];

  for (const Layer& l : m.layers) {
    layernorm(a.xb.data(), a.x.data(), l.ln_1_w, l.ln_1_b, T, d, eps);
    matmul(a.qkv.data(), a.xb.data(), l.c_attn_w, l.c_attn_b, T, d, 3 * d);
    attention(a.xb2.data(), a.qkv.data(), a.scores.data(), T, d, m.cfg.n_head);
    matmul(a.xb.data(), a.xb2.data(), l.c_proj_w, l.c_proj_b, T, d, d);
    for (int i = 0; i < T * d; ++i) a.x[i] += a.xb[i];

    layernorm(a.xb.data(), a.x.data(), l.ln_2_w, l.ln_2_b, T, d, eps);
    matmul(a.h.data(), a.xb.data(), l.fc_w, l.fc_b, T, d, 4 * d);
    gelu(a.h.data(), T * 4 * d);
    matmul(a.xb.data(), a.h.data(), l.fc_proj_w, l.fc_proj_b, T, 4 * d, d);
    for (int i = 0; i < T * d; ++i) a.x[i] += a.xb[i];
  }

  layernorm(a.xb.data(), a.x.data(), m.ln_f_w, m.ln_f_b, T, d, eps);
  // The output projection is the tied token embedding, and carries no bias.
  matmul(a.logits.data(), a.xb.data() + (T - 1) * d, m.wte, nullptr, 1, d,
         m.cfg.vocab);
}

inline int argmax(const std::vector<float>& v) {
  int best = 0;
  for (size_t i = 1; i < v.size(); ++i)
    if (v[i] > v[best]) best = static_cast<int>(i);
  return best;
}

}  // namespace gpt2
