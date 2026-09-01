// Tensor primitives. Activations are [T, n] row-major; weights are [out, in],
// so one output is a dot product of two contiguous rows. Nothing allocates.
#pragma once

#include <cmath>

namespace gpt2 {

// Normalizes each of the T rows over its d features, then scales by w and
// shifts by b. Statistics are per row; w and b are [d]. out must not alias in.
inline void layernorm(float* out, const float* in, const float* w,
                      const float* b, int T, int d, float eps) {
  for (int t = 0; t < T; ++t) {
    const float* row = in + t * d;
    float* dst = out + t * d;

    float sum = 0.0f;
    for (int j = 0; j < d; ++j) sum += row[j];
    float mean = sum / d;

    float sq = 0.0f;
    for (int j = 0; j < d; ++j) {
      float dev = row[j] - mean;
      sq += dev * dev;
    }
    // Population variance, to match PyTorch.
    float inv = 1.0f / std::sqrt(sq / d + eps);

    for (int j = 0; j < d; ++j) dst[j] = (row[j] - mean) * inv * w[j] + b[j];
  }
}

// out = in @ w^T + b, with in [T, n_in], w [n_out, n_in], out [T, n_out].
// b is [n_out], or null for a layer without bias.
inline void matmul(float* out, const float* in, const float* w, const float* b,
                   int T, int n_in, int n_out) {
  for (int t = 0; t < T; ++t) {
    const float* in_row = in + t * n_in;
    for (int o = 0; o < n_out; ++o) {
      const float* w_row = w + o * n_in;
      float sum = 0.0f;
      for (int j = 0; j < n_in; ++j) sum += in_row[j] * w_row[j];
      out[t * n_out + o] = b ? sum + b[o] : sum;
    }
  }
}

// GELU in place. GPT-2 was trained with the tanh approximation rather than the
// erf form; the two differ enough to show up when diffing against PyTorch.
inline void gelu(float* x, int n) {
  constexpr float kSqrt2OverPi = 0.7978845608028654f;
  for (int i = 0; i < n; ++i) {
    float v = x[i];
    x[i] = 0.5f * v * (1.0f + std::tanh(kSqrt2OverPi * (v + 0.044715f * v * v * v)));
  }
}

// Softmax over x[0..n) in place. Shifts by the max first so exp() cannot
// overflow; the shift cancels out in the ratio.
inline void softmax(float* x, int n) {
  float max = x[0];
  for (int i = 1; i < n; ++i) if (x[i] > max) max = x[i];
  float sum = 0.0f;
  for (int i = 0; i < n; ++i) { x[i] = std::exp(x[i] - max); sum += x[i]; }
  float inv = 1.0f / sum;
  for (int i = 0; i < n; ++i) x[i] *= inv;
}

// Multi-head causal self-attention. qkv is [T, 3d], each row laid out as
// Q(d) | K(d) | V(d) with n_head slices of d/n_head inside each third.
// out is [T, d]; head h writes to out + i*d + h*hd, which is the concatenation.
// scores is caller-owned scratch of length T.
inline void attention(float* out, const float* qkv, float* scores, int T, int d,
                      int n_head) {
  const int hd = d / n_head;
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

  for (int h = 0; h < n_head; ++h) {
    for (int i = 0; i < T; ++i) {
      const float* q = qkv + i * 3 * d + 0 * d + h * hd;

      // Bounding j at i is the causal mask, and cheaper than -inf entries.
      for (int j = 0; j <= i; ++j) {
        const float* k = qkv + j * 3 * d + 1 * d + h * hd;
        float s = 0.0f;
        for (int e = 0; e < hd; ++e) s += q[e] * k[e];
        scores[j] = s * scale;
      }
      softmax(scores, i + 1);

      float* dst = out + i * d + h * hd;
      for (int e = 0; e < hd; ++e) dst[e] = 0.0f;
      for (int j = 0; j <= i; ++j) {
        const float* v = qkv + j * 3 * d + 2 * d + h * hd;
        float wgt = scores[j];
        for (int e = 0; e < hd; ++e) dst[e] += wgt * v[e];
      }
    }
  }
}

}  // namespace gpt2
