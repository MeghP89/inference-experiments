// Tensor primitives. Activations are [T, n] row-major; weights are [out, in],
// so one output is a dot product of two contiguous rows. Nothing allocates.
#pragma once

#include <cmath>

#ifdef __AVX2__
#include <immintrin.h>
#endif

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

#ifdef __AVX2__
inline float hsum(__m256 v) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_movehdup_ps(s));
  return _mm_cvtss_f32(s);
}
#endif

inline float dot(const float* a, const float* b, int n) {
  float sum = 0.0f;
  for (int j = 0; j < n; ++j) sum += a[j] * b[j];
  return sum;
}

// Computes 4 outputs per pass so each load of in_row feeds 4 weight rows
inline void matmul(float* out, const float* in, const float* w, const float* b,
                   int T, int n_in, int n_out) {
  for (int t = 0; t < T; ++t) {
    const float* in_row = in + t * n_in;
    float* dst = out + t * n_out;
    int o = 0;
#ifdef __AVX2__
    if (n_in % 8 == 0) {
      for (; o + 4 <= n_out; o += 4) {
        const float* w0 = w + (size_t)o * n_in;
        const float* w1 = w0 + n_in;
        const float* w2 = w1 + n_in;
        const float* w3 = w2 + n_in;
        __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
        for (int j = 0; j < n_in; j += 8) {
          __m256 x = _mm256_loadu_ps(in_row + j);
          s0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(w0 + j), s0);
          s1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(w1 + j), s1);
          s2 = _mm256_fmadd_ps(x, _mm256_loadu_ps(w2 + j), s2);
          s3 = _mm256_fmadd_ps(x, _mm256_loadu_ps(w3 + j), s3);
        }
        dst[o] = hsum(s0) + (b ? b[o] : 0.0f);
        dst[o + 1] = hsum(s1) + (b ? b[o + 1] : 0.0f);
        dst[o + 2] = hsum(s2) + (b ? b[o + 2] : 0.0f);
        dst[o + 3] = hsum(s3) + (b ? b[o + 3] : 0.0f);
      }
    }
#endif
    for (; o < n_out; ++o) {
      float sum = dot(in_row, w + (size_t)o * n_in, n_in);
      dst[o] = b ? sum + b[o] : sum;
    }
  }
}

// GELU in place. GPT-2 was trained with the tanh approximation rather than the
// erf form
inline void gelu(float* x, int n) {
  constexpr float kSqrt2OverPi = 0.7978845608028654f;
  for (int i = 0; i < n; ++i) {
    float v = x[i];
    x[i] = 0.5f * v * (1.0f + std::tanh(kSqrt2OverPi * (v + 0.044715f * v * v * v)));
  }
}

// Softmax over x[0..n) in place. Shifts by the max first so exp() cannot
// overflow
inline void softmax(float* x, int n) {
  float max = x[0];
  for (int i = 1; i < n; ++i) if (x[i] > max) max = x[i];
  float sum = 0.0f;
  for (int i = 0; i < n; ++i) { x[i] = std::exp(x[i] - max); sum += x[i]; }
  float inv = 1.0f / sum;
  for (int i = 0; i < n; ++i) x[i] *= inv;
}

// Multi-head causal self-attention for n new rows starting at position start.
// k_cache and v_cache are [max_T, d] and already hold rows up to start + n
inline void attention(float* out, const float* qkv, const float* k_cache,
                      const float* v_cache, float* scores, int n, int start,
                      int d, int n_head) {
  const int hd = d / n_head;
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

  for (int h = 0; h < n_head; ++h) {
    for (int i = 0; i < n; ++i) {
      const float* q = qkv + i * 3 * d + h * hd;
      const int pos = start + i;

      // Bounding j at pos is the causal mask
      for (int j = 0; j <= pos; ++j) {
        const float* k = k_cache + j * d + h * hd;
        float s = 0.0f;
        for (int e = 0; e < hd; ++e) s += q[e] * k[e];
        scores[j] = s * scale;
      }
      softmax(scores, pos + 1);

      float* dst = out + i * d + h * hd;
      for (int e = 0; e < hd; ++e) dst[e] = 0.0f;
      for (int j = 0; j <= pos; ++j) {
        const float* v = v_cache + j * d + h * hd;
        float wgt = scores[j];
        for (int e = 0; e < hd; ++e) dst[e] += wgt * v[e];
      }
    }
  }
}

}  // namespace gpt2
