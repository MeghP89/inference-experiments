#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "forward.h"
#include "gpt2.h"
#include "tokenizer.h"

int main(int argc, char** argv) {
  const char* prompt = argc > 1 ? argv[1] : "The capital of France is";
  const int n_new = argc > 2 ? std::atoi(argv[2]) : 20;

  gpt2::Model m;
  gpt2::Tokenizer tok;
  std::string err;
  if (!gpt2::load("../python/gpt2.bin", m, err) ||
      !tok.load("../python/tokenizer.bin", err)) {
    std::fprintf(stderr, "load: %s\n", err.c_str());
    return 1;
  }

  std::vector<int> ids = tok.encode(prompt);
  if (ids.empty()) { std::fprintf(stderr, "empty prompt\n"); return 1; }

  gpt2::Activations a(m, ids.size() + n_new);
  std::printf("%s", prompt);
  std::fflush(stdout);

  auto start = std::chrono::steady_clock::now();
  int generated = 0;
  for (int i = 0; i < n_new; ++i) {
    if (static_cast<int>(ids.size()) >= m.cfg.ctx) break;
    gpt2::forward(m, a, ids);
    int next = gpt2::argmax(a.logits);
    ids.push_back(next);
    std::printf("%s", tok.decode({next}).c_str());
    std::fflush(stdout);
    generated++;
  }
  std::printf("\n");

  double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::fprintf(stderr, "%d tokens in %.2fs (%.1f tok/s)\n", generated, secs, generated / secs);

  m.close();
  return 0;
}
