// Byte-level BPE for GPT-2, reading tokenizer.bin from python/tokenizer.py.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace gpt2 {

class Tokenizer {
 public:
  ~Tokenizer() { if (map_) munmap(map_, map_bytes_); }

  bool load(const char* path, std::string& err) {
    int fd = ::open(path, O_RDONLY);
    if (fd < 0) { err = std::string("open failed: ") + path; return false; }
    struct stat st {};
    if (fstat(fd, &st) != 0 || (size_t)st.st_size < 32) {
      ::close(fd); err = "file too small"; return false;
    }
    void* m = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (m == MAP_FAILED) { err = "mmap failed"; return false; }

    const uint8_t* base = static_cast<const uint8_t*>(m);
    if (std::memcmp(base, "GPT2TOK\0", 8) != 0) {
      munmap(m, st.st_size); err = "bad magic"; return false;
    }
    const uint32_t* h = reinterpret_cast<const uint32_t*>(base + 8);
    if (h[0] != 1) { munmap(m, st.st_size); err = "unsupported version"; return false; }
    n_vocab_ = h[1];
    n_merges_ = h[2];
    eot_id_ = h[3];

    size_t off = 32;
    byte_to_id_ = reinterpret_cast<const uint32_t*>(base + off);
    off += 256 * sizeof(uint32_t);
    merges_ = reinterpret_cast<const uint32_t*>(base + off);
    off += (size_t)n_merges_ * 3 * sizeof(uint32_t);
    offsets_ = reinterpret_cast<const uint32_t*>(base + off);
    off += ((size_t)n_vocab_ + 1) * sizeof(uint32_t);
    blob_ = base + off;
    if (off + offsets_[n_vocab_] > (size_t)st.st_size) {
      munmap(m, st.st_size); err = "file truncated"; return false;
    }

    rank_.reserve(n_merges_ * 2);
    for (uint32_t r = 0; r < n_merges_; r++)
      rank_.emplace(key(merges_[r * 3], merges_[r * 3 + 1]), r);

    map_ = m;
    map_bytes_ = st.st_size;
    return true;
  }

  std::vector<int> encode(const std::string& text) const {
    std::vector<int> out;
    size_t i = 0, n = text.size();
    const uint8_t* s = reinterpret_cast<const uint8_t*>(text.data());
    while (i < n) {
      size_t j = split(s, i, n);
      encode_word(s + i, j - i, out);
      i = j;
    }
    return out;
  }

  std::string decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) {
      if (id < 0 || (uint32_t)id >= n_vocab_) continue;
      out.append(reinterpret_cast<const char*>(blob_ + offsets_[id]),
                 offsets_[id + 1] - offsets_[id]);
    }
    return out;
  }

  uint32_t n_vocab() const { return n_vocab_; }
  uint32_t eot_id() const { return eot_id_; }

 private:
  static uint64_t key(uint32_t a, uint32_t b) { return ((uint64_t)a << 32) | b; }

  static bool is_space(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
  }
  static bool is_digit(uint8_t c) { return c >= '0' && c <= '9'; }
  // Treats every non-ASCII byte as a letter
  static bool is_letter(uint8_t c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80;
  }

  // One step of GPT-2's pre-tokenizer pattern
  static size_t split(const uint8_t* s, size_t i, size_t n) {
    if (s[i] == '\'' && i + 1 < n) {
      uint8_t c = s[i + 1];
      if (c == 's' || c == 't' || c == 'm' || c == 'd') return i + 2;
      if (i + 2 < n) {
        if ((c == 'r' && s[i + 2] == 'e') || (c == 'v' && s[i + 2] == 'e') ||
            (c == 'l' && s[i + 2] == 'l'))
          return i + 3;
      }
    }
    // A single leading space binds to the word that follows it
    size_t j = (s[i] == ' ') ? i + 1 : i;
    if (j < n && is_letter(s[j])) {
      while (j < n && is_letter(s[j])) j++;
      return j;
    }
    if (j < n && is_digit(s[j])) {
      while (j < n && is_digit(s[j])) j++;
      return j;
    }
    if (j < n && !is_space(s[j])) {
      while (j < n && !is_space(s[j]) && !is_letter(s[j]) && !is_digit(s[j])) j++;
      return j;
    }
    // Whitespace run. If a non-space follows, the last space belongs to it
    j = i;
    while (j < n && is_space(s[j])) j++;
    if (j < n && j > i + 1) j--;
    return j;
  }

  void encode_word(const uint8_t* w, size_t len, std::vector<int>& out) const {
    if (len == 0) return;
    ids_.clear();
    for (size_t k = 0; k < len; k++) ids_.push_back(byte_to_id_[w[k]]);

    // Merge the lowest-ranked adjacent pair until none match. Words are a few
    // bytes long, so the quadratic scan beats maintaining a heap.
    while (ids_.size() > 1) {
      uint32_t best = UINT32_MAX;
      size_t at = 0;
      for (size_t k = 0; k + 1 < ids_.size(); k++) {
        auto it = rank_.find(key(ids_[k], ids_[k + 1]));
        if (it != rank_.end() && it->second < best) { best = it->second; at = k; }
      }
      if (best == UINT32_MAX) break;
      ids_[at] = merges_[(size_t)best * 3 + 2];
      ids_.erase(ids_.begin() + at + 1);
    }
    out.insert(out.end(), ids_.begin(), ids_.end());
  }

  void* map_ = nullptr;
  size_t map_bytes_ = 0;
  uint32_t n_vocab_ = 0, n_merges_ = 0, eot_id_ = 0;
  const uint32_t* byte_to_id_ = nullptr;
  const uint32_t* merges_ = nullptr;
  const uint32_t* offsets_ = nullptr;
  const uint8_t* blob_ = nullptr;
  std::unordered_map<uint64_t, uint32_t> rank_;
  mutable std::vector<uint32_t> ids_;  // scratch, reused across words
};

}  // namespace gpt2
