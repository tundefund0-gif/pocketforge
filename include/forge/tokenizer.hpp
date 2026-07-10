#ifndef FORGE_TOKENIZER_HPP
#define FORGE_TOKENIZER_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace forge {

// ============================================================
//  GPT-2 BPE Tokenizer
// ============================================================

class BPETokenizer {
public:
    BPETokenizer();

    // Load token data from GGUF-style arrays
    void load_tokens(const std::vector<std::string>& token_strings);
    void load_merges(const std::vector<std::string>& merge_strings);

    // Check if tokenizer has been loaded
    bool is_loaded() const { return !token_strings_.empty(); }
    size_t vocab_size() const { return token_strings_.size(); }

    // Encode text → token IDs
    std::vector<int32_t> encode(const std::string& text) const;

    // Decode token ID → text
    std::string decode(int32_t token_id) const;

    // Flush any buffered decode state (for new text)
    void reset() {}

    // Get the raw token string (for debugging)
    const std::string& token_str(int32_t id) const;

private:
    std::vector<std::string> token_strings_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<std::string, int> merge_ranks_;

    // Byte-to-unicode encoder/decoder tables
    std::vector<std::string> byte_encoder_;  // byte → unicode string
    std::unordered_map<std::string, uint8_t> byte_decoder_;  // unicode string → byte

    // Build the GPT-2 byte-to-unicode mapping
    void build_byte_tables();

    // BPE merge: convert a word (list of symbols) into merged token IDs
    std::vector<int32_t> bpe_merge(const std::vector<int32_t>& symbol_ids) const;

    // Get the merge rank for a pair of token IDs
    int get_pair_rank(int32_t id1, int32_t id2) const;

    // Convert input bytes to unicode strings (GPT-2 encoding)
    std::vector<std::string> bytes_to_unicode_strings(const std::string& text) const;
};

} // namespace forge

#endif // FORGE_TOKENIZER_HPP
