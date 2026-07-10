#include "forge/tokenizer.hpp"
#include <algorithm>
#include <iostream>
#include <sstream>

namespace forge {

BPETokenizer::BPETokenizer() {
    build_byte_tables();
}

void BPETokenizer::build_byte_tables() {
    byte_encoder_.resize(256);
    
    // GPT-2 byte-to-unicode mapping
    // Printable ASCII (33-126) and Latin-1 (161-172, 174-255) map to same codepoints
    // Other bytes are mapped to codepoints 256+ sequentially
    bool in_special[256] = {false};
    for (int b = 33; b < 127; b++) in_special[b] = true;
    for (int b = 161; b < 173; b++) in_special[b] = true;
    for (int b = 174; b < 256; b++) in_special[b] = true;
    
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (in_special[b]) {
            byte_encoder_[b] = std::string(1, (char)b);
        } else {
            // Codepoint 256 + n (encoded as UTF-8)
            int cp = 256 + n;
            n++;
            if (cp < 0x80) {
                byte_encoder_[b] = std::string(1, (char)cp);
            } else if (cp < 0x800) {
                byte_encoder_[b] = {(char)(0xC0 | (cp >> 6)), (char)(0x80 | (cp & 0x3F))};
            } else {
                byte_encoder_[b] = {(char)(0xE0 | (cp >> 12)), (char)(0x80 | ((cp >> 6) & 0x3F)), (char)(0x80 | (cp & 0x3F))};
            }
        }
    }
    
    // Build reverse decoder
    for (int b = 0; b < 256; b++) {
        byte_decoder_[byte_encoder_[b]] = (uint8_t)b;
    }
}

void BPETokenizer::load_tokens(const std::vector<std::string>& token_strings) {
    token_strings_ = token_strings;
    token_to_id_.clear();
    for (size_t i = 0; i < token_strings_.size(); i++) {
        token_to_id_[token_strings_[i]] = (int32_t)i;
    }
}

void BPETokenizer::load_merges(const std::vector<std::string>& merge_strings) {
    merge_ranks_.clear();
    for (size_t i = 0; i < merge_strings.size(); i++) {
        // Each merge string is like "Ġ t" (pair separated by space)
        const auto& s = merge_strings[i];
        auto space_pos = s.find(' ');
        if (space_pos != std::string::npos && space_pos > 0 && space_pos + 1 < s.size()) {
            std::string first = s.substr(0, space_pos);
            std::string second = s.substr(space_pos + 1);
            std::string key = first + " " + second;
            merge_ranks_[key] = (int)i;
        }
    }
}

const std::string& BPETokenizer::token_str(int32_t id) const {
    static const std::string empty;
    if (id >= 0 && id < (int32_t)token_strings_.size()) {
        return token_strings_[id];
    }
    return empty;
}

std::string BPETokenizer::decode(int32_t token_id) const {
    if (token_id < 0 || token_id >= (int32_t)token_strings_.size()) {
        return "[" + std::to_string(token_id) + "]";
    }
    
    const std::string& raw = token_strings_[token_id];
    std::string result;
    
    // Decode byte-to-unicode: convert unicode strings back to raw bytes
    for (size_t i = 0; i < raw.size(); ) {
        unsigned char c = (unsigned char)raw[i];
        int cp;
        int char_len;
        
        // Decode UTF-8 codepoint
        if (c < 0x80) {
            cp = c;
            char_len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            if (i + 1 >= raw.size()) break;
            cp = ((c & 0x1F) << 6) | (raw[i+1] & 0x3F);
            char_len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            if (i + 2 >= raw.size()) break;
            cp = ((c & 0x0F) << 12) | ((raw[i+1] & 0x3F) << 6) | (raw[i+2] & 0x3F);
            char_len = 3;
        } else {
            char_len = 1; // Skip invalid
            i++;
            continue;
        }
        
        // Reverse byte-to-unicode mapping
        if (cp >= 33 && cp <= 126) {
            result += (char)cp;
        } else if (cp >= 161 && cp <= 172) {
            result += (char)cp;
        } else if (cp >= 174 && cp <= 255) {
            result += (char)cp;
        } else if (cp >= 256 && cp <= 288) {
            result += (char)(cp - 256);  // bytes 0-32
        } else if (cp >= 289 && cp <= 322) {
            result += (char)(cp - 162);  // bytes 127-160
        } else if (cp == 323) {
            result += (char)173;
        } else {
            // For special tokens, output as-is
            result.append(raw.c_str() + i, char_len);
        }
        
        i += char_len;
    }
    
    return result;
}

std::vector<std::string> BPETokenizer::bytes_to_unicode_strings(const std::string& text) const {
    std::vector<std::string> result;
    for (unsigned char c : text) {
        result.push_back(byte_encoder_[c]);
    }
    return result;
}

int BPETokenizer::get_pair_rank(int32_t id1, int32_t id2) const {
    if (id1 < 0 || id1 >= (int32_t)token_strings_.size()) return 999999;
    if (id2 < 0 || id2 >= (int32_t)token_strings_.size()) return 999999;
    
    std::string key = token_strings_[id1] + " " + token_strings_[id2];
    auto it = merge_ranks_.find(key);
    if (it != merge_ranks_.end()) return it->second;
    return 999999;
}

std::vector<int32_t> BPETokenizer::bpe_merge(const std::vector<int32_t>& symbol_ids) const {
    if (symbol_ids.size() <= 1) return symbol_ids;
    
    // Work with a mutable list
    std::vector<int32_t> symbols = symbol_ids;
    
    while (symbols.size() > 1) {
        // Find the pair with the lowest rank
        int best_rank = 999999;
        size_t best_pos = 0;
        
        for (size_t i = 0; i + 1 < symbols.size(); i++) {
            int rank = get_pair_rank(symbols[i], symbols[i+1]);
            if (rank < best_rank) {
                best_rank = rank;
                best_pos = i;
                if (rank == 0) break; // Can't beat rank 0
            }
        }
        
        if (best_rank >= 999999) break; // No more merges possible
        
        // Merge the pair at best_pos
        std::vector<int32_t> new_symbols;
        for (size_t i = 0; i < symbols.size(); i++) {
            if (i == best_pos) {
                // Find the token ID for the merged string
                std::string merged = token_strings_[symbols[i]] + token_strings_[symbols[i+1]];
                auto it = token_to_id_.find(merged);
                if (it != token_to_id_.end()) {
                    new_symbols.push_back(it->second);
                } else {
                    // Should not happen for valid BPE, but fallback
                    new_symbols.push_back(symbols[i]);
                    new_symbols.push_back(symbols[i+1]);
                }
                i++; // Skip next
            } else {
                new_symbols.push_back(symbols[i]);
            }
        }
        symbols = std::move(new_symbols);
    }
    
    return symbols;
}

std::vector<int32_t> BPETokenizer::encode(const std::string& text) const {
    if (token_strings_.empty()) return {};
    
    // Step 1: Convert bytes to unicode strings
    auto unicode_chars = bytes_to_unicode_strings(text);
    
    // Step 2: Map each unicode char to its token ID
    // If a unicode char isn't directly in the vocab, it might be a byte-level token
    std::vector<int32_t> symbol_ids;
    for (const auto& uc : unicode_chars) {
        auto it = token_to_id_.find(uc);
        if (it != token_to_id_.end()) {
            symbol_ids.push_back(it->second);
        } else {
            // Try as raw byte
            unsigned char byte_val = 0;
            bool found = false;
            for (int b = 0; b < 256; b++) {
                if (byte_encoder_[b] == uc) {
                    byte_val = (unsigned char)b;
                    found = true;
                    break;
                }
            }
            if (found) {
                std::string byte_str(1, (char)byte_val);
                auto it2 = token_to_id_.find(byte_str);
                if (it2 != token_to_id_.end()) {
                    symbol_ids.push_back(it2->second);
                }
            }
        }
    }
    
    if (symbol_ids.empty()) return {};
    
    // Step 3: Apply BPE merges to the sequence
    // Split by word boundaries (spaces), apply BPE per word
    // For simplicity: apply BPE to the whole sequence
    auto result = bpe_merge(symbol_ids);
    
    return result;
}

} // namespace forge
