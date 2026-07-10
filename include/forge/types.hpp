#ifndef FORGE_TYPES_HPP
#define FORGE_TYPES_HPP

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <memory>
#include <functional>

namespace forge {

// ============================================================
//  Quantized types
// ============================================================

struct Q8_0 {
    float d;
    int8_t q[32];
};

struct Q4_0 {
    float d;
    uint8_t q[16];
};

struct Q2_0 {
    float d;
    uint8_t q[8];
};

struct Q1_5 {
    uint8_t q[16];
};

struct SamplingConfig {
    float temperature  = 0.8f;
    float top_p        = 0.95f;
    uint32_t top_k     = 40;
    bool   greedy      = false;
};

struct GenerationConfig {
    SamplingConfig sampling;
    uint32_t max_new_tokens  = 256;
    uint32_t max_context     = 131072;
    bool     mtp_enabled     = true;
    bool     stream          = false;
    float    skip_threshold  = 0.01f;
};

struct ToolCall {
    std::string id;
    std::string type = "function";
    std::string function_name;
    std::string function_args; // JSON string of arguments
};

struct ToolDef {
    std::string name;
    std::string description;
    std::string parameters_json; // JSON schema

    std::string to_prompt_entry() const {
        std::string out;
        out += "{\n";
        out += "  \"name\": \"" + name + "\",\n";
        out += "  \"description\": \"" + description + "\",\n";
        out += "  \"parameters\": " + parameters_json + "\n";
        out += "}";
        return out;
    }
};

struct ToolPromptBuilder {
    // Build a system prompt that injects tool definitions
    static std::string build(
        const std::string& base_system,
        const std::vector<ToolDef>& tools
    ) {
        if (tools.empty()) return base_system;

        std::string prompt;
        if (!base_system.empty()) {
            prompt = base_system + "\n\n";
        }
        prompt += "You have access to the following functions. Use them when the user asks for something that matches a function's purpose.\n\n";
        prompt += "AVAILABLE FUNCTIONS:\n";
        for (const auto& tool : tools) {
            prompt += tool.to_prompt_entry() + "\n\n";
        }
        prompt += "To call a function, respond with ONLY a valid JSON object in exactly this format (no other text):\n";
        prompt += "{\"name\": \"function_name\", \"arguments\": {\"param1\": \"value1\", ...}}\n";
        prompt += "\nIf no function is needed, respond with a normal message.\n";
        return prompt;
    }

    // Try to parse output as a tool call JSON (pure string-based, no deps)
    // Returns true if a tool call was detected
    static bool detect_tool_call(
        const std::string& output,
        std::vector<ToolCall>& calls_out
    ) {
        // Find first JSON object in output
        size_t brace_start = output.find('{');
        if (brace_start == std::string::npos) return false;

        size_t brace_end = output.rfind('}');
        if (brace_end == std::string::npos || brace_end <= brace_start) return false;

        std::string json_str = output.substr(brace_start, brace_end - brace_start + 1);

        // Look for "name" field in the JSON
        auto name_key = json_str.find("\"name\"");
        if (name_key == std::string::npos) return false;

        // Find the value after "name":
        auto colon = json_str.find(':', name_key + 6);
        if (colon == std::string::npos) return false;

        // Skip whitespace
        auto val_start = json_str.find_first_not_of(" \t", colon + 1);
        if (val_start == std::string::npos) return false;

        // Read the string value
        std::string fn_name;
        if (json_str[val_start] == '"') {
            auto str_end = json_str.find('"', val_start + 1);
            if (str_end != std::string::npos) {
                fn_name = json_str.substr(val_start + 1, str_end - val_start - 1);
            }
        }
        if (fn_name.empty()) return false;

        // Look for "arguments" field
        auto args_key = json_str.find("\"arguments\"");
        if (args_key == std::string::npos) return false;

        auto args_colon = json_str.find(':', args_key + 10);
        if (args_colon == std::string::npos) return false;

        auto args_start = json_str.find_first_not_of(" \t", args_colon + 1);
        if (args_start == std::string::npos) return false;

        std::string args_str;
        if (json_str[args_start] == '{') {
            // Object arguments - find matching closing brace
            int depth = 0;
            size_t args_end = args_start;
            for (; args_end < json_str.size(); args_end++) {
                if (json_str[args_end] == '{') depth++;
                else if (json_str[args_end] == '}') { depth--; if (depth == 0) break; }
            }
            if (depth == 0 && args_end > args_start) {
                args_str = json_str.substr(args_start, args_end - args_start + 1);
            }
        } else if (json_str[args_start] == '"') {
            // String arguments - find closing quote
            auto str_end = json_str.find('"', args_start + 1);
            if (str_end != std::string::npos) {
                args_str = json_str.substr(args_start, str_end - args_start + 1);
            }
        }

        ToolCall call;
        call.id = "call_" + std::to_string(time(nullptr));
        call.function_name = fn_name;
        call.function_args = args_str.empty() ? "{}" : args_str;
        calls_out.push_back(call);
        return true;
    }
};

struct WeightBlock {
    uint32_t layer_id;
    uint32_t matrix_id;
    uint32_t n_rows;
    uint32_t n_cols;
    uint32_t compressed_size;
    uint32_t original_size;
    uint8_t  quant_type;
    uint32_t offset;
};

// ============================================================
//  Model configuration — SANE DEFAULTS for 250MB budget
// ============================================================

struct ModelConfig {
    // Architecture (1B-scale)
    uint32_t n_layers       = 24;
    uint32_t n_embd         = 2048;
    uint32_t n_heads        = 32;
    uint32_t n_kv_heads     = 4;    // GQA
    uint32_t n_ff           = 8192;
    uint32_t n_vocab        = 32000;

    // Memory-safe defaults
    uint32_t max_seq_len    = 131072; // 131K context
    bool     mtp_enabled    = true;
    uint32_t mtp_heads      = 4;

    // Layer skip
    float    skip_threshold = 0.01f;
    uint32_t skip_interval  = 4;

    // KV cache: slide window size (tokens kept)
    uint32_t kv_cache_size  = 8192;   // sliding window size

    // Hard memory cap
        size_t   max_memory     = 250 * 1024 * 1024; // 250 MB
    bool     use_bpe        = false;  // false = char-level ASCII fallback

    // Derived
    uint32_t head_dim()    const { return n_embd / n_heads; }
    uint32_t kv_head_dim() const { return n_embd / n_kv_heads; }

    // Memory budget estimate: returns OK or prints warning
    bool check_memory_budget() const;
};

// ============================================================
//  Generation result
// ============================================================

struct GenerationResult {
    std::vector<int32_t> tokens;
    float                tokens_per_second = 0.0f;
    size_t               peak_memory_bytes = 0;
    int                  mtp_accepted      = 0;
    int                  layers_skipped    = 0;
        bool                 oom_protected     = false; // set if we hit memory cap
    int                  mtp_rejected      = 0;
};

using TokenCallback = std::function<void(int32_t token, float prob)>;
using ProgressCallback = std::function<void(float progress)>;

} // namespace forge
#endif // FORGE_TYPES_HPP
