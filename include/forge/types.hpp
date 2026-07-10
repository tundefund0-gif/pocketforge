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

// ============================================================
//  Tool choice policy — matches OpenAI API
// ============================================================

struct ToolChoice {
    enum Policy { AUTO = 0, NONE = 1, REQUIRED = 2 } policy = AUTO;
    std::string function_name; // used when REQUIRED
};

// ============================================================
//  Tool prompt builder — matches llama.cpp/Ollama format
// ============================================================

struct ToolPromptBuilder {
    // Build a system prompt that injects tool definitions
    // Uses the same format as llama.cpp's server
    static std::string build(
        const std::string& base_system,
        const std::vector<ToolDef>& tools,
        const ToolChoice& choice = ToolChoice()
    ) {
        if (tools.empty() || choice.policy == ToolChoice::NONE) {
            return base_system;
        }

        std::string prompt;
        if (!base_system.empty()) {
            prompt = base_system + "\n\n";
        }
        prompt += "You have access to the following functions:\n\n";
        for (const auto& tool : tools) {
            prompt += tool.to_prompt_entry() + "\n\n";
        }

        if (choice.policy == ToolChoice::REQUIRED && !choice.function_name.empty()) {
            prompt += "You MUST call the function: " + choice.function_name + ".\n";
            prompt += "Respond with ONLY a JSON object matching that function's schema.\n";
        } else {
            prompt += "To call a function, respond with a JSON object in this format:\n";
            prompt += "{\"name\": \"function_name\", \"arguments\": {\"arg1\": \"value1\", ...}}\n";
            prompt += "\nOtherwise, respond with a normal message.\n";
        }
        return prompt;
    }

    // Parse tool_choice from request body (handles string and object forms)
    static ToolChoice parse_tool_choice(const std::string& value_json) {
        ToolChoice choice;

        // Try string form: "auto", "none", "required"
        if (value_json == "\"auto\"") {
            choice.policy = ToolChoice::AUTO;
        } else if (value_json == "\"none\"") {
            choice.policy = ToolChoice::NONE;
        } else if (value_json == "\"required\"") {
            choice.policy = ToolChoice::REQUIRED;
        } else if (!value_json.empty() && value_json[0] == '{') {
            // Object form: {"type": "function", "function": {"name": "..."}}
            // Simple string scan for "name"
            auto name_pos = value_json.find("\"name\"");
            if (name_pos != std::string::npos) {
                auto colon = value_json.find(':', name_pos + 6);
                if (colon != std::string::npos) {
                    auto quote1 = value_json.find('"', colon + 1);
                    if (quote1 != std::string::npos) {
                        auto quote2 = value_json.find('"', quote1 + 1);
                        if (quote2 != std::string::npos) {
                            choice.function_name = value_json.substr(quote1 + 1, quote2 - quote1 - 1);
                            choice.policy = ToolChoice::REQUIRED;
                        }
                    }
                }
            }
        }
        return choice;
    }

    // Try to parse output as a tool call JSON (pure string-based, no deps)
    // Handles: standard JSON, surrounded by text, markdown code blocks
    // Supports both {"name": "...", "arguments": ...} and {"function": "...", "arguments": ...}
    static bool detect_tool_call(
        const std::string& output,
        std::vector<ToolCall>& calls_out
    ) {
        if (output.empty()) return false;

        std::string cleaned = output;

        // Strip markdown code blocks if present
        auto strip_markdown = [](std::string& s) {
            auto start = s.find("```");
            while (start != std::string::npos) {
                auto end = s.find("```", start + 3);
                if (end == std::string::npos) break;
                // Extract content between code fences
                auto content_start = s.find('\n', start) + 1;
                if (content_start == std::string::npos || content_start > end) {
                    content_start = start + 3;
                    // Skip language identifier
                    auto nl = s.find('\n', content_start);
                    if (nl != std::string::npos && nl < end) content_start = nl + 1;
                }
                std::string code_content = s.substr(content_start, end - content_start);
                s = s.substr(0, start) + code_content + s.substr(end + 3);
                start = s.find("```");
            }
        };
        strip_markdown(cleaned);

        // Find outermost JSON object
        auto brace_start = cleaned.find('{');
        if (brace_start == std::string::npos) return false;

        // Count braces to find matching close
        int depth = 0;
        size_t brace_end = cleaned.size();
        for (size_t i = brace_start; i < cleaned.size(); i++) {
            if (cleaned[i] == '{') depth++;
            else if (cleaned[i] == '}') {
                depth--;
                if (depth == 0) { brace_end = i; break; }
            }
        }
        if (depth != 0) return false;

        std::string json_str = cleaned.substr(brace_start, brace_end - brace_start + 1);

        // Helper: find JSON string value for a given key
        auto get_json_string_value = [](const std::string& j, const std::string& key) -> std::string {
            auto key_pos = j.find("\"" + key + "\"");
            if (key_pos == std::string::npos) return "";
            auto colon = j.find(':', key_pos + key.size() + 2);
            if (colon == std::string::npos) return "";
            auto val_start = j.find_first_not_of(" \t\r\n", colon + 1);
            if (val_start == std::string::npos || j[val_start] != '"') return "";
            auto val_end = j.find('"', val_start + 1);
            if (val_end == std::string::npos) return "";
            return j.substr(val_start + 1, val_end - val_start - 1);
        };

        // Helper: find JSON value after a key (for objects or strings)
        auto get_json_value_after_key = [](const std::string& j, const std::string& key) -> std::string {
            auto key_pos = j.find("\"" + key + "\"");
            if (key_pos == std::string::npos) return "";
            auto colon = j.find(':', key_pos + key.size() + 2);
            if (colon == std::string::npos) return "";
            auto val_start = j.find_first_not_of(" \t\r\n", colon + 1);
            if (val_start == std::string::npos) return "";

            if (j[val_start] == '{') {
                int d = 0;
                size_t end = val_start;
                for (; end < j.size(); end++) {
                    if (j[end] == '{') d++;
                    else if (j[end] == '}') { d--; if (d == 0) break; }
                }
                if (d == 0) return j.substr(val_start, end - val_start + 1);
            } else if (j[val_start] == '"') {
                auto end = j.find('"', val_start + 1);
                if (end != std::string::npos) return j.substr(val_start, end - val_start + 1);
            }
            return "";
        };

        // Try "name" field first (standard format)
        std::string fn_name = get_json_string_value(json_str, "name");

        // Fallback: try "function" field (alternative format used by some models)
        if (fn_name.empty()) {
            fn_name = get_json_string_value(json_str, "function");
        }

        if (fn_name.empty()) return false;

        // Get arguments
        std::string args_str = get_json_value_after_key(json_str, "arguments");
        if (args_str.empty()) {
            // Try "params" or "parameters" as fallback
            args_str = get_json_value_after_key(json_str, "params");
            if (args_str.empty()) {
                args_str = get_json_value_after_key(json_str, "parameters");
            }
        }

        ToolCall call;
        call.id = "call_" + std::to_string(time(nullptr));
        call.function_name = fn_name;
        call.function_args = args_str.empty() ? "{}" : args_str;
        calls_out.push_back(call);

        // Check if there's another JSON object after this one (multiple calls)
        std::string remaining = cleaned.substr(brace_end + 1);
        auto next_brace = remaining.find('{');
        if (next_brace != std::string::npos) {
            // Recursively find more tool calls (max 10 to prevent stack overflow)
            static int depth_limit = 0;
            if (depth_limit < 10) {
                depth_limit++;
                detect_tool_call(remaining, calls_out);
                depth_limit--;
            }
        }

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
