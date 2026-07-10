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
    // Build prompt with tool definitions for MiniCPM5 XML format
    // Format: <tools>{...tools...}</tools>\n<calls>
    static std::string build(
        const std::string& base_system,
        const std::vector<ToolDef>& tools,
        const ToolChoice& choice = ToolChoice()
    ) {
        if (tools.empty() || choice.policy == ToolChoice::NONE) {
            return base_system;
        }
        std::string prompt;
        if (choice.policy == ToolChoice::REQUIRED && !choice.function_name.empty()) {
            prompt += "You must call the '" + choice.function_name + "' function. ";
        }
        if (!base_system.empty()) {
            prompt += base_system + " ";
        }
        prompt += "<tools>";
        for (size_t i = 0; i < tools.size(); i++) {
            if (i > 0) prompt += "\n";
            prompt += "{\"name\":\"" + tools[i].name + "\",";
            prompt += "\"description\":\"" + tools[i].description + "\"";
            if (!tools[i].parameters_json.empty()) {
                prompt += ",\"parameters\":" + tools[i].parameters_json;
            }
            prompt += "}";
        }
        prompt += "</tools>\n";
        prompt += "<calls>";
        return prompt;
    }

    static ToolChoice parse_tool_choice(const std::string& value_json) {
        ToolChoice choice;
        if (value_json == "\"auto\"") choice.policy = ToolChoice::AUTO;
        else if (value_json == "\"none\"") choice.policy = ToolChoice::NONE;
        else if (value_json == "\"required\"") choice.policy = ToolChoice::REQUIRED;
        else if (!value_json.empty() && value_json[0] == '{') {
            auto name_pos = value_json.find("\"name\"");
            if (name_pos != std::string::npos) {
                auto colon = value_json.find(':', name_pos + 6);
                if (colon != std::string::npos) {
                    auto q1 = value_json.find('"', colon + 1);
                    if (q1 != std::string::npos) {
                        auto q2 = value_json.find('"', q1 + 1);
                        if (q2 != std::string::npos) {
                            choice.function_name = value_json.substr(q1 + 1, q2 - q1 - 1);
                            choice.policy = ToolChoice::REQUIRED;
                        }
                    }
                }
            }
        }
        return choice;
    }

    // Detect tool calls in MiniCPM5 XML format
    // Parses: <function name="tool_name"><param name="arg">value</param></function>
    static bool detect_tool_call(
        const std::string& output,
        std::vector<ToolCall>& calls_out
    ) {
        if (output.empty()) return false;
        size_t pos = 0;
        int max_calls = 10;
        int found = 0;
        while (found < max_calls) {
            auto fn_start = output.find("<function", pos);
            if (fn_start == std::string::npos) break;
            auto name_pos = output.find("name=\"", fn_start);
            if (name_pos == std::string::npos || name_pos > fn_start + 100) { pos = fn_start + 9; continue; }
            auto name_start = name_pos + 6;
            auto name_end = output.find('"', name_start);
            if (name_end == std::string::npos) break;
            std::string fn_name = output.substr(name_start, name_end - name_start);
            if (fn_name.empty()) { pos = name_end + 1; continue; }
            auto fn_close = output.find("</function>", fn_start);
            if (fn_close == std::string::npos) break;
            auto content_start = output.find('>', fn_start);
            if (content_start == std::string::npos || content_start >= fn_close) break;
            content_start++;
            std::string inner = output.substr(content_start, fn_close - content_start);
            std::string args_json = "{";
            size_t ppos = 0;
            bool first_arg = true;
            while (true) {
                auto param_start = inner.find("<param", ppos);
                if (param_start == std::string::npos) break;
                auto pname_pos = inner.find("name=\"", param_start);
                if (pname_pos == std::string::npos || pname_pos > param_start + 100) { ppos = param_start + 6; continue; }
                auto pname_start = pname_pos + 6;
                auto pname_end = inner.find('"', pname_start);
                if (pname_end == std::string::npos) break;
                std::string pname = inner.substr(pname_start, pname_end - pname_start);
                auto pcontent_start = inner.find('>', pname_end);
                if (pcontent_start == std::string::npos) break;
                pcontent_start++;
                auto pclose = inner.find("</param>", pcontent_start);
                if (pclose == std::string::npos) break;
                std::string pvalue = inner.substr(pcontent_start, pclose - pcontent_start);
                if (!first_arg) args_json += ",";
                args_json += "\"" + pname + "\":\"" + pvalue + "\"";
                first_arg = false;
                ppos = pclose + 8;
            }
            args_json += "}";
            ToolCall call;
            call.id = "call_" + std::to_string(time(nullptr));
            call.function_name = fn_name;
            call.function_args = args_json;
            calls_out.push_back(call);
            found++;
            pos = fn_close + 11;
        }
        return found > 0;
    }
};struct WeightBlock {
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
    uint32_t n_embd         = 1536;
    uint32_t n_heads        = 16;
    uint32_t n_kv_heads     = 2;    // GQA
    uint32_t n_ff           = 4608;
    uint32_t n_vocab        = 130560;
    uint32_t head_dim_      = 0;    // 0 = auto (n_embd/n_heads); set explicitly for non-standard

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
    uint32_t head_dim()    const { return head_dim_ ? head_dim_ : (n_embd / n_heads); }
    uint32_t kv_head_dim() const {
        return (head_dim_ && n_kv_heads) ? head_dim_ : (n_embd / n_kv_heads);
    }

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
