#include "forge/engine.hpp"
#include "forge/http_server.hpp"
#include <iostream>
#include <sstream>
#include <cstring>
#include <csignal>
#include <vector>

static std::unique_ptr<forge::Engine> g_engine;

// ============================================================
//  Tool calling helpers
// ============================================================

// Parse tools array from request body into ToolDef vector
static std::vector<forge::ToolDef> parse_tools(const forge::JsonValue& body) {
    std::vector<forge::ToolDef> tools;
    if (!body.has("tools")) return tools;

    auto raw_tools = body.get("tools");
    if (raw_tools.type != forge::JsonValue::ARRAY) return tools;

    for (const auto& entry : raw_tools.array_val) {
        if (entry.type != forge::JsonValue::OBJECT) continue;

        // Support both "function" type and direct format
        forge::ToolDef def;
        if (entry.has("function")) {
            // OpenAI format: {type: "function", function: {name, description, parameters}}
            auto fn = entry.get("function");
            def.name = fn.get("name").string_val;
            def.description = fn.get("description").string_val;
            if (fn.has("parameters")) {
                def.parameters_json = fn.get("parameters").serialize();
            }
        } else {
            // Direct format: {name, description, parameters}
            def.name = entry.get("name").string_val;
            def.description = entry.get("description").string_val;
            if (entry.has("parameters")) {
                def.parameters_json = entry.get("parameters").serialize();
            }
        }

        if (!def.name.empty()) {
            tools.push_back(def);
        }
    }
    return tools;
}

// Build prompt from messages array, injecting tools into system message
static std::string build_chat_prompt(
    const forge::JsonValue& messages,
    const std::vector<forge::ToolDef>& tools
) {
    if (messages.type != forge::JsonValue::ARRAY) return "";

    std::string system_prompt;
    std::string conversation;

    // First pass: find the system message and collect all messages
    for (const auto& msg : messages.array_val) {
        std::string role = msg.get("role").string_val;
        std::string content = msg.get("content").string_val;

        if (role == "system") {
            system_prompt = content;
        }
    }

    // Inject tool descriptions into system prompt
    if (!tools.empty()) {
        std::string tool_prompt = forge::ToolPromptBuilder::build(system_prompt, tools);
        if (tool_prompt != system_prompt) {
            system_prompt = tool_prompt;
        }
    }

    // Second pass: build conversation string
    for (const auto& msg : messages.array_val) {
        std::string role = msg.get("role").string_val;
        std::string content = msg.get("content").string_val;
        std::string name = msg.get("name").string_val;

        // Handle tool call responses
        bool is_tool_response = (role == "tool");
        if (is_tool_response) {
            std::string tool_call_id = msg.get("tool_call_id").string_val;
            if (!conversation.empty()) conversation += "\n";
            conversation += "[TOOL_RESULT: " + tool_call_id + "]\n" + content + "\n[/TOOL_RESULT]";
            continue;
        }

        // Handle assistant messages with tool_calls
        bool has_tool_calls = (role == "assistant" && msg.has("tool_calls"));
        if (has_tool_calls) {
            if (!conversation.empty()) conversation += "\n";
            conversation += "assistant: ";
            if (!content.empty()) conversation += content + " ";

            auto tool_calls_arr = msg.get("tool_calls");
            if (tool_calls_arr.type == forge::JsonValue::ARRAY) {
                for (const auto& tc : tool_calls_arr.array_val) {
                    auto fn = tc.get("function");
                    std::string fn_name = fn.get("name").string_val;
                    std::string fn_args = fn.get("arguments").string_val;
                    conversation += "{\"name\": \"" + fn_name + "\", \"arguments\": " + fn_args + "} ";
                }
            }
            continue;
        }

        // Normal messages
        std::string prefix = (role == "system") ? "system" : role;
        if (!conversation.empty()) conversation += "\n";
        conversation += prefix + ": " + content;
    }

    // If we have a system prompt, prepend it
    std::string result;
    if (!system_prompt.empty()) {
        result = "system: " + system_prompt + "\n\n" + conversation;
    } else {
        result = conversation;
    }

    return result;
}

// ============================================================
//  Main server
// ============================================================

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: forge-server <model.squeeze> [port]\n";
        return 1;
    }

    std::string model_path = argv[1];
    uint16_t port = (argc > 2) ? (uint16_t)std::atoi(argv[2]) : 8080;

    std::cout << "╔══════════════════════════════════════╗\n";
    std::cout << "║   PocketForge API Server v1.1       ║\n";
    std::cout << "║   < 250 MB RAM, 131K context        ║\n";
    std::cout << "║   Tool calling support               ║\n";
    std::cout << "╚══════════════════════════════════════╝\n\n";

    // Load model
    g_engine = std::make_unique<forge::Engine>();
    if (!g_engine->load_model(model_path)) {
        std::cerr << "Failed to load model: " << model_path << "\n";
        return 1;
    }

    auto cfg = g_engine->config();
    std::cout << "Model: " << cfg.n_layers << " layers, "
              << cfg.n_embd << " embd, " << cfg.n_heads << " heads\n";
    std::cout << "KV cache: " << cfg.n_kv_heads << " heads, "
              << cfg.kv_cache_size << " context\n";
    std::cout << "MTP: " << (cfg.mtp_enabled ? "enabled" : "disabled") << "\n";

    auto stats = g_engine->get_memory_stats();
    std::cout << "Estimated RAM: " << (stats.total() / 1024 / 1024) << " MB / 250 MB\n\n";

    // Create HTTP server
    forge::HttpServer server(port);

    // Health endpoint
    server.route("GET", "/health", [](const forge::HttpRequest&) -> forge::HttpResponse {
        forge::HttpResponse resp;
        resp.body = "{\"status\":\"ok\",\"engine\":\"pocketforge\"}";
        return resp;
    });

    // Models endpoint
    server.route("GET", "/v1/models", [](const forge::HttpRequest&) -> forge::HttpResponse {
        auto cfg = g_engine->config();
        auto stats = g_engine->get_memory_stats();
        std::ostringstream json;
        json << "{\n"
             << "  \"id\":\"pocketforge-1b\",\n"
             << "  \"object\":\"model\",\n"
             << "  \"created\":" << time(nullptr) << ",\n"
             << "  \"owned_by\":\"pocketforge\",\n"
             << "  \"metadata\":{\n"
             << "    \"layers\":" << cfg.n_layers << ",\n"
             << "    \"embedding_size\":" << cfg.n_embd << ",\n"
             << "    \"num_heads\":" << cfg.n_heads << ",\n"
             << "    \"num_kv_heads\":" << cfg.n_kv_heads << ",\n"
             << "    \"max_context\":" << cfg.kv_cache_size << ",\n"
             << "    \"memory_mb\":" << (stats.total() / 1024 / 1024) << ",\n"
             << "    \"supports_tool_calling\":true\n"
             << "  }\n"
             << "}";
        forge::HttpResponse resp;
        resp.body = json.str();
        return resp;
    });

    // Completions endpoint (OpenAI-compatible)
    server.route("POST", "/v1/completions", [](const forge::HttpRequest& req) -> forge::HttpResponse {
        try {
            forge::JsonValue body = forge::JsonValue::parse(req.body);

            std::string prompt = body.get("prompt").string_val;
            if (prompt.empty()) {
                forge::HttpResponse err;
                err.status_code = 400;
                err.body = "{\"error\":\"empty_prompt\"}";
                return err;
            }

            int max_tokens = (int)body.get("max_tokens").number_val;
            if (max_tokens <= 0 || max_tokens > 4096) max_tokens = 128;

            float temperature = (float)body.get("temperature").number_val;
            if (temperature <= 0) temperature = 1.0f;

            bool stream = body.get("stream").bool_val; (void)stream;

            // Tokenize
            std::vector<int32_t> prompt_tokens;
            for (char c : prompt) prompt_tokens.push_back((int32_t)(unsigned char)c);

            std::string generated_text;
            auto result = g_engine->generate(prompt_tokens, max_tokens,
                [&](int32_t token, float prob) {
                    if (token >= 32 && token < 127) {
                        generated_text += (char)token;
                    } else {
                        generated_text += "[" + std::to_string(token) + "]";
                    }
                }
            );

            std::ostringstream json;
            json << "{\n"
                 << "  \"id\":\"cmpl-" << time(nullptr) << "\",\n"
                 << "  \"object\":\"text_completion\",\n"
                 << "  \"created\":" << time(nullptr) << ",\n"
                 << "  \"model\":\"pocketforge-1b\",\n"
                 << "  \"choices\":[{\n"
                 << "    \"text\":\"" << forge::json_escape(generated_text) << "\",\n"
                 << "    \"index\":0,\n"
                 << "    \"finish_reason\":\"length\"\n"
                 << "  }],\n"
                 << "  \"usage\":{\n"
                 << "    \"prompt_tokens\":" << prompt_tokens.size() << ",\n"
                 << "    \"completion_tokens\":" << result.tokens.size() - prompt_tokens.size() << ",\n"
                 << "    \"total_tokens\":" << result.tokens.size() << "\n"
                 << "  }\n"
                 << "}";

            forge::HttpResponse resp;
            resp.body = json.str();
            return resp;

        } catch (const std::exception& e) {
            forge::HttpResponse err;
            err.status_code = 500;
            err.body = "{\"error\":\"" + std::string(e.what()) + "\"}";
            return err;
        }
    });

    // Chat completions endpoint with tool calling support
    server.route("POST", "/v1/chat/completions", [](const forge::HttpRequest& req) -> forge::HttpResponse {
        try {
            forge::JsonValue body = forge::JsonValue::parse(req.body);

            // Parse messages
            auto messages = body.get("messages");
            if (messages.type != forge::JsonValue::ARRAY || messages.array_val.empty()) {
                forge::HttpResponse err;
                err.status_code = 400;
                err.body = "{\"error\":\"empty_messages\"}";
                return err;
            }

            // Parse tools
            auto tools = parse_tools(body);
            bool has_tools = !tools.empty();

            int max_tokens = (int)body.get("max_tokens").number_val;
            if (max_tokens <= 0 || max_tokens > 4096) max_tokens = 256;

            float temperature = (float)body.get("temperature").number_val;
            if (temperature <= 0) temperature = 0.8f;

            bool stream = body.get("stream").bool_val; (void)stream;

            // Build prompt with tool injection
            std::string prompt = build_chat_prompt(messages, tools);
            if (prompt.empty()) {
                forge::HttpResponse err;
                err.status_code = 400;
                err.body = "{\"error\":\"empty_messages\"}";
                return err;
            }

            // Tokenize
            std::vector<int32_t> prompt_tokens;
            for (char c : prompt) prompt_tokens.push_back((int32_t)(unsigned char)c);

            // Setup sampling config
            forge::SamplingConfig sampling;
            sampling.temperature = temperature;
            if (body.has("top_p")) sampling.top_p = (float)body.get("top_p").number_val;
            if (body.has("top_k")) sampling.top_k = (uint32_t)body.get("top_k").number_val;

            g_engine->config().check_memory_budget();

            // Generate
            std::string generated_text;
            auto result = g_engine->generate(prompt_tokens, max_tokens,
                [&](int32_t token, float prob) {
                    if (token >= 32 && token < 127) {
                        generated_text += (char)token;
                    } else {
                        generated_text += "\\x" + std::to_string(token);
                    }
                }
            );

            // Build response
            std::ostringstream json;

            // Try to detect tool calls in the generated text
            std::vector<forge::ToolCall> detected_calls;
            bool is_tool_call = false;
            if (has_tools) {
                is_tool_call = forge::ToolPromptBuilder::detect_tool_call(
                    generated_text, detected_calls);
            }

            json << "{\n"
                 << "  \"id\":\"chatcmpl-" << time(nullptr) << "\",\n"
                 << "  \"object\":\"chat.completion\",\n"
                 << "  \"created\":" << time(nullptr) << ",\n"
                 << "  \"model\":\"pocketforge-1b\",\n"
                 << "  \"choices\":[{\n"
                 << "    \"index\":0,\n"
                 << "    \"message\":{\n"
                 << "      \"role\":\"assistant\"";

            if (is_tool_call && !detected_calls.empty()) {
                // Tool call response — content is null
                json << ",\n"
                     << "      \"content\":null,\n"
                     << "      \"tool_calls\":[\n";

                for (size_t i = 0; i < detected_calls.size(); i++) {
                    if (i > 0) json << ",\n";
                    json << "        {\n"
                         << "          \"id\":\"" << forge::json_escape(detected_calls[i].id) << "\",\n"
                         << "          \"type\":\"function\",\n"
                         << "          \"function\":{\n"
                         << "            \"name\":\"" << forge::json_escape(detected_calls[i].function_name) << "\",\n"
                         << "            \"arguments\":\"" << forge::json_escape(detected_calls[i].function_args) << "\"\n"
                         << "          }\n"
                         << "        }";
                }
                json << "\n"
                     << "      ]\n";
            } else {
                // Normal text response
                json << ",\n"
                     << "      \"content\":\"" << forge::json_escape(generated_text) << "\"\n";
            }

            json << "    },\n"
                 << "    \"finish_reason\":\"" << (is_tool_call ? "tool_calls" : "length") << "\"\n"
                 << "  }],\n"
                 << "  \"usage\":{\n"
                 << "    \"prompt_tokens\":" << prompt_tokens.size() << ",\n"
                 << "    \"completion_tokens\":" << (result.tokens.size() - prompt_tokens.size()) << ",\n"
                 << "    \"total_tokens\":" << result.tokens.size() << ",\n"
                 << "    \"mtp_accepted\":" << result.mtp_accepted << ",\n"
                 << "    \"mtp_rejected\":" << result.mtp_rejected << "\n"
                 << "  }\n"
                 << "}";

            forge::HttpResponse resp;
            resp.body = json.str();
            return resp;

        } catch (const std::exception& e) {
            forge::HttpResponse err;
            err.status_code = 500;
            err.body = "{\"error\":\"" + std::string(e.what()) + "\"}";
            return err;
        }
    });

    // Stats endpoint
    server.route("GET", "/stats", [](const forge::HttpRequest&) -> forge::HttpResponse {
        auto stats = g_engine->get_memory_stats();
        std::ostringstream json;
        json << "{\n"
             << "  \"kv_cache_kb\":" << (stats.kv_cache / 1024) << ",\n"
             << "  \"activations_kb\":" << (stats.activations / 1024) << ",\n"
             << "  \"prefetch_kb\":" << (stats.prefetch_buffer / 1024) << ",\n"
             << "  \"mtp_kb\":" << (stats.mtp_heads / 1024) << ",\n"
             << "  \"other_kb\":" << (stats.other / 1024) << ",\n"
             << "  \"total_kb\":" << (stats.total() / 1024) << ",\n"
             << "  \"tool_calling\":true\n"
             << "}";
        forge::HttpResponse resp;
        resp.body = json.str();
        return resp;
    });

    // Start server
    if (!server.start()) {
        std::cerr << "Failed to start server on port " << port << "\n";
        return 1;
    }

    std::cout << "Ready — listening on http://0.0.0.0:" << port << "\n";
    std::cout << "Endpoints:\n";
    std::cout << "  GET  /health          — Health check\n";
    std::cout << "  GET  /v1/models       — Model info (tool_calling: true)\n";
    std::cout << "  POST /v1/completions  — Generate text\n";
    std::cout << "  POST /v1/chat/completions  — Chat + Tool Calling\n";
    std::cout << "  GET  /stats           — Memory stats\n\n";

    server.run();

    return 0;
}
