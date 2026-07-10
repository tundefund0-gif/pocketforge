#include "forge/engine.hpp"
#include "forge/http_server.hpp"
#include <iostream>
#include <sstream>
#include <cstring>
#include <csignal>

static std::unique_ptr<forge::Engine> g_engine;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: forge-server <model.squeeze> [port]\n";
        return 1;
    }

    std::string model_path = argv[1];
    uint16_t port = (argc > 2) ? (uint16_t)std::atoi(argv[2]) : 8080;

    std::cout << "╔══════════════════════════════════════╗\n";
    std::cout << "║   PocketForge API Server v1.0       ║\n";
    std::cout << "║   < 250 MB RAM, 16K+ context        ║\n";
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
             << "    \"memory_mb\":" << (stats.total() / 1024 / 1024) << "\n"
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

            // Parse prompt
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

            // Tokenize prompt (simple char-level for now)
            std::vector<int32_t> prompt_tokens;
            for (char c : prompt) prompt_tokens.push_back((int32_t)(unsigned char)c);

            // Generate
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

            // Build response
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
                 << "    \"completion_tokens\":" << result.tokens.size() << ",\n"
                 << "    \"total_tokens\":" << (prompt_tokens.size() + result.tokens.size()) << "\n"
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
             << "  " 
             << "\"activations_kb\":" << (stats.activations / 1024) << ",\n"
             << "  \"prefetch_kb\":" << (stats.prefetch_buffer / 1024) << ",\n"
             << "  \"mtp_kb\":" << (stats.mtp_heads / 1024) << ",\n"
             << "  \"total_kb\":" << (stats.total() / 1024) << "\n"
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
    std::cout << "  GET  /v1/models       — Model info\n";
    std::cout << "  POST /v1/completions  — Generate text\n";
    std::cout << "  GET  /stats           — Memory stats\n\n";

    server.run();

    return 0;
}
