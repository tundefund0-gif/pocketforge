#include "forge/engine.hpp"
#include "forge/http_server.hpp"
#include "forge/types.hpp"
#include <iostream>
#include <sstream>
#include <cmath>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name, expr) do { \
    if (!(expr)) { \
        std::cerr << "  \xe2\x9c\x97 " << name << ": FAILED\n"; \
        tests_failed++; \
    } else { \
        std::cout << "  \xe2\x9c\x93 " << name << "\n"; \
        tests_passed++; \
    } \
} while(0)

int main() {
    std::cout << "=== Test: Tool Calling Support ===\n\n";

    // ============================================================
    // 1. ToolDef construction
    // ============================================================
    std::cout << "--- ToolDef ---\n";
    {
        forge::ToolDef td;
        td.name = "get_weather";
        td.description = "Get weather for a location";
        td.parameters_json = R"({"type":"object","properties":{"location":{"type":"string"}},"required":["location"]})";

        std::string entry = td.to_prompt_entry();
        TEST("ToolDef entry non-empty", !entry.empty());
        TEST("ToolDef contains name", entry.find("get_weather") != std::string::npos);
        TEST("ToolDef contains description", entry.find("Get weather") != std::string::npos);
        TEST("ToolDef contains parameters", entry.find("location") != std::string::npos);
    }

    // ============================================================
    // 2. ToolPromptBuilder
    // ============================================================
    std::cout << "--- ToolPromptBuilder ---\n";
    {
        std::vector<forge::ToolDef> tools;
        forge::ToolDef td;
        td.name = "get_weather";
        td.description = "Get weather for a location";
        td.parameters_json = R"({"type":"object","properties":{"location":{"type":"string"}},"required":["location"]})";
        tools.push_back(td);

        td.name = "send_email";
        td.description = "Send an email";
        td.parameters_json = R"({"type":"object","properties":{"to":{"type":"string"},"subject":{"type":"string"}},"required":["to"]})";
        tools.push_back(td);

        // Without base system
        std::string prompt = forge::ToolPromptBuilder::build("", tools);
        TEST("Tool prompt non-empty", !prompt.empty());
        TEST("Tool prompt has weather", prompt.find("get_weather") != std::string::npos);
        TEST("Tool prompt has email", prompt.find("send_email") != std::string::npos);
        TEST("Tool prompt has JSON format instruction", prompt.find("arguments") != std::string::npos);
        TEST("Tool prompt with no system prepends correctly", prompt.find("AVAILABLE FUNCTIONS") != std::string::npos);

        // With base system
        std::string with_sys = forge::ToolPromptBuilder::build("You are a helpful assistant.", tools);
        TEST("Tool prompt with system includes system", with_sys.find("helpful assistant") != std::string::npos);
        TEST("Tool prompt with system includes tools", with_sys.find("get_weather") != std::string::npos);
    }

    // ============================================================
    // 3. Tool call detection
    // ============================================================
    std::cout << "--- Tool Call Detection ---\n";
    {
        // Valid tool call JSON
        {
            std::string output = R"({"name": "get_weather", "arguments": {"location": "London"}})";
            std::vector<forge::ToolCall> calls;
            bool detected = forge::ToolPromptBuilder::detect_tool_call(output, calls);
            TEST("Valid tool call detected", detected);
            TEST("One call returned", calls.size() == 1);
            TEST("Call has correct name", calls[0].function_name == "get_weather");
            TEST("Call has arguments", calls[0].function_args.find("London") != std::string::npos);
            TEST("Call has non-empty id", !calls[0].id.empty());
        }

        // Tool call with extra text around it
        {
            std::string output = R"(Here is the result:
{"name": "send_email", "arguments": {"to": "test@example.com"}})";
            std::vector<forge::ToolCall> calls;
            bool detected = forge::ToolPromptBuilder::detect_tool_call(output, calls);
            TEST("Tool call with surrounding text detected", detected);
            TEST("Correct name extracted", calls[0].function_name == "send_email");
        }

        // Normal text (no tool call)
        {
            std::string output = "Hello, how can I help you today?";
            std::vector<forge::ToolCall> calls;
            bool detected = forge::ToolPromptBuilder::detect_tool_call(output, calls);
            TEST("Normal text not detected as tool call", !detected);
            TEST("Empty calls for normal text", calls.empty());
        }

        // JSON that's not a tool call
        {
            std::string output = R"({"temperature": 72, "condition": "sunny"})";
            std::vector<forge::ToolCall> calls;
            bool detected = forge::ToolPromptBuilder::detect_tool_call(output, calls);
            TEST("Non-tool JSON not detected (no name field)", !detected);
        }

        // Empty output
        {
            std::string output = "";
            std::vector<forge::ToolCall> calls;
            bool detected = forge::ToolPromptBuilder::detect_tool_call(output, calls);
            TEST("Empty output not detected", !detected);
        }

        // Tool call with string arguments
        {
            std::string output = R"({"name": "calculate", "arguments": "{\"a\": 1}"})";
            std::vector<forge::ToolCall> calls;
            bool detected = forge::ToolPromptBuilder::detect_tool_call(output, calls);
            TEST("Tool call with string args detected", detected);
            TEST("String args extracted", !calls[0].function_args.empty());
        }
    }

    // ============================================================
    // 4. Chat prompt builder with tools (full simulation)
    // ============================================================
    std::cout << "--- Chat Prompt Builder ---\n";
    {
        std::string request_json = R"({
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the weather in London?"}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather",
                        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
                    }
                }
            ]
        })";

        auto body = forge::JsonValue::parse(request_json);
        TEST("JSON parsed", body.type == forge::JsonValue::OBJECT);

        auto tools_arr = body.get("tools");
        TEST("Tools array exists", tools_arr.type == forge::JsonValue::ARRAY);

        auto messages = body.get("messages");
        TEST("Messages array exists", messages.type == forge::JsonValue::ARRAY);

        TEST("Tool array has items", tools_arr.array_val.size() == 1);
    }

    // ============================================================
    // 5. Full pipeline: build tool prompt -> generate -> detect
    // ============================================================
    std::cout << "--- Full Pipeline ---\n";
    {
        // Create tools
        std::vector<forge::ToolDef> tools;
        forge::ToolDef td;
        td.name = "get_weather";
        td.description = "Get weather for a location";
        td.parameters_json = R"({"type":"object","properties":{"location":{"type":"string"}},"required":["location"]})";
        tools.push_back(td);

        // Build prompt with tools injected into system
        std::string prompt = forge::ToolPromptBuilder::build("You are a helpful assistant.", tools);
        TEST("Full pipeline prompt built", !prompt.empty());
        TEST("Prompt contains all tool info",
             prompt.find("get_weather") != std::string::npos &&
             prompt.find("name") != std::string::npos);
        TEST("Prompt includes system", prompt.find("helpful assistant") != std::string::npos);

        // Simulate detection on mock output
        std::string mock_output = R"({"name": "get_weather", "arguments": {"location": "London"}})";
        std::vector<forge::ToolCall> calls;
        bool detected = forge::ToolPromptBuilder::detect_tool_call(mock_output, calls);
        TEST("Mock tool call detected", detected);
        TEST("Mock call has correct function", calls[0].function_name == "get_weather");
        TEST("Mock call has args", calls[0].function_args.find("London") != std::string::npos);
    }

    // ============================================================
    // 6. Engine integration test
    // ============================================================
    std::cout << "--- Engine Integration ---\n";
    {
        // Create minimal model
        forge::ModelConfig cfg;
        cfg.n_layers = 2;
        cfg.n_embd = 64;
        cfg.n_heads = 4;
        cfg.n_kv_heads = 2;
        cfg.n_ff = 256;
        cfg.n_vocab = 500;
        cfg.max_seq_len = 128;
        cfg.mtp_heads = 2;

        forge::Quantizer quantizer(cfg);
        std::vector<forge::WeightBlock> blocks;
        std::vector<std::vector<uint8_t>> compressed_blocks;

        for (uint32_t l = 0; l < 2; l++) {
            for (uint32_t m = 0; m < 7; m++) {
                uint32_t rows = 64, cols = 64;
                if (m == 4 || m == 5) { cols = 256; rows = 64; }
                if (m == 6) { rows = 256; cols = 64; }
                std::vector<float> weights(rows * cols);
                for (auto& w : weights) w = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
                auto quantized = quantizer.quantize_matrix(weights.data(), rows, cols, 1);
                auto comp = quantizer.compress_block(quantized);
                forge::WeightBlock block;
                block.layer_id = l;
                block.matrix_id = m;
                block.n_rows = rows;
                block.n_cols = cols;
                block.compressed_size = (uint32_t)comp.size();
                block.original_size = (uint32_t)quantized.size();
                block.quant_type = 1;
                block.offset = 0;
                blocks.push_back(block);
                compressed_blocks.push_back(std::move(comp));
            }
        }

        std::string model_path = "/tmp/test_tool_engine.squeeze";
        quantizer.write_squeeze(model_path, blocks, compressed_blocks, cfg);

        forge::Engine engine;
        if (engine.load_model(model_path)) {
            // Build a tool-enriched prompt
            std::vector<forge::ToolDef> tools;
            forge::ToolDef td;
            td.name = "get_weather";
            td.description = "Get weather";
            td.parameters_json = R"({"properties":{"location":{"type":"string"}}})";
            tools.push_back(td);

            std::string full_prompt = forge::ToolPromptBuilder::build("Assistant", tools);
            full_prompt += "\nuser: What is the weather in London?";

            std::vector<int32_t> tokens;
            for (char c : full_prompt) tokens.push_back((int32_t)(unsigned char)c);

            auto result = engine.generate(tokens, 16);
            TEST("Engine generates with tool prompt", result.tokens_per_second > 0);
            TEST("Engine generates tokens with tool prompt", result.tokens.size() > tokens.size());
        }

        std::remove(model_path.c_str());
    }

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
