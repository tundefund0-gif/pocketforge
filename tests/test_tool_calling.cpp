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
    std::cout << "=== Test: Tool Calling Support (MiniCPM5 XML) ===\n\n";

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
    // 2. ToolPromptBuilder (MiniCPM5 XML format)
    // ============================================================
    std::cout << "--- ToolPromptBuilder (XML format) ---\n";
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
        TEST("Tool prompt has <tools> tag", prompt.find("<tools>") != std::string::npos);
        TEST("Tool prompt has <calls> tag", prompt.find("<calls>") != std::string::npos);
        TEST("Tool prompt has weather", prompt.find("get_weather") != std::string::npos);
        TEST("Tool prompt has email", prompt.find("send_email") != std::string::npos);

        // With base system
        std::string with_sys = forge::ToolPromptBuilder::build("You are a helpful assistant.", tools);
        TEST("Tool prompt with system includes tools", with_sys.find("get_weather") != std::string::npos);
        TEST("Tool prompt has tool definitions in JSON format", with_sys.find("\"name\":") != std::string::npos);
    }

    // ============================================================
    // 3. Tool call detection (MiniCPM5 XML)
    // ============================================================
    std::cout << "--- XML Tool Call Detection ---\n";
    {
        // Valid single XML tool call
        {
            std::string output = R"(<function name="get_weather"><param name="location">London</param></function>)";
            std::vector<forge::ToolCall> calls;
            bool detected = forge::ToolPromptBuilder::detect_tool_call(output, calls);
            TEST("Valid XML tool call detected", detected);
            TEST("One call returned", calls.size() == 1);
            TEST("Call name = get_weather", calls[0].function_name == "get_weather");
            TEST("Call has args", calls[0].function_args.find("London") != std::string::npos);
            TEST("Call has non-empty id", !calls[0].id.empty());
        }

        // XML with surrounding text
        {
            std::string output = "Based on your request\n<function name=\"send_email\"><param name=\"to\">test@example.com</param></function>\nSent!";
            std::vector<forge::ToolCall> calls;
            bool detected = forge::ToolPromptBuilder::detect_tool_call(output, calls);
            TEST("XML with surrounding text detected", detected);
            TEST("Name = send_email", calls[0].function_name == "send_email");
        }

        // Normal text (no tool call)
        {
            std::string output = "Hello, how can I help you today?";
            std::vector<forge::ToolCall> calls;
            bool detected = forge::ToolPromptBuilder::detect_tool_call(output, calls);
            TEST("Normal text not detected", !detected);
            TEST("No calls for text", calls.empty());
        }

        // Empty output
        {
            std::string output = "";
            std::vector<forge::ToolCall> calls;
            bool detected = forge::ToolPromptBuilder::detect_tool_call(output, calls);
            TEST("Empty output not detected", !detected);
        }

        // Multiple XML tool calls
        {
            std::string output = R"(<function name="search_web"><param name="query">weather</param></function><function name="get_weather"><param name="location">Berlin</param></function>)";
            std::vector<forge::ToolCall> calls;
            bool detected = forge::ToolPromptBuilder::detect_tool_call(output, calls);
            TEST("Multiple XML calls detected", detected);
            TEST("First call name = search_web", calls.size() >= 1 && calls[0].function_name == "search_web");
            TEST("Second call name = get_weather", calls.size() >= 2 && calls[1].function_name == "get_weather");
        }

        // XML with multiple params
        {
            std::string output = R"(<function name="calculate"><param name="a">10</param><param name="b">20</param><param name="op">add</param></function>)";
            std::vector<forge::ToolCall> calls;
            bool detected = forge::ToolPromptBuilder::detect_tool_call(output, calls);
            TEST("Multiple params detected", detected);
            TEST("Args contain a", calls[0].function_args.find("a") != std::string::npos);
            TEST("Args contain b", calls[0].function_args.find("b") != std::string::npos);
            TEST("Args contain op", calls[0].function_args.find("op") != std::string::npos);
        }

        // JSON input (should NOT be detected by XML parser)
        {
            std::string output = R"({"temperature": 72, "condition": "sunny"})";
            std::vector<forge::ToolCall> calls;
            bool detected = forge::ToolPromptBuilder::detect_tool_call(output, calls);
            TEST("Plain JSON not detected (no <function> tag)", !detected);
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
    // 5. Full pipeline: build tool prompt + detect
    // ============================================================
    std::cout << "--- Full Pipeline ---\n";
    {
        std::vector<forge::ToolDef> tools;
        forge::ToolDef td;
        td.name = "get_weather";
        td.description = "Get weather for a location";
        td.parameters_json = R"({"type":"object","properties":{"location":{"type":"string"}},"required":["location"]})";
        tools.push_back(td);

        std::string prompt = forge::ToolPromptBuilder::build("You are a helpful assistant.", tools);
        TEST("Pipeline prompt built", !prompt.empty());
        TEST("Prompt has <tools>", prompt.find("<tools>") != std::string::npos);
        TEST("Prompt has <calls>", prompt.find("<calls>") != std::string::npos);
        TEST("Prompt has tool name", prompt.find("get_weather") != std::string::npos);

        // Detection round-trip
        std::string mock_output = R"(<function name="get_weather"><param name="location">London</param></function>)";
        std::vector<forge::ToolCall> calls;
        bool detected = forge::ToolPromptBuilder::detect_tool_call(mock_output, calls);
        TEST("Round-trip: XML detected", detected);
        TEST("Round-trip: correct function", calls.size() > 0 && calls[0].function_name == "get_weather");
        TEST("Round-trip: has args", calls.size() > 0 && calls[0].function_args.find("London") != std::string::npos);
    }

    // ============================================================
    // 6. Engine integration with tool prompt
    // ============================================================
    std::cout << "--- Engine Integration ---\n";
    {
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
            TEST("Engine generates tokens", result.tokens.size() > tokens.size());
        }
        std::remove(model_path.c_str());
    }

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
