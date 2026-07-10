#include "forge/types.hpp"
#include "forge/http_server.hpp"
#include <iostream>
#include <sstream>

static int passed = 0, failed = 0;
#define TEST(name, expr) do { \
    if (!(expr)) { std::cerr << "  FAIL: " << name << "\n"; failed++; } \
    else { std::cout << "  PASS: " << name << "\n"; passed++; } \
} while(0)

int main() {
    std::cout << "=== Tool Calling Format Validation (MiniCPM5 XML) ===\n\n";

    // ============================================================
    // 1. Validate XML tool call response format -> OpenAI compat
    // ============================================================
    std::cout << "--- OpenAI Response Conversion ---\n";
    {
        forge::ToolCall call;
        call.id = "call_abc123";
        call.type = "function";
        call.function_name = "get_weather";
        call.function_args = R"({"location": "London"})";

        std::ostringstream json;
        json << "{\n"
             << "  \"id\":\"" << call.id << "\",\n"
             << "  \"type\":\"" << call.type << "\",\n"
             << "  \"function\":{\n"
             << "    \"name\":\"" << call.function_name << "\",\n"
             << "    \"arguments\":\"" << forge::json_escape(call.function_args) << "\"\n"
             << "  }\n"
             << "}";

        auto parsed = forge::JsonValue::parse(json.str());
        TEST("Response parses", parsed.type == forge::JsonValue::OBJECT);
        TEST("id matches", parsed.get("id").string_val == "call_abc123");
        TEST("type = function", parsed.get("type").string_val == "function");
        auto fn = parsed.get("function");
        TEST("function.name = get_weather", fn.get("name").string_val == "get_weather");
        TEST("function.arguments exists", fn.has("arguments"));
    }

    // ============================================================
    // 2. Full chat completions response structure
    // ============================================================
    std::cout << "\n--- Chat Completions Response ---\n";
    {
        forge::ToolCall call;
        call.id = "call_xyz789";
        call.type = "function";
        call.function_name = "search_web";
        call.function_args = R"({"query": "latest news"})";

        std::ostringstream json;
        json << "{\n"
             << "  \"id\":\"chatcmpl-123\",\n"
             << "  \"object\":\"chat.completion\",\n"
             << "  \"created\":1234567890,\n"
             << "  \"model\":\"pocketforge-1b\",\n"
             << "  \"choices\":[{\n"
             << "    \"index\":0,\n"
             << "    \"message\":{\n"
             << "      \"role\":\"assistant\",\n"
             << "      \"content\":null,\n"
             << "      \"tool_calls\":[\n"
             << "        {\n"
             << "          \"id\":\"" << call.id << "\",\n"
             << "          \"type\":\"" << call.type << "\",\n"
             << "          \"function\":{\n"
             << "            \"name\":\"" << call.function_name << "\",\n"
             << "            \"arguments\":\"" << forge::json_escape(call.function_args) << "\"\n"
             << "          }\n"
             << "        }\n"
             << "      ]\n"
             << "    },\n"
             << "    \"finish_reason\":\"tool_calls\"\n"
             << "  }],\n"
             << "  \"usage\":{\"prompt_tokens\":10,\"completion_tokens\":15,\"total_tokens\":25}\n"
             << "}";

        auto parsed = forge::JsonValue::parse(json.str());
        TEST("Full response parses", parsed.type == forge::JsonValue::OBJECT);
        TEST("object = chat.completion", parsed.get("object").string_val == "chat.completion");
        auto choices = parsed.get("choices");
        auto msg = choices.array_val[0].get("message");
        TEST("message.role = assistant", msg.get("role").string_val == "assistant");
        TEST("message.content is null", msg.get("content").type == forge::JsonValue::NULL_VAL);
        auto tc = msg.get("tool_calls").array_val[0];
        TEST("tool_call.id matches", tc.get("id").string_val == "call_xyz789");
        TEST("tool_call.type = function", tc.get("type").string_val == "function");
        auto func = tc.get("function");
        TEST("function.name = search_web", func.get("name").string_val == "search_web");
        TEST("function.arguments exists", func.has("arguments"));
        TEST("finish_reason = tool_calls", choices.array_val[0].get("finish_reason").string_val == "tool_calls");
    }

    // ============================================================
    // 3. XML tool call formats
    // ============================================================
    std::cout << "\n--- XML Tool Call Formats ---\n";
    // Format 1: Standard XML
    {
        std::string out = R"(<function name="get_weather"><param name="location">Paris</param></function>)";
        std::vector<forge::ToolCall> calls;
        bool ok = forge::ToolPromptBuilder::detect_tool_call(out, calls);
        TEST("Format 1: XML detected", ok);
        TEST("Format 1: name = get_weather", calls.size() > 0 && calls[0].function_name == "get_weather");
    }
    // Format 2: XML with surrounding text
    {
        std::string out = "I'll check.\n<function name=\"get_weather\"><param name=\"location\">Tokyo</param></function>\nOK";
        std::vector<forge::ToolCall> calls;
        bool ok = forge::ToolPromptBuilder::detect_tool_call(out, calls);
        TEST("Format 2: XML + text detected", ok);
        TEST("Format 2: name = get_weather", calls.size() > 0 && calls[0].function_name == "get_weather");
        TEST("Format 2: has Tokyo", calls.size() > 0 && calls[0].function_args.find("Tokyo") != std::string::npos);
    }
    // Format 3: Multiple XML calls
    {
        std::string out = R"(<function name="search_web"><param name="q">weather</param></function><function name="get_weather"><param name="loc">Berlin</param></function>)";
        std::vector<forge::ToolCall> calls;
        bool ok = forge::ToolPromptBuilder::detect_tool_call(out, calls);
        TEST("Format 3: multiple XML calls", ok);
        TEST("Format 3: at least 2 calls", calls.size() >= 2);
        TEST("Format 3: first = search_web", calls[0].function_name == "search_web");
        TEST("Format 3: second = get_weather", calls[1].function_name == "get_weather");
    }
    // Format 4: Multiple params
    {
        std::string out = R"(<function name="calc"><param name="a">10</param><param name="b">20</param><param name="op">add</param></function>)";
        std::vector<forge::ToolCall> calls;
        bool ok = forge::ToolPromptBuilder::detect_tool_call(out, calls);
        TEST("Format 4: multi-param XML", ok);
        TEST("Format 4: name = calc", calls.size() > 0 && calls[0].function_name == "calc");
        TEST("Format 4: has a", calls[0].function_args.find("a") != std::string::npos);
        TEST("Format 4: has b", calls[0].function_args.find("b") != std::string::npos);
        TEST("Format 4: has op", calls[0].function_args.find("op") != std::string::npos);
    }

    // ============================================================
    // 4. Tool definition prompt format
    // ============================================================
    std::cout << "\n--- Tool Definition Prompt Format ---\n";
    {
        std::vector<forge::ToolDef> tools;
        forge::ToolDef td;
        td.name = "get_weather";
        td.description = "Get current weather for a location";
        td.parameters_json = R"({"type":"object","properties":{"location":{"type":"string","description":"City name"}},"required":["location"]})";
        tools.push_back(td);

        std::string prompt = forge::ToolPromptBuilder::build("You are a helpful assistant.", tools);
        TEST("Prompt has tool name", prompt.find("get_weather") != std::string::npos);
        TEST("Prompt has description", prompt.find("Get current weather") != std::string::npos);
        TEST("Prompt has <tools> tag", prompt.find("<tools>") != std::string::npos);
        TEST("Prompt has <calls> tag", prompt.find("<calls>") != std::string::npos);

        // Round-trip
        std::string mock = R"(<function name="get_weather"><param name="location">London</param></function>)";
        std::vector<forge::ToolCall> calls;
        forge::ToolPromptBuilder::detect_tool_call(mock, calls);
        TEST("Round-trip: detected", calls.size() == 1);
        TEST("Round-trip: correct function", calls[0].function_name == "get_weather");
    }

    // ============================================================
    // 5. ToolChoice handling
    // ============================================================
    std::cout << "\n--- Tool Choice Handling ---\n";
    {
        std::string req = R"({"tool_choice": "auto"})";
        auto body = forge::JsonValue::parse(req);
        TEST("tool_choice = auto", body.get("tool_choice").string_val == "auto");
    }
    {
        std::string req = R"({"tool_choice": "none"})";
        auto body = forge::JsonValue::parse(req);
        TEST("tool_choice = none", body.get("tool_choice").string_val == "none");
    }
    {
        std::string req = R"({"tool_choice": {"type": "function", "function": {"name": "get_weather"}}})";
        auto body = forge::JsonValue::parse(req);
        auto tc = body.get("tool_choice");
        TEST("tool_choice with specific function", tc.type == forge::JsonValue::OBJECT);
        TEST("Specific function name", tc.get("function").get("name").string_val == "get_weather");
    }

    std::cout << "\n=== Results: " << passed << " passed, "
              << failed << " failed ===\n";
    return failed > 0 ? 1 : 0;
}
