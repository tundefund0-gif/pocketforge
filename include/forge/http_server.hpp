#ifndef FORGE_HTTP_SERVER_HPP
#define FORGE_HTTP_SERVER_HPP

#include <string>
#include <functional>
#include <map>
#include <vector>
#include <cstdint>

namespace forge {

// ============================================================
//  Minimal HTTP/1.1 server (no external dependencies)
// ============================================================
//
// Listens on TCP, parses JSON requests, dispatches to handler.
// Designed for ARM32/ARM64, single-threaded, poll-based.
//

struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
    std::map<std::string, std::string> headers;
};

struct HttpResponse {
    int status_code = 200;
    std::string content_type = "application/json";
    std::string body;
};

// Minimal JSON parser (only what we need)
struct JsonValue {
    enum Type { NULL_VAL, BOOL, NUMBER, STRING, ARRAY, OBJECT } type = NULL_VAL;
    bool bool_val = false;
    double number_val = 0.0;
    std::string string_val;
    std::vector<JsonValue> array_val;
    std::map<std::string, JsonValue> object_val;

    static JsonValue parse(const std::string& json);
    std::string serialize() const;
    JsonValue get(const std::string& key) const;
    bool has(const std::string& key) const;
};

std::string json_escape(const std::string& s);
using HttpHandler = std::function<HttpResponse(const HttpRequest&)>;

class HttpServer {
public:
    HttpServer(uint16_t port = 8080);
    ~HttpServer();

    void route(const std::string& method, const std::string& path, HttpHandler handler);
    bool start();
    void stop();
    void run();  // blocks until stop

private:
    uint16_t port_;
    int server_fd_ = -1;
    bool running_ = false;
    std::vector<std::tuple<std::string, std::string, HttpHandler>> routes_;

    void handle_client(int client_fd);
    HttpRequest parse_request(const std::string& raw);
    std::string build_response(const HttpResponse& resp);
};

} // namespace forge
#endif // FORGE_HTTP_SERVER_HPP
