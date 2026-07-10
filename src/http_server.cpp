#include "forge/http_server.hpp"
#include <iostream>
#include <sstream>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <poll.h>
#include <fcntl.h>
#include <cstdlib>

namespace forge {

// ============================================================
//  Minimal JSON parser
// ============================================================

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c;
        }
    }
    return out;
}

static JsonValue parse_json(const char*& p) {
    JsonValue val;
    while (*p && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++;

    if (!*p) return val;

    if (*p == '{') {
        val.type = JsonValue::OBJECT;
        p++; // skip {
        while (*p && *p != '}') {
            while (*p && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++;
            if (*p == '}') break;
            // Parse key
            if (*p == '"') {
                p++;
                std::string key;
                while (*p && *p != '"') { if (*p == '\\') { p++; if (*p) key += *p++; } else key += *p++; }
                if (*p == '"') p++;
                while (*p && (*p == ' ' || *p == ':')) p++;
                if (*p == ':') p++;
                while (*p && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++;
                val.object_val[key] = parse_json(p);
                while (*p && *p != ',' && *p != '}') p++;
                if (*p == ',') p++;
            } else break;
        }
        if (*p == '}') p++;
    }
    else if (*p == '[') {
        val.type = JsonValue::ARRAY;
        p++;
        while (*p && *p != ']') {
            val.array_val.push_back(parse_json(p));
            while (*p && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++;
            if (*p == ',') { p++; continue; }
            if (*p == ']') break;
        }
        if (*p == ']') p++;
    }
    else if (*p == '"') {
        val.type = JsonValue::STRING;
        p++;
        while (*p && *p != '"') { if (*p == '\\') { p++; if (*p) val.string_val += *p++; } else val.string_val += *p++; }
        if (*p == '"') p++;
    }
    else if (*p == 't' && strncmp(p, "true", 4) == 0) {
        val.type = JsonValue::BOOL; val.bool_val = true; p += 4;
    }
    else if (*p == 'f' && strncmp(p, "false", 5) == 0) {
        val.type = JsonValue::BOOL; val.bool_val = false; p += 5;
    }
    else if (*p == 'n' && strncmp(p, "null", 4) == 0) {
        val.type = JsonValue::NULL_VAL; p += 4;
    }
    else if (*p == '-' || (*p >= '0' && *p <= '9')) {
        val.type = JsonValue::NUMBER;
        char* end = nullptr;
        val.number_val = strtod(p, &end);
        if (end) p = end;
    }
    return val;
}

JsonValue JsonValue::parse(const std::string& json) {
    const char* p = json.c_str();
    return parse_json(p);
}

std::string JsonValue::serialize() const {
    switch (type) {
        case NULL_VAL: return "null";
        case BOOL: return bool_val ? "true" : "false";
        case NUMBER: {
            char buf[64];
            if (number_val == (int64_t)number_val)
                snprintf(buf, sizeof(buf), "%lld", (long long)number_val);
            else
                snprintf(buf, sizeof(buf), "%.6f", (double)number_val);
            return buf;
        }
        case STRING: return "\"" + json_escape(string_val) + "\"";
        case ARRAY: {
            std::string s = "[";
            for (size_t i = 0; i < array_val.size(); i++) {
                if (i > 0) s += ",";
                s += array_val[i].serialize();
            }
            return s + "]";
        }
        case OBJECT: {
            std::string s = "{";
            bool first = true;
            for (auto& [k, v] : object_val) {
                if (!first) s += ",";
                s += "\"" + json_escape(k) + "\":" + v.serialize();
                first = false;
            }
            return s + "}";
        }
    }
    return "null";
}

JsonValue JsonValue::get(const std::string& key) const {
    if (type == OBJECT) {
        auto it = object_val.find(key);
        if (it != object_val.end()) return it->second;
    }
    return JsonValue();
}

bool JsonValue::has(const std::string& key) const {
    return type == OBJECT && object_val.find(key) != object_val.end();
}

// ============================================================
//  HTTP Server
// ============================================================

HttpServer::HttpServer(uint16_t port) : port_(port) {}
HttpServer::~HttpServer() { stop(); }

void HttpServer::route(const std::string& method, const std::string& path, HttpHandler handler) {
    routes_.push_back({method, path, handler});
}

bool HttpServer::start() {
    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) { std::cerr << "socket() failed\n"; return false; }

    int opt = 1;
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port_);

    if (bind(server_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        std::cerr << "bind() failed on port " << port_ << "\n";
        close(server_fd_); server_fd_ = -1;
        return false;
    }

    if (listen(server_fd_, 5) < 0) {
        std::cerr << "listen() failed\n";
        close(server_fd_); server_fd_ = -1;
        return false;
    }

    running_ = true;
    std::cout << "Server listening on http://0.0.0.0:" << port_ << "\n";
    return true;
}

void HttpServer::stop() {
    running_ = false;
    if (server_fd_ >= 0) { close(server_fd_); server_fd_ = -1; }
}

void HttpServer::run() {
    if (!running_) start();

    while (running_) {
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);
        int client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &addr_len);

        if (client_fd >= 0) {
            handle_client(client_fd);
        } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
            // No pending connections, wait a bit
            struct pollfd pfd = {server_fd_, POLLIN, 0};
            poll(&pfd, 1, 100); // 100ms timeout
        } else if (errno != EINTR) {
            break;
        }
    }
}

void HttpServer::handle_client(int client_fd) {
    // Read request (up to 64KB)
    std::string raw;
    char buf[4096];
    ssize_t n;
    
    // Read headers (handle partial reads)
    while (true) {
        n = read(client_fd, buf, sizeof(buf));
        if (n > 0) {
            raw.append(buf, n);
            if (raw.find("\r\n\r\n") != std::string::npos) break;
        } else if (n == 0) {
            break;
        } else if (errno == EINTR) {
            continue;
        } else {
            break;
        }
    }

    if (raw.empty()) { close(client_fd); return; }

    HttpRequest req = parse_request(raw);

    // Read body (if Content-Length specified)
    auto it = req.headers.find("content-length");
    if (it != req.headers.end()) {
        size_t content_length = std::stoul(it->second);
        size_t header_end = raw.find("\r\n\r\n") + 4;
        size_t body_read = raw.size() - header_end;
        while (body_read < content_length) {
            n = read(client_fd, buf, sizeof(buf));
            if (n <= 0) break;
            raw.append(buf, n);
            body_read += n;
        }
        req.body = raw.substr(header_end);
    }

    // Dispatch
    HttpResponse resp;
    bool matched = false;
    for (auto& [method, path, handler] : routes_) {
        if (req.method == method && req.path == path) {
            resp = handler(req);
            matched = true;
            break;
        }
    }
    if (!matched) {
        resp.status_code = 404;
        resp.body = "{\"error\":\"not_found\"}";
    }

    // Send response
    std::string response_str = build_response(resp);
    (void)write(client_fd, response_str.c_str(), response_str.size());
    close(client_fd);
}

HttpRequest HttpServer::parse_request(const std::string& raw) {
    HttpRequest req;
    std::istringstream ss(raw);
    std::string line;

    // Request line
    if (std::getline(ss, line)) {
        // Remove \r
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream line_ss(line);
        line_ss >> req.method >> req.path;
    }

    // Headers
    while (std::getline(ss, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) break;
        auto colon = line.find(':');
        if (colon != std::string::npos) {
            std::string key = line.substr(0, colon);
            std::string val = line.substr(colon + 1);
            // Trim
            if (!val.empty() && val[0] == ' ') val = val.substr(1);
            for (auto& c : key) c = tolower(c);
            req.headers[key] = val;
        }
    }

    return req;
}

std::string HttpServer::build_response(const HttpResponse& resp) {
    std::string status_str = "200 OK";
    if (resp.status_code == 404) status_str = "404 Not Found";
    else if (resp.status_code == 400) status_str = "400 Bad Request";
    else if (resp.status_code == 500) status_str = "500 Internal Server Error";

    std::string header = "HTTP/1.1 " + status_str + "\r\n"
                         "Content-Type: " + resp.content_type + "\r\n"
                         "Content-Length: " + std::to_string(resp.body.size()) + "\r\n"
                         "Access-Control-Allow-Origin: *\r\n"
                         "Connection: close\r\n"
                         "\r\n";
    return header + resp.body;
}

} // namespace forge
