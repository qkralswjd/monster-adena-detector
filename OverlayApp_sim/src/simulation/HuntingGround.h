#pragma once
// HuntingGround.h
// ===============
// 사냥터 좌표 데이터 구조 + JSON 로드/저장.
// 실제 게임 입력 없이 시뮬레이션 엔진에서만 사용.

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

// ── 웨이포인트 ───────────────────────────────────────────────────────────────
struct Waypoint {
    std::string name;
    float       x = 0.0f;   // 게임 화면 픽셀 or 정규화 좌표 (사용자 정의)
    float       y = 0.0f;
};

// ── 사냥터 ───────────────────────────────────────────────────────────────────
struct HuntingGround {
    std::string              name;
    std::vector<Waypoint>    points;
    std::vector<std::vector<std::string>> routes;  // 웨이포인트 이름 순서 목록

    // 이름으로 웨이포인트 찾기
    const Waypoint* FindPoint(const std::string& n) const {
        for (auto& p : points)
            if (p.name == n) return &p;
        return nullptr;
    }

    // 웨이포인트 추가
    void AddPoint(const std::string& n, float x, float y) {
        RemovePoint(n);   // 중복 방지
        points.push_back({n, x, y});
    }

    // 웨이포인트 삭제
    void RemovePoint(const std::string& n) {
        points.erase(std::remove_if(points.begin(), points.end(),
            [&](const Waypoint& p){ return p.name == n; }), points.end());
    }

    // 루트 추가
    void AddRoute(const std::vector<std::string>& route) {
        routes.push_back(route);
    }

    // JSON 직렬화 (수동 구현, 외부 라이브러리 없음)
    std::string ToJson() const {
        std::ostringstream ss;
        ss << "{\n";
        ss << "  \"name\": \"" << EscapeJson(name) << "\",\n";
        ss << "  \"points\": [\n";
        for (size_t i = 0; i < points.size(); i++) {
            const auto& p = points[i];
            ss << "    {\"name\": \"" << EscapeJson(p.name) << "\""
               << ", \"x\": " << p.x
               << ", \"y\": " << p.y << "}";
            ss << (i + 1 < points.size() ? "," : "") << "\n";
        }
        ss << "  ],\n";
        ss << "  \"routes\": [\n";
        for (size_t i = 0; i < routes.size(); i++) {
            ss << "    [";
            for (size_t j = 0; j < routes[i].size(); j++) {
                ss << "\"" << EscapeJson(routes[i][j]) << "\"";
                ss << (j + 1 < routes[i].size() ? ", " : "");
            }
            ss << "]" << (i + 1 < routes.size() ? "," : "") << "\n";
        }
        ss << "  ]\n";
        ss << "}\n";
        return ss.str();
    }

    // JSON 저장
    bool SaveToFile(const std::string& path) const {
        std::ofstream f(path);
        if (!f.is_open()) return false;
        f << ToJson();
        return true;
    }

    // JSON 로드 (간단한 수동 파서)
    static HuntingGround LoadFromFile(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open())
            throw std::runtime_error("Cannot open: " + path);
        std::string json((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
        return ParseJson(json);
    }

private:
    static std::string EscapeJson(const std::string& s) {
        std::string out;
        for (char c : s) {
            if (c == '"')  out += "\\\"";
            else if (c == '\\') out += "\\\\";
            else out += c;
        }
        return out;
    }

    // 간단한 JSON 파서 (points/routes 필드만 추출)
    static HuntingGround ParseJson(const std::string& json) {
        HuntingGround hg;

        // name
        hg.name = ExtractStringField(json, "name");

        // points: {"name":"P1","x":100,"y":200}
        auto pts = ExtractArrayContent(json, "points");
        for (auto& obj : SplitObjects(pts)) {
            Waypoint wp;
            wp.name = ExtractStringField(obj, "name");
            wp.x    = ExtractFloatField(obj, "x");
            wp.y    = ExtractFloatField(obj, "y");
            if (!wp.name.empty())
                hg.points.push_back(wp);
        }

        // routes: [["P1","P2"],...]
        auto rts = ExtractArrayContent(json, "routes");
        // 각 루트는 [ "P1", "P2", ... ] 형태
        size_t pos = 0;
        while (pos < rts.size()) {
            size_t lb = rts.find('[', pos);
            size_t rb = rts.find(']', lb == std::string::npos ? 0 : lb);
            if (lb == std::string::npos || rb == std::string::npos) break;
            std::string routeStr = rts.substr(lb + 1, rb - lb - 1);
            auto names = ExtractStringsFromArray(routeStr);
            if (!names.empty())
                hg.routes.push_back(names);
            pos = rb + 1;
        }

        return hg;
    }

    static std::string ExtractStringField(const std::string& json, const std::string& key) {
        std::string search = "\"" + key + "\"";
        size_t pos = json.find(search);
        if (pos == std::string::npos) return "";
        pos = json.find(':', pos + search.size());
        if (pos == std::string::npos) return "";
        pos = json.find('"', pos);
        if (pos == std::string::npos) return "";
        size_t end = json.find('"', pos + 1);
        if (end == std::string::npos) return "";
        return json.substr(pos + 1, end - pos - 1);
    }

    static float ExtractFloatField(const std::string& json, const std::string& key) {
        std::string search = "\"" + key + "\"";
        size_t pos = json.find(search);
        if (pos == std::string::npos) return 0.0f;
        pos = json.find(':', pos + search.size());
        if (pos == std::string::npos) return 0.0f;
        pos++;
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
        try { return std::stof(json.substr(pos)); } catch (...) { return 0.0f; }
    }

    static std::string ExtractArrayContent(const std::string& json, const std::string& key) {
        std::string search = "\"" + key + "\"";
        size_t pos = json.find(search);
        if (pos == std::string::npos) return "";
        pos = json.find('[', pos + search.size());
        if (pos == std::string::npos) return "";
        int depth = 0;
        size_t start = pos;
        for (size_t i = pos; i < json.size(); i++) {
            if (json[i] == '[') depth++;
            else if (json[i] == ']') { depth--; if (depth == 0) return json.substr(start + 1, i - start - 1); }
        }
        return "";
    }

    static std::vector<std::string> SplitObjects(const std::string& s) {
        std::vector<std::string> out;
        int depth = 0; size_t start = 0;
        for (size_t i = 0; i < s.size(); i++) {
            if (s[i] == '{') { if (depth == 0) start = i; depth++; }
            else if (s[i] == '}') { depth--; if (depth == 0) out.push_back(s.substr(start, i - start + 1)); }
        }
        return out;
    }

    static std::vector<std::string> ExtractStringsFromArray(const std::string& s) {
        std::vector<std::string> out;
        size_t pos = 0;
        while (pos < s.size()) {
            size_t q1 = s.find('"', pos);
            if (q1 == std::string::npos) break;
            size_t q2 = s.find('"', q1 + 1);
            if (q2 == std::string::npos) break;
            out.push_back(s.substr(q1 + 1, q2 - q1 - 1));
            pos = q2 + 1;
        }
        return out;
    }
};
