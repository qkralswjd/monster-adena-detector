#pragma once
// MockPicoHID.h
// =============
// 실제 Pico HID 장치 대신 모든 입력 명령을 이벤트 큐와 로그에만 기록.
// 실제 마우스/키보드/HID 장치에 절대 전달하지 않음.
// 기존 PicoHidDevice.h / HardwareInputManager.h 와 완전 독립.

#include <string>
#include <vector>
#include <deque>
#include <functional>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <mutex>

// ── 이벤트 타입 ──────────────────────────────────────────────────────────────
enum class MockHidEventType {
    MouseMove,
    MouseClick,
    MouseDrag,
    KeyDown,
    KeyUp,
    Delay,
};

inline const char* MockHidEventTypeName(MockHidEventType t) {
    switch (t) {
    case MockHidEventType::MouseMove:  return "MouseMove";
    case MockHidEventType::MouseClick: return "MouseClick";
    case MockHidEventType::MouseDrag:  return "MouseDrag";
    case MockHidEventType::KeyDown:    return "KeyDown";
    case MockHidEventType::KeyUp:      return "KeyUp";
    case MockHidEventType::Delay:      return "Delay";
    default:                           return "Unknown";
    }
}

// ── 단일 Mock 이벤트 ─────────────────────────────────────────────────────────
struct MockHidEvent {
    MockHidEventType type;
    double           timestampMs  = 0.0;  // 앱 시작 기준 ms
    int              x            = 0;    // 마우스 좌표 (픽셀)
    int              y            = 0;
    int              toX          = 0;    // drag 종료점
    int              toY          = 0;
    int              button       = 0;    // 0=left,1=right,2=middle
    int              keyCode      = 0;    // 가상 키코드
    double           durationMs   = 0.0;  // drag 소요 ms / delay ms
    std::string      tag;                 // 호출자 식별 태그
};

// ── MockPicoHID 클래스 ───────────────────────────────────────────────────────
class MockPicoHID {
public:
    // 싱글턴 (시뮬레이션 전체에서 하나의 로그 큐 공유)
    static MockPicoHID& Instance() {
        static MockPicoHID inst;
        return inst;
    }

    // 명령 기록 API (실제 장치에 절대 전달하지 않음)
    void Move(int x, int y, const std::string& tag = "") {
        Enqueue(MockHidEventType::MouseMove, x, y, 0, 0, 0, 0, 0.0, tag);
    }

    void Click(int x, int y, int button = 0, const std::string& tag = "") {
        Enqueue(MockHidEventType::MouseClick, x, y, 0, 0, button, 0, 0.0, tag);
    }

    void Drag(int fromX, int fromY, int toX, int toY,
              double durationMs = 200.0, const std::string& tag = "") {
        Enqueue(MockHidEventType::MouseDrag, fromX, fromY, toX, toY, 0, 0, durationMs, tag);
    }

    void KeyDown(int keyCode, const std::string& tag = "") {
        Enqueue(MockHidEventType::KeyDown, 0, 0, 0, 0, 0, keyCode, 0.0, tag);
    }

    void KeyUp(int keyCode, const std::string& tag = "") {
        Enqueue(MockHidEventType::KeyUp, 0, 0, 0, 0, 0, keyCode, 0.0, tag);
    }

    void Delay(double ms, const std::string& tag = "") {
        Enqueue(MockHidEventType::Delay, 0, 0, 0, 0, 0, 0, ms, tag);
    }

    // 큐 접근
    std::vector<MockHidEvent> DrainEvents() {
        std::lock_guard<std::mutex> lk(m_mutex);
        std::vector<MockHidEvent> out(m_queue.begin(), m_queue.end());
        m_queue.clear();
        return out;
    }

    const std::vector<MockHidEvent>& GetHistory() const { return m_history; }
    void ClearHistory() { m_history.clear(); }

    // 콜백 (이벤트 발생 시 즉시 호출 - 로그 출력 등)
    void SetOnEvent(std::function<void(const MockHidEvent&)> cb) { m_onEvent = cb; }

    // JSON 직렬화
    std::string HistoryToJson() const {
        std::ostringstream ss;
        ss << "[\n";
        for (size_t i = 0; i < m_history.size(); i++) {
            const auto& e = m_history[i];
            ss << "  {\n";
            ss << "    \"type\": \""       << MockHidEventTypeName(e.type) << "\",\n";
            ss << "    \"timestampMs\": "  << std::fixed << std::setprecision(2) << e.timestampMs << ",\n";
            ss << "    \"x\": "            << e.x << ",\n";
            ss << "    \"y\": "            << e.y << ",\n";
            ss << "    \"toX\": "          << e.toX << ",\n";
            ss << "    \"toY\": "          << e.toY << ",\n";
            ss << "    \"button\": "       << e.button << ",\n";
            ss << "    \"keyCode\": "      << e.keyCode << ",\n";
            ss << "    \"durationMs\": "   << e.durationMs << ",\n";
            ss << "    \"tag\": \""        << e.tag << "\"\n";
            ss << "  }" << (i + 1 < m_history.size() ? "," : "") << "\n";
        }
        ss << "]\n";
        return ss.str();
    }

private:
    MockPicoHID() {
        m_startTime = std::chrono::steady_clock::now();
    }

    double NowMs() const {
        auto d = std::chrono::steady_clock::now() - m_startTime;
        return std::chrono::duration<double, std::milli>(d).count();
    }

    void Enqueue(MockHidEventType type,
                 int x, int y, int toX, int toY,
                 int button, int keyCode,
                 double durationMs, const std::string& tag)
    {
        MockHidEvent ev;
        ev.type       = type;
        ev.timestampMs= NowMs();
        ev.x          = x;  ev.y    = y;
        ev.toX        = toX; ev.toY = toY;
        ev.button     = button;
        ev.keyCode    = keyCode;
        ev.durationMs = durationMs;
        ev.tag        = tag;

        {
            std::lock_guard<std::mutex> lk(m_mutex);
            m_queue.push_back(ev);
            m_history.push_back(ev);
        }
        if (m_onEvent) m_onEvent(ev);
    }

    std::chrono::steady_clock::time_point m_startTime;
    std::deque<MockHidEvent>   m_queue;
    std::vector<MockHidEvent>  m_history;
    std::function<void(const MockHidEvent&)> m_onEvent;
    std::mutex m_mutex;
};
