"""recorder.py
─────────────────────────────────────────────────────────────────────────
테스트 세션 기록 모듈.

--record 플래그 없이 실행하면 이 모듈의 모든 기능은 완전히 비활성화(no-op)됩니다.
기존 자동사냥 로직, 상태머신, 전투 판단에는 일절 영향 없음.

사용 방법:
    # main.py 시작 시
    from recorder import init_recorder, get_recorder

    if record_mode:
        init_recorder()         # 세션 폴더 생성, 녹화·로그·이벤트 시작

    # 이벤트 기록 (record_mode=False일 때는 완전 no-op)
    get_recorder().log_event("COMBAT_START", {"cx": 300, "cy": 400})

    # main.py 종료 시 (finally)
    get_recorder().stop()

생성 파일:
    test_runs/<YYYY-MM-DD_HH-MM-SS>/
        screen.mp4    — 게임 화면 전체 녹화 (mss + cv2.VideoWriter, mp4v)
        console.log   — stdout Tee 저장 (터미널도 동시 출력)
        events.json   — 주요 이벤트 구조화 기록
"""

from __future__ import annotations

import io
import json
import os
import queue
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional


# ── 타임스탬프 헬퍼 ───────────────────────────────────────────────────────────

def _ts_log() -> str:
    """콘솔 로그용 타임스탬프: [YYYY-MM-DD HH:MM:SS.mmm]"""
    now = datetime.now()
    return now.strftime("[%Y-%m-%d %H:%M:%S.") + f"{now.microsecond // 1000:03d}]"


def _ts_iso() -> str:
    """events.json용 ISO 타임스탬프: YYYY-MM-DDThh:mm:ss.mmm"""
    now = datetime.now()
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}"


# ── TeeStream: stdout → 터미널 + 파일 동시 출력 ──────────────────────────────

class TeeStream(io.TextIOBase):
    """sys.stdout / sys.stderr를 교체해 터미널과 파일에 동시 출력하는 래퍼.

    --record 없이 사용하지 않음.
    thread-safe write: 파일 I/O 실패는 조용히 무시 (자동사냥 방해 방지).

    줄 단위 timestamp prefix 기능:
        add_timestamp=True 이면 새 줄이 시작되는 시점에
        "[YYYY-MM-DD HH:MM:SS.mmm] " prefix를 console.log 파일에만 삽입.
        터미널 출력에는 prefix 없이 원본 그대로 출력해 가독성 유지.
    """

    def __init__(
        self,
        original: io.TextIOBase,
        log_file: io.TextIOBase,
        add_timestamp: bool = False,
    ) -> None:
        super().__init__()
        self._orig          = original
        self._file          = log_file
        self._add_ts        = add_timestamp
        self._lock          = threading.Lock()
        self._at_line_start = True   # 다음 write가 새 줄 시작인지 추적

    # ── TextIOBase 인터페이스 ─────────────────────────────────────────

    @property
    def encoding(self):
        return getattr(self._orig, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._orig, "errors", "replace")

    def write(self, s: str) -> int:
        if not s:
            return 0
        with self._lock:
            # 터미널: 원본 그대로 출력 (timestamp prefix 없음)
            try:
                n = self._orig.write(s)
                self._orig.flush()
            except Exception:
                n = len(s)

            # 파일: timestamp prefix 삽입 후 기록
            try:
                if self._add_ts:
                    self._file.write(self._prefix_lines(s))
                else:
                    self._file.write(s)
                self._file.flush()
            except Exception:
                pass

        return n

    def _prefix_lines(self, s: str) -> str:
        """새 줄 시작마다 [YYYY-MM-DD HH:MM:SS.mmm] prefix 삽입 (파일 전용)."""
        result = []
        for ch in s:
            if self._at_line_start and ch != '\n':
                result.append(_ts_log() + ' ')
                self._at_line_start = False
            result.append(ch)
            if ch == '\n':
                self._at_line_start = True
        return ''.join(result)

    def flush(self) -> None:
        try:
            self._orig.flush()
        except Exception:
            pass
        try:
            self._file.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        try:
            return self._orig.isatty()
        except Exception:
            return False

    def fileno(self) -> int:
        # 일부 라이브러리가 fileno()를 호출할 수 있으므로 원본에 위임
        return self._orig.fileno()


# ── ScreenRecorder: mss + cv2.VideoWriter 별도 스레드 녹화 ───────────────────

class ScreenRecorder:
    """게임 화면을 별도 스레드에서 mp4v로 녹화.

    메인 루프(YOLO/Pico)를 전혀 차단하지 않도록 생산자-소비자 큐 패턴 사용:
      캡처 스레드  → frames_q → 인코더 스레드
    녹화 실패(cv2 없음, 화면 없음 등)는 예외 없이 처리해 자동사냥 유지.
    """

    # 메모리 초과 방지용 큐 최대 크기 (프레임 단위)
    _QUEUE_MAXSIZE = 60

    def __init__(
        self,
        output_path: str,
        monitor_index: int = 1,
        fps: float = 10.0,
    ) -> None:
        """
        Args:
            output_path  : 저장할 .mp4 경로
            monitor_index: mss 모니터 번호 (1 = 기본 모니터)
            fps          : 녹화 프레임레이트 (기본 10 fps — 메인 루프 부하 최소화)
        """
        self._output_path  = output_path
        self._monitor_idx  = monitor_index
        self._fps          = fps
        self._active       = False
        self._frames_q: queue.Queue = queue.Queue(maxsize=self._QUEUE_MAXSIZE)

        self._capture_thread:  Optional[threading.Thread] = None
        self._encoder_thread:  Optional[threading.Thread] = None

        # 녹화 가능 여부 (cv2/mss import 실패 시 False)
        self._available = False

    def start(self) -> bool:
        """녹화 시작. 성공 여부 반환 (False여도 자동사냥 계속)."""
        try:
            import cv2
            import mss
            import numpy as np
        except ImportError as e:
            print(f"[REC] 화면 녹화 패키지 없음 ({e}) → 녹화 비활성화")
            return False

        # 화면 크기 사전 확인
        try:
            with mss.mss() as sct:
                mon = sct.monitors[self._monitor_idx]
                w, h = mon["width"], mon["height"]
        except Exception as e:
            print(f"[REC] 모니터 정보 취득 실패 ({e}) → 녹화 비활성화")
            return False

        # VideoWriter 초기화
        try:
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self._output_path, fourcc, self._fps, (w, h))
            if not self._writer.isOpened():
                print(f"[REC] VideoWriter 열기 실패 → 녹화 비활성화")
                return False
        except Exception as e:
            print(f"[REC] VideoWriter 초기화 실패 ({e}) → 녹화 비활성화")
            return False

        self._available = True
        self._active    = True
        self._w         = w
        self._h         = h

        # 캡처 스레드 (프레임 생산)
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="rec-capture",
            daemon=True,
        )
        self._capture_thread.start()

        # 인코더 스레드 (프레임 소비)
        self._encoder_thread = threading.Thread(
            target=self._encode_loop,
            name="rec-encode",
            daemon=True,
        )
        self._encoder_thread.start()

        print(f"[REC] 화면 녹화 시작: {w}x{h}  {self._fps:.0f}fps → {self._output_path}")
        return True

    def _capture_loop(self) -> None:
        """mss로 화면 캡처 → 큐에 삽입 (별도 스레드)."""
        import mss
        import numpy as np

        interval = 1.0 / self._fps
        try:
            with mss.mss() as sct:
                mon = sct.monitors[self._monitor_idx]
                while self._active:
                    t0 = time.perf_counter()
                    try:
                        raw = sct.grab(mon)
                        # BGRA → BGR (cv2 형식)
                        frame = np.array(raw)[:, :, :3]
                        # 큐가 가득 찬 경우 가장 오래된 프레임 버리고 삽입
                        if self._frames_q.full():
                            try:
                                self._frames_q.get_nowait()
                            except queue.Empty:
                                pass
                        self._frames_q.put_nowait(frame)
                    except Exception:
                        pass  # 캡처 실패 — 조용히 무시
                    elapsed = time.perf_counter() - t0
                    sleep_t = interval - elapsed
                    if sleep_t > 0:
                        time.sleep(sleep_t)
        except Exception as e:
            print(f"[REC] 캡처 스레드 예외 ({e})")
        finally:
            # 종료 신호: None 삽입
            try:
                self._frames_q.put(None, timeout=2.0)
            except queue.Full:
                pass

    def _encode_loop(self) -> None:
        """큐에서 프레임 꺼내 VideoWriter에 기록 (별도 스레드)."""
        import cv2

        try:
            while True:
                try:
                    frame = self._frames_q.get(timeout=2.0)
                except queue.Empty:
                    # _active=False 이면 종료, 아니면 대기 계속
                    if not self._active:
                        break
                    continue

                if frame is None:
                    break  # 종료 신호

                try:
                    self._writer.write(frame)
                except Exception:
                    pass
        except Exception as e:
            print(f"[REC] 인코더 스레드 예외 ({e})")
        finally:
            try:
                self._writer.release()
            except Exception:
                pass
            print("[REC] 녹화 종료")

    def stop(self) -> None:
        """녹화 정지 및 파일 플러시."""
        if not self._available:
            return
        self._active = False

        # 캡처 스레드 종료 대기 (최대 3초)
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=3.0)
            if self._capture_thread.is_alive():
                # 3초 내 미종료: daemon이므로 프로세스 종료 시 강제 종료됨
                # None 삽입 시도로 encoder가 종료 신호를 받도록 보조
                print("[REC] 캡처 스레드 3초 내 미종료 (daemon — 프로세스 종료 시 강제 종료)")
                try:
                    self._frames_q.put_nowait(None)
                except Exception:
                    pass

        # 인코더 스레드 종료 대기 (최대 5초)
        # encoder_loop finally에서 writer.release() 호출됨.
        # timeout 초과 시 daemon 강제 종료로 release() 미보장 → 직접 release 시도.
        if self._encoder_thread and self._encoder_thread.is_alive():
            self._encoder_thread.join(timeout=5.0)
            if self._encoder_thread.is_alive():
                print("[REC] 인코더 스레드 5초 내 미종료 → writer.release() 직접 시도")
                try:
                    self._writer.release()
                except Exception:
                    pass


# ── SessionRecorder: 세션 전체 관리 ─────────────────────────────────────────

class SessionRecorder:
    """테스트 세션 기록 총괄.

    - test_runs/<세션폴더>/ 생성
    - ScreenRecorder 시작/정지
    - sys.stdout → TeeStream 교체/복원
    - events.json 기록
    """

    def __init__(
        self,
        base_dir: str = "test_runs",
        monitor_index: int = 1,
        record_fps: float = 10.0,
    ) -> None:
        # 세션 폴더 경로 결정
        session_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._session_dir = os.path.join(base_dir, session_name)
        os.makedirs(self._session_dir, exist_ok=True)

        self._screen_path  = os.path.join(self._session_dir, "screen.mp4")
        self._console_path = os.path.join(self._session_dir, "console.log")
        self._events_path  = os.path.join(self._session_dir, "events.json")

        self._monitor_index = monitor_index
        self._record_fps    = record_fps

        # 이벤트 리스트 (메모리 → 종료 시 파일 기록)
        self._events: List[Dict[str, Any]] = []
        self._events_lock = threading.Lock()

        # 내부 객체
        self._screen_rec:  Optional[ScreenRecorder] = None
        self._tee_out:     Optional[TeeStream]       = None   # stdout Tee
        self._tee_err:     Optional[TeeStream]       = None   # stderr Tee
        self._log_file:    Optional[io.TextIOBase]   = None
        self._orig_stdout  = None
        self._orig_stderr  = None

        self._stopped = False

    # ── 공개 API ─────────────────────────────────────────────────────

    def start(self) -> None:
        """세션 기록 시작."""
        print(f"[REC] 테스트 세션 시작: {self._session_dir}")

        # console.log Tee 설치
        self._install_tee()

        # 화면 녹화 시작
        self._screen_rec = ScreenRecorder(
            output_path   = self._screen_path,
            monitor_index = self._monitor_index,
            fps           = self._record_fps,
        )
        self._screen_rec.start()

    def log_event(self, event: str, data: Optional[Dict[str, Any]] = None) -> None:
        """이벤트를 events.json에 기록.

        Args:
            event : 이벤트 이름 (예: "COMBAT_START")
            data  : 추가 데이터 dict (없으면 None)
        """
        entry: Dict[str, Any] = {
            "timestamp": _ts_iso(),
            "event":     event,
        }
        if data:
            entry["data"] = data

        with self._events_lock:
            self._events.append(entry)

    def stop(self) -> None:
        """세션 기록 종료. Ctrl+C / 정상종료 / 예외 모두 안전."""
        if self._stopped:
            return
        self._stopped = True

        # 화면 녹화 종료
        if self._screen_rec:
            self._screen_rec.stop()

        # Tee 제거 (원본 stdout 복원)
        self._remove_tee()

        # events.json 저장
        self._flush_events()

        print(f"[REC] 테스트 세션 종료: {self._session_dir}")

    # ── 내부 헬퍼 ────────────────────────────────────────────────────

    def _install_tee(self) -> None:
        """sys.stdout + sys.stderr → TeeStream으로 교체.

        stdout: 줄 단위 timestamp prefix 포함 (파일에만)
        stderr: traceback 등 예외 출력도 파일에 기록 (timestamp prefix 포함)
        """
        try:
            self._log_file    = open(self._console_path, "w", encoding="utf-8", buffering=1)

            # stdout Tee (add_timestamp=True: 파일에 줄 단위 timestamp)
            self._orig_stdout = sys.stdout
            self._tee_out     = TeeStream(sys.stdout, self._log_file, add_timestamp=True)
            sys.stdout        = self._tee_out

            # stderr Tee (traceback, warning 등도 파일에 기록)
            self._orig_stderr = sys.stderr
            self._tee_err     = TeeStream(sys.stderr, self._log_file, add_timestamp=True)
            sys.stderr        = self._tee_err

            print(f"[REC] 콘솔 로그 저장 (stdout+stderr): {self._console_path}")
        except Exception as e:
            print(f"[REC] 콘솔 로그 설치 실패 ({e}) → 터미널 출력만 유지")
            # 부분 설치된 경우 롤백
            if self._orig_stdout is not None:
                sys.stdout = self._orig_stdout
            if self._orig_stderr is not None:
                sys.stderr = self._orig_stderr
            self._tee_out  = None
            self._tee_err  = None
            self._log_file = None

    def _remove_tee(self) -> None:
        """sys.stdout / sys.stderr 원본 복원 및 로그 파일 닫기."""
        if self._orig_stdout is not None:
            sys.stdout = self._orig_stdout
            self._orig_stdout = None
        if self._orig_stderr is not None:
            sys.stderr = self._orig_stderr
            self._orig_stderr = None
        if self._log_file is not None:
            try:
                self._log_file.flush()
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None
        print("[REC] 로그 저장 완료")

    def _flush_events(self) -> None:
        """이벤트 리스트를 events.json으로 저장."""
        try:
            with self._events_lock:
                data = list(self._events)
            with open(self._events_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[REC] 이벤트 저장 완료 ({len(data)}건): {self._events_path}")
        except Exception as e:
            print(f"[REC] 이벤트 저장 실패 ({e})")

    @property
    def session_dir(self) -> str:
        return self._session_dir


# ── 글로벌 싱글톤 ────────────────────────────────────────────────────────────
#
# --record 없이 실행하면 _recorder는 _NullRecorder 인스턴스.
# 모든 메서드가 no-op이므로 호출 측에서 guard 불필요.

class _NullRecorder:
    """--record 없을 때 사용하는 완전 no-op 레코더."""

    def log_event(self, event: str, data: Optional[Dict[str, Any]] = None) -> None:
        pass

    def stop(self) -> None:
        pass


_recorder: Optional[SessionRecorder | _NullRecorder] = _NullRecorder()


def init_recorder(
    base_dir:      str   = "test_runs",
    monitor_index: int   = 1,
    record_fps:    float = 10.0,
) -> SessionRecorder:
    """SessionRecorder를 초기화하고 글로벌 싱글톤에 등록.

    --record 플래그가 True일 때 main.py에서 한 번만 호출.
    반환된 객체를 직접 사용해도 되고 get_recorder()를 사용해도 됨.
    """
    global _recorder
    rec = SessionRecorder(
        base_dir      = base_dir,
        monitor_index = monitor_index,
        record_fps    = record_fps,
    )
    _recorder = rec
    rec.start()
    return rec


def get_recorder() -> SessionRecorder | _NullRecorder:
    """글로벌 싱글톤 레코더 반환.

    --record 없을 때는 _NullRecorder 반환 (no-op).
    어디서든 import 후 get_recorder().log_event(...) 호출 가능.
    """
    return _recorder
