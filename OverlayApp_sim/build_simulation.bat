@echo off
REM ============================================================
REM  build_simulation.bat
REM  SimulationTest.exe 빌드 배치 파일
REM
REM  사전 요구사항:
REM    - CMake 3.20 이상 (PATH에 등록)
REM    - Visual Studio 2022 (MSVC x64) 또는 Visual Studio 2019
REM    - external/onnxruntime/ (또는 onnxruntime_dml/) 세팅 완료
REM
REM  사용법:
REM    build_simulation.bat            → Release x64 빌드
REM    build_simulation.bat Debug      → Debug x64 빌드
REM    build_simulation.bat clean      → 빌드 폴더 삭제
REM
REM  빌드 결과:
REM    build\Release\SimulationTest.exe  (또는 build\Debug\...)
REM
REM  실행 예:
REM    build\Release\SimulationTest.exe --log 2
REM    build\Release\SimulationTest.exe --ground config\hunting_ground.json
REM    build\Release\SimulationTest.exe --image dataset\sample.png --out result.json
REM ============================================================

setlocal EnableDelayedExpansion

cd /d "%~dp0"

REM ── 빌드 타입 파싱 ──────────────────────────────────────────
set BUILD_TYPE=Release
if /I "%~1"=="Debug"   set BUILD_TYPE=Debug
if /I "%~1"=="debug"   set BUILD_TYPE=Debug
if /I "%~1"=="clean"   goto :CLEAN
if /I "%~1"=="--clean" goto :CLEAN

echo.
echo ============================================================
echo   SimulationTest Build  [%BUILD_TYPE%]
echo ============================================================
echo.

REM ── CMake 존재 확인 ─────────────────────────────────────────
where cmake >nul 2>&1
if errorlevel 1 (
    echo [ERROR] CMake not found. Please install CMake 3.20+ and add to PATH.
    pause
    exit /b 1
)

REM ── 빌드 폴더 생성 ──────────────────────────────────────────
if not exist build mkdir build

REM ── CMake 구성 (첫 실행 또는 CMakeLists.txt 변경 시) ────────
echo [1/3] CMake configure...
cmake -B build -S . -A x64 ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DOVERLAY_USE_DIRECTML=ON
if errorlevel 1 (
    echo [ERROR] CMake configure 실패
    pause
    exit /b 1
)

REM ── SimulationTest 타겟만 빌드 ──────────────────────────────
echo.
echo [2/3] Building SimulationTest...
cmake --build build --config %BUILD_TYPE% --target SimulationTest -j4
if errorlevel 1 (
    echo [ERROR] SimulationTest 빌드 실패
    pause
    exit /b 1
)

REM ── 결과 확인 ───────────────────────────────────────────────
echo.
echo [3/3] Build result:
set EXE_PATH=build\%BUILD_TYPE%\SimulationTest.exe
if exist "%EXE_PATH%" (
    echo   [OK] %EXE_PATH%
    echo.
    echo ============================================================
    echo   빌드 성공!
    echo ============================================================
    echo.
    echo 실행 예시:
    echo   %EXE_PATH% --log 2
    echo   %EXE_PATH% --ground config\hunting_ground.json --log 2
    echo   %EXE_PATH% --image dataset\sample.png --out sim_result.json
    echo   %EXE_PATH% --model models\monster_best.onnx --classes config\classes_monster.txt
    echo.
) else (
    echo   [WARN] 실행 파일을 찾을 수 없음: %EXE_PATH%
    echo          빌드 로그를 확인하세요.
)

goto :END

REM ── 빌드 폴더 삭제 ──────────────────────────────────────────
:CLEAN
echo.
echo [Clean] build\ 폴더 삭제 중...
if exist build (
    rmdir /s /q build
    echo   [OK] build\ 삭제 완료
) else (
    echo   [OK] build\ 폴더 없음 (이미 clean)
)
echo.

:END
endlocal
