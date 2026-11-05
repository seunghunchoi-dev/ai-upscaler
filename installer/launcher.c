/*
 * AI 영상·사진 업스케일러 실행 파일 (런처)
 * Copyright (c) 2026 최승훈. All rights reserved.
 *
 * 설치 폴더의 내장 Python(runtime)으로 컴파일된 프로그램(vup .pyd)을 실행한다.
 * 한글 경로(사용자 이름 등)에서도 동작하도록 모든 경로를 유니코드(W) API 로 다룬다.
 *   DEBUG 를 정의하고 빌드하면: 콘솔 python.exe 로 실행하고, 오류 시 창을 닫지 않고 기다린다.
 */
#include <windows.h>
#include <stdio.h>
#include <wchar.h>

#ifdef DEBUG
#define PYEXE L"runtime\\python.exe"
#else
#define PYEXE L"runtime\\pythonw.exe"
#endif

static void run(void) {
    wchar_t dir[MAX_PATH * 2], cmd[32768], exe[MAX_PATH * 2];
    GetModuleFileNameW(NULL, dir, MAX_PATH * 2);
    wchar_t *slash = wcsrchr(dir, L'\\');
    if (slash) *slash = 0;
    swprintf(exe, MAX_PATH * 2, L"%ls\\%ls", dir, PYEXE);

    /* 프로그램 인자(파일을 아이콘에 끌어다 놓은 경우 등)를 그대로 넘긴다 */
    int argc = 0;
    LPWSTR *argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    swprintf(cmd, 32768, L"\"%ls\" -X utf8 -c \"import vup.gui; vup.gui.main()\"", exe);
    for (int i = 1; i < argc; i++) {
        wcscat(cmd, L" \"");
        wcscat(cmd, argv[i]);
        wcscat(cmd, L"\"");
    }
    SetEnvironmentVariableW(L"PYTHONUTF8", L"1");
#ifdef DEBUG
    SetEnvironmentVariableW(L"VUP_DEBUG", L"1");
#endif

    STARTUPINFOW si = {sizeof(si)};
    PROCESS_INFORMATION pi;
    if (!CreateProcessW(exe, cmd, NULL, NULL, FALSE, 0, NULL, dir, &si, &pi)) {
        wchar_t msg[MAX_PATH * 3];
        swprintf(msg, MAX_PATH * 3, L"프로그램을 시작하지 못했습니다.\n\n%ls\n\n설치 폴더가 손상되었을 수 있습니다. 다시 설치해 주세요.", exe);
        MessageBoxW(NULL, msg, L"AI 업스케일러", MB_ICONERROR);
        return;
    }
#ifdef DEBUG
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    if (code != 0) {
        wprintf(L"\n[디버그] 프로그램이 오류 코드 %lu 로 끝났습니다.\n", code);
        wprintf(L"이 창의 내용과 설치 폴더의 cache\\error.log 를 Claude 에게 보여 주세요.\n");
        wprintf(L"Enter 를 누르면 창이 닫힙니다...");
        getchar();
    }
#endif
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
}

#ifdef DEBUG
int wmain(void) { run(); return 0; }
#else
int WINAPI wWinMain(HINSTANCE h, HINSTANCE p, LPWSTR c, int n) { run(); return 0; }
#endif
