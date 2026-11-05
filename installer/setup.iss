; AI 영상·사진 업스케일러 설치 파일 (Inno Setup)
; Copyright (c) 2026 최승훈. All rights reserved.
; tools\build_installer.ps1 이 /D 옵션으로 AppVersion, StageDir, OutDir 를 넘긴다.

#ifndef AppVersion
  #define AppVersion "1.1.0"
#endif

[Setup]
AppId={{6F1C9B3A-2D4E-4C8A-9B71-3E5D2A7C4F10}
AppName=AI 영상·사진 업스케일러
AppVersion={#AppVersion}
AppVerName=AI 영상·사진 업스케일러 {#AppVersion}
AppPublisher=최승훈
AppCopyright=Copyright (c) 2026 최승훈. All rights reserved.
VersionInfoCompany=최승훈
VersionInfoCopyright=Copyright (c) 2026 최승훈. All rights reserved.
VersionInfoProductName=AI 영상·사진 업스케일러
VersionInfoVersion={#AppVersion}
; 관리자 권한 없이 사용자 폴더에 설치 (설정·캐시·오류 기록을 저장할 수 있도록)
PrivilegesRequired=lowest
DefaultDirName={localappdata}\Programs\AI-Upscaler
DefaultGroupName=AI 영상·사진 업스케일러
DisableProgramGroupPage=yes
LicenseFile={#StageDir}\LICENSE.txt
OutputDir={#OutDir}
OutputBaseFilename=AI업스케일러_설치_v{#AppVersion}
SetupIconFile={#StageDir}\app.ico
UninstallDisplayIcon={app}\AIUpscaler.exe
; ultra64 + 8 스레드는 Inno Setup(32비트) 압축기에서 메모리 부족이 나서 max + 2 스레드로
Compression=lzma2/max
SolidCompression=yes
LZMANumBlockThreads=2
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern

[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"

[Tasks]
Name: "desktopicon"; Description: "바탕 화면에 바로 가기 만들기"; GroupDescription: "추가 작업:"

[Files]
Source: "{#StageDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\AI 업스케일러"; Filename: "{app}\AIUpscaler.exe"
Name: "{group}\AI 업스케일러 (디버그 모드)"; Filename: "{app}\AIUpscaler-Debug.exe"; Comment: "오류가 날 때 콘솔에 내용을 보여 줍니다"
Name: "{group}\사용법"; Filename: "{app}\사용법.md"
Name: "{group}\사용권 계약서"; Filename: "{app}\LICENSE.txt"
Name: "{group}\제거"; Filename: "{uninstallexe}"
Name: "{autodesktop}\AI 업스케일러"; Filename: "{app}\AIUpscaler.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\AIUpscaler.exe"; Description: "지금 실행"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; 프로그램이 만든 임시 파일 정리 (사용자 설정 config.json 과 presets 는 남김 → 다시 설치하면 설정 유지)
Type: filesandordirs; Name: "{app}\cache"
; 내장 Python 이 실행 중 만든 라이브러리 캐시(.pyc)까지 지우도록 runtime 폴더 전체 삭제
Type: filesandordirs; Name: "{app}\runtime"
