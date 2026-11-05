# AI 영상·사진 업스케일러 설치 파일(.exe) 빌드
# Copyright (c) 2026 최승훈. All rights reserved.
#
# 사용법:  powershell -ExecutionPolicy Bypass -File tools\build_installer.ps1
# 결과:    dist\installer\AI업스케일러_설치_v<버전>.exe
#          release\v<버전>\  (그 버전의 소스 사본 + 빌드 정보 — 나중에 Claude 로 디버깅할 때 사용)
#
# 보호: 프로그램 코드(vup)는 Nuitka 로 C → 기계어 컴파일한 .pyd 하나로만 배포하고 .py 소스는 넣지 않는다.
# MinGW 컴파일러가 한글 경로를 처리하지 못해서 C:\vupbuild (영문 경로)에서 컴파일한다.
param([switch]$KeepStage)  # 테스트용: 설치 내용 폴더(build\installer_stage)를 지우지 않고 남긴다
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$root = Split-Path $PSScriptRoot -Parent
$ver = ([regex]'__version__ = "([^"]+)"').Match((Get-Content "$root\vup\__init__.py" -Raw -Encoding utf8)).Groups[1].Value
Write-Host "버전 $ver"
$B = "C:\vupbuild"
$stage = Join-Path $root "build\installer_stage"
$outDir = Join-Path $root "dist\installer"
$release = Join-Path $root "release\v$ver"

# ── 1. 영문 경로 빌드 환경 (없으면 만든다) ─────────────────────────────────
New-Item -ItemType Directory -Force $B, "$B\tmp", "$B\launcher" | Out-Null
if (-not (Test-Path "$B\venv\Scripts\python.exe")) {
    Write-Host "[1/7] 빌드 환경 만들기 (처음 한 번)"
    $py = (& "$root\.venv\Scripts\python.exe" -c "import sys; print(sys.base_prefix)").Trim()
    robocopy $py "$B\py311" /E /XD "$py\Lib\site-packages" /NFL /NDL /NJH /NJS /NP | Out-Null
    New-Item -ItemType Directory -Force "$B\py311\Lib\site-packages" | Out-Null
    & "$B\py311\python.exe" -m venv "$B\venv"
    & "$B\venv\Scripts\python.exe" -m pip install -q nuitka ordered-set zstandard
} else { Write-Host "[1/7] 빌드 환경 있음" }
$env:NUITKA_CACHE_DIR = "$B\cache"; $env:TEMP = "$B\tmp"; $env:TMP = "$B\tmp"

# ── 2. 포터블 구성 (내장 Python·라이브러리·FFmpeg·모델) ─────────────────────
Write-Host "[2/7] 포터블 구성"
& powershell -ExecutionPolicy Bypass -File "$root\tools\build_portable.ps1" -Out "build\installer_stage"
if ($LASTEXITCODE -ne 0) { throw "포터블 구성 실패" }

# ── 3. 프로그램 코드 컴파일 (vup → .pyd) ─────────────────────────────────
Write-Host "[3/7] 프로그램 코드 컴파일 (Nuitka)"
Remove-Item -Recurse -Force "$B\src", "$B\out" -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force "$B\src", "$B\out" | Out-Null
robocopy "$root\vup" "$B\src\vup" *.py /NFL /NDL /NJH /NJS /NP | Out-Null
Push-Location "$B\src"
& "$B\venv\Scripts\python.exe" -m nuitka --module vup --include-package=vup --output-dir="$B\out" `
    --assume-yes-for-downloads --remove-output --no-pyi-file --python-flag=no_docstrings --lto=no
Pop-Location
$pyd = Get-ChildItem "$B\out\vup.*.pyd" | Select-Object -First 1
if (-not $pyd) { throw "컴파일 실패" }

# ── 4. 실행 파일 (런처) ─────────────────────────────────────────────────
Write-Host "[4/7] 실행 파일"
$gcc = (Get-ChildItem "$B\cache\downloads\gcc" -Recurse -Filter gcc.exe | Where-Object { $_.FullName -match 'mingw64\\bin\\gcc.exe$' } | Select-Object -First 1).DirectoryName
$env:PATH = "$gcc;$env:PATH"
$p = $ver.Split('.')
$rc = (Get-Content "$root\installer\launcher.rc" -Raw -Encoding utf8).Replace('@MAJOR@', $p[0]).Replace('@MINOR@', $p[1]).Replace('@PATCH@', $p[2]).Replace('@VERSION@', $ver)
[IO.File]::WriteAllText("$B\launcher\launcher.rc", $rc, (New-Object System.Text.UTF8Encoding $false))
Copy-Item "$root\installer\launcher.c" "$B\launcher\" -Force
& "$root\.venv\Scripts\python.exe" "$root\installer\make_icon.py" "$B\launcher\app.ico" | Out-Null
Push-Location "$B\launcher"
windres --codepage=65001 launcher.rc -O coff -o res.o
gcc -O2 -s -municode -mwindows launcher.c res.o -o AIUpscaler.exe -lshell32
gcc -O2 -s -municode -DDEBUG launcher.c res.o -o AIUpscaler-Debug.exe -lshell32
Pop-Location

# ── 5. 설치 내용 정리: 소스 제거, 컴파일본·실행 파일·계약서·라이선스 넣기 ──────
Write-Host "[5/7] 설치 내용 정리"
# FFmpeg 과 GFPGAN 은 라이선스 때문에 함께 배포하지 않는다 (프로그램의 [추가 구성요소] 에서 사용자가 받음)
foreach ($x in "vup", "tools", "cache", "config.json", "실행.bat", "명령줄.bat", "NVIDIA_GPU_지원_설치.bat",
               "설치파일_만들기.bat", "개발자_안내.md", "사용법.md", "README.md", "requirements.txt",
               "bin", "models\gfpgan_1.4.onnx") {
    $t = Join-Path $stage $x
    if (Test-Path $t) { Remove-Item -Recurse -Force $t }
}
Copy-Item $pyd.FullName $stage
Copy-Item "$B\launcher\AIUpscaler.exe", "$B\launcher\AIUpscaler-Debug.exe", "$B\launcher\app.ico" $stage
Copy-Item "$root\LICENSE.txt", "$root\THIRD_PARTY_NOTICES.txt" $stage
New-Item -ItemType Directory -Force "$stage\presets" | Out-Null
$utf8bom = New-Object System.Text.UTF8Encoding $true
$cli = "@echo off`r`nrem 명령줄 실행 예: 명령줄.bat 입력.mp4 --auto`r`nchcp 65001 >nul`r`ncd /d `"%~dp0`"`r`nset PYTHONUTF8=1`r`n`"runtime\python.exe`" -X utf8 -c `"import sys, vup.cli; sys.exit(vup.cli.main())`" %*`r`n"
[IO.File]::WriteAllText("$stage\명령줄.bat", $cli, (New-Object System.Text.UTF8Encoding $false))
$nv = (Get-Content "$root\NVIDIA_GPU_지원_설치.bat" -Raw -Encoding utf8).Replace('"%PY%" -m vup --devices', '"%PY%" -X utf8 -c "import sys, vup.cli; sys.exit(vup.cli.main([''--devices'']))"')
[IO.File]::WriteAllText("$stage\NVIDIA_GPU_지원_설치.bat", $nv, (New-Object System.Text.UTF8Encoding $false))
# 라이선스 원문: 각 패키지와 함께 배포된 파일을 그대로 복사
$lic = New-Item -ItemType Directory -Force "$stage\licenses"
Get-ChildItem "$stage\runtime\Lib\site-packages" -Directory -Filter *.dist-info | ForEach-Object {
    $files = Get-ChildItem $_.FullName -Recurse -File | Where-Object { $_.Name -match '^(LICEN[CS]E|COPYING|NOTICE|AUTHORS|LGPL|GPL)' -or $_.Directory.Name -eq 'licenses' }
    if ($files) {
        $d = New-Item -ItemType Directory -Force (Join-Path $lic.FullName ($_.Name -replace '\.dist-info$', ''))
        $files | Copy-Item -Destination $d.FullName
    }
}
if (Test-Path "$stage\runtime\LICENSE.txt") { Copy-Item "$stage\runtime\LICENSE.txt" "$($lic.FullName)\Python-LICENSE.txt" }
Copy-Item "$root\licenses\*" $lic.FullName
$ort = New-Item -ItemType Directory -Force "$($lic.FullName)\onnxruntime"
Copy-Item "$stageuntime\Lib\site-packages\onnxruntime\LICENSE", "$stageuntime\Lib\site-packages\onnxruntime\ThirdPartyNotices.txt" $ort.FullName

# ── 6. 빌드 정보 + 그 버전의 소스 사본 (디버깅용, 설치 파일에는 넣지 않음) ────
Write-Host "[6/7] 소스 사본 보관 → $release"
New-Item -ItemType Directory -Force $release | Out-Null
$zip = Join-Path $release "source_v$ver.zip"
if (Test-Path $zip) { Remove-Item $zip }
$tmpSrc = "$B\snapshot"; Remove-Item -Recurse -Force $tmpSrc -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $tmpSrc | Out-Null
foreach ($x in "vup", "installer", "tools", "licenses") { robocopy "$root\$x" "$tmpSrc\$x" /E /XD __pycache__ /NFL /NDL /NJH /NJS /NP | Out-Null }
Copy-Item "$root\LICENSE.txt", "$root\THIRD_PARTY_NOTICES.txt", "$root\사용법.md", "$root\개발자_안내.md", "$root\실행.bat", "$root\명령줄.bat", "$root\설치파일_만들기.bat", "$root\NVIDIA_GPU_지원_설치.bat" $tmpSrc -ErrorAction SilentlyContinue
Compress-Archive -Path "$tmpSrc\*" -DestinationPath $zip
$info = @"
AI 영상·사진 업스케일러 v$ver
빌드 시각: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
컴파일본: $($pyd.Name)  SHA256 $((Get-FileHash $pyd.FullName -Algorithm SHA256).Hash)
소스 사본: source_v$ver.zip  SHA256 $((Get-FileHash $zip -Algorithm SHA256).Hash)
Copyright (c) 2026 최승훈. All rights reserved.
"@
[IO.File]::WriteAllText("$release\BUILD_INFO.txt", $info, $utf8bom)
[IO.File]::WriteAllText("$stage\BUILD_INFO.txt", ($info -split "`n" | Where-Object { $_ -notmatch '소스 사본' }) -join "`n", $utf8bom)

# ── 7. 설치 파일 ────────────────────────────────────────────────────────
Write-Host "[7/7] 설치 파일 만들기 (Inno Setup)"
$iscc = @("$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe", "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe", "$env:ProgramFiles\Inno Setup 6\ISCC.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) { throw "Inno Setup 이 없습니다: winget install JRSoftware.InnoSetup" }
New-Item -ItemType Directory -Force $outDir | Out-Null
& $iscc /Q "/DAppVersion=$ver" "/DStageDir=$stage" "/DOutDir=$outDir" "$root\installer\setup.iss"
if ($LASTEXITCODE -ne 0) { throw "설치 파일 만들기 실패" }
$exe = Get-ChildItem $outDir -Filter "*v$ver.exe" | Select-Object -First 1
# 중간 작업 폴더 정리 (약 1.6GB)
Remove-Item -Recurse -Force "$B\snapshot", "$B\src", "$B\out" -ErrorAction SilentlyContinue
if (-not $KeepStage) { Remove-Item -Recurse -Force $stage -ErrorAction SilentlyContinue }
Write-Host ("완료: {0} ({1:N0} MB)" -f $exe.FullName, ($exe.Length / 1MB))
