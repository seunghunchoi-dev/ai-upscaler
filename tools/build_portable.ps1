# Builds a portable folder (no installation needed) at dist\AI-Upscaler.
# Usage:  powershell -ExecutionPolicy Bypass -File tools\build_portable.ps1
param([string]$Out = "dist\AI-Upscaler", [string]$PyVer = "3.11.9")
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$root = Split-Path $PSScriptRoot -Parent
$out = Join-Path $root $Out
$rt = Join-Path $out "runtime"
$sp = Join-Path $rt "Lib\site-packages"
$devPy = Join-Path $root ".venv\Scripts\python.exe"

# Keep the user's own data (settings, saved presets, compiled-model cache) across rebuilds
$keep = @("config.json", "presets", "cache")
$backup = Join-Path $env:TEMP ("vup_keep_" + [guid]::NewGuid().ToString("N"))
if (Test-Path $out) {
    New-Item -ItemType Directory -Force $backup | Out-Null
    foreach ($k in $keep) {
        $p = Join-Path $out $k
        if (Test-Path $p) { Move-Item $p (Join-Path $backup $k) }
    }
    Remove-Item -Recurse -Force $out
}
New-Item -ItemType Directory -Force $rt, $sp, "$out\bin", "$out\models" | Out-Null

Write-Host "[1/6] Python $PyVer embeddable"
$zip = Join-Path $env:TEMP "python-$PyVer-embed-amd64.zip"
if (-not (Test-Path $zip)) {
    Invoke-WebRequest "https://www.python.org/ftp/python/$PyVer/python-$PyVer-embed-amd64.zip" -OutFile $zip
}
Expand-Archive $zip -DestinationPath $rt -Force
$tag = ($PyVer -split '\.')[0..1] -join ''
# '..' puts the program folder (with the vup package) on sys.path
Set-Content (Join-Path $rt "python$tag._pth") "python$tag.zip`r`n.`r`nLib\site-packages`r`n..`r`nimport site" -Encoding ascii

Write-Host "[2/6] Python packages"
& $devPy -m pip install --disable-pip-version-check --no-warn-script-location -q `
    --target $sp --only-binary=:all: --platform win_amd64 --python-version $PyVer --implementation cp `
    numpy opencv-python openvino onnx onnxruntime-directml PySide6-Essentials pip
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

Write-Host "[3/6] Trimming unused parts"
$trim = @(
    "PySide6\qml", "PySide6\translations", "PySide6\include", "PySide6\typesystems", "PySide6\glue",
    "PySide6\designer.exe", "PySide6\linguist.exe", "PySide6\assistant.exe", "PySide6\lupdate.exe",
    "PySide6\lrelease.exe", "PySide6\qmlls.exe", "PySide6\qmlformat.exe", "PySide6\qmllint.exe",
    "PySide6\Qt6Designer*.dll", "PySide6\Qt6Quick*.dll", "PySide6\Qt6Qml*.dll", "PySide6\Qt6Pdf*.dll",
    "PySide6\Qt6ShaderTools.dll", "PySide6\Qt6Help.dll", "PySide6\Qt6Sql.dll", "PySide6\Qt6Test.dll",
    "PySide6\Qt6Designer*.dll", "PySide6\QtQuick*.pyd", "PySide6\QtQml*.pyd", "PySide6\QtDesigner*.pyd",
    "openvino\libs\openvino_tensorflow*.dll", "openvino\libs\openvino_paddle*.dll",
    "openvino\libs\openvino_pytorch*.dll", "openvino\libs\openvino_jax*.dll",
    "openvino\tools", "openvino\frontend\tensorflow", "openvino\frontend\pytorch", "openvino\frontend\jax"
)
foreach ($p in $trim) {
    Get-ChildItem (Join-Path $sp $p) -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
}
Get-ChildItem $sp -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
Get-ChildItem $sp -Recurse -Directory -Filter tests | Where-Object { $_.FullName -match 'numpy|onnx\\' } |
    Remove-Item -Recurse -Force

Write-Host "[4/6] Visual C++ runtime DLLs (for PCs without the redistributable)"
foreach ($d in "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "vcruntime140.dll", "vcruntime140_1.dll", "concrt140.dll") {
    $src = Join-Path $env:WINDIR "System32\$d"
    if ((Test-Path $src) -and -not (Test-Path (Join-Path $rt $d))) { Copy-Item $src $rt }
}

Write-Host "[5/6] FFmpeg"
foreach ($t in "ffmpeg", "ffprobe") {
    $local = Join-Path $root "bin\$t.exe"
    $src = if (Test-Path $local) { $local } else { (Get-Command $t).Source }
    $item = Get-Item $src
    if ($item.LinkType) { $src = $item.Target }   # winget installs symlinks
    Copy-Item $src "$out\bin\$t.exe"
}

Write-Host "[6/6] Program, models, launchers"
Copy-Item (Join-Path $root "vup") $out -Recurse
Copy-Item (Join-Path $root "tools") $out -Recurse
Copy-Item (Join-Path $root "models\*") "$out\models" -Recurse   # models\int8 등 하위 폴더 포함
Get-ChildItem $root -File | Where-Object { $_.Extension -in ".bat", ".md", ".txt" } | Copy-Item -Destination $out
Get-ChildItem $out -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force

if (Test-Path $backup) {
    foreach ($k in $keep) {
        $p = Join-Path $backup $k
        if (Test-Path $p) { Move-Item $p (Join-Path $out $k); Write-Host "Kept user data: $k" }
    }
    Remove-Item -Recurse -Force $backup
}

$size = (Get-ChildItem $out -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("Done: {0} ({1:N0} MB)" -f $out, $size)
