# One-time Windows setup for opensource-clipping. Run via setup.cmd.
# Safe to re-run: every step skips work that is already done.
#   1. FFmpeg + Deno (yt-dlp's YouTube challenge solver) + uv, via winget
#   2. Python 3.12 venv in .venv with PyTorch (CUDA build if an NVIDIA GPU is found)
#   3. Project requirements, with PyTorch pinned so nothing swaps it for the CPU build
#   4. .env from .env.sample
#   5. Whisper large-v3-turbo model in models\ (resumable download)
#   6. Smoke test

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Update-SessionPath {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
}
function Test-Command($name) { return [bool](Get-Command $name -ErrorAction SilentlyContinue) }
function Install-WithWinget($command, $id) {
    if (Test-Command $command) { Write-Host "   $command found"; return }
    if (-not (Test-Command "winget")) { throw "$command is missing and winget is not available. Install $id manually." }
    Write-Host "   Installing $id ..."
    winget install --id $id -e --silent --accept-source-agreements --accept-package-agreements
    Update-SessionPath
    if (-not (Test-Command $command)) { throw "$id installed but '$command' is not on PATH. Open a new terminal and re-run setup.cmd." }
}

Write-Step "System tools (FFmpeg, Deno, uv)"
Update-SessionPath  # pick up tools installed since this terminal was opened
Install-WithWinget "ffmpeg" "Gyan.FFmpeg"
Install-WithWinget "deno" "DenoLand.Deno"
Install-WithWinget "uv" "astral-sh.uv"

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
Write-Step "Python environment (.venv)"
if (-not (Test-Path $py)) {
    uv venv .venv --python 3.12
    if ($LASTEXITCODE -ne 0) { throw "uv venv failed" }
} else { Write-Host "   .venv exists" }

$hasNvidia = Test-Command "nvidia-smi"
$torchIndex = if ($hasNvidia) { "https://download.pytorch.org/whl/cu128" } else { "https://download.pytorch.org/whl/cpu" }
Write-Step ("PyTorch (" + $(if ($hasNvidia) { "CUDA 12.8 build for your NVIDIA GPU" } else { "CPU build, no NVIDIA GPU found" }) + ")")
uv pip install --python $py torch torchvision torchaudio --index-url $torchIndex
if ($LASTEXITCODE -ne 0) { throw "PyTorch install failed" }

Write-Step "Project requirements"
$pins = Join-Path $env:TEMP "clipper-torch-pins.txt"
uv pip freeze --python $py | Select-String -Pattern "^torch(vision|audio)?==" | ForEach-Object { $_.Line } | Set-Content -Encoding ascii $pins
uv pip install --python $py -r requirements.txt gdown -c $pins
if ($LASTEXITCODE -ne 0) { throw "Requirements install failed" }

Write-Step "Settings file (.env)"
if (-not (Test-Path ".env")) {
    (Get-Content ".env.sample") `
        -replace "^PEXELS_API_KEY=your-pexels-api-key-here$", "PEXELS_API_KEY=" `
        -replace "^HF_TOKEN=your-hf-token-here$", "HF_TOKEN=" |
        Set-Content -Encoding utf8 ".env"
    Write-Host "   Created .env - add your GOOGLE_API_KEY to it (free key: https://aistudio.google.com/apikey)"
} else { Write-Host "   .env exists" }

Write-Step "Whisper model (large-v3-turbo, ~1.6 GB, resumes if interrupted)"
$modelDir = Join-Path $PSScriptRoot "models\faster-whisper-large-v3-turbo"
$repo = "https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo/resolve/main"
New-Item -ItemType Directory -Force $modelDir | Out-Null
foreach ($f in @("config.json", "preprocessor_config.json", "tokenizer.json", "vocabulary.json")) {
    $dest = Join-Path $modelDir $f
    if (-not (Test-Path $dest)) { curl.exe -sSL --retry 5 -o $dest "$repo/$f" }
}
$bin = Join-Path $modelDir "model.bin"
$remoteSize = 0
$lenLine = curl.exe -sIL "$repo/model.bin" | Select-String -Pattern "^content-length:\s*(\d+)" | Select-Object -Last 1
if ($lenLine) { $remoteSize = [int64]$lenLine.Matches[0].Groups[1].Value }
for ($i = 1; $i -le 20; $i++) {
    $localSize = if (Test-Path $bin) { (Get-Item $bin).Length } else { 0 }
    if (($remoteSize -gt 0 -and $localSize -eq $remoteSize) -or ($remoteSize -eq 0 -and $localSize -gt 1.5GB)) {
        Write-Host "   model.bin complete"; break
    }
    Write-Host ("   downloading model.bin ({0:N0} of {1:N0} MB)" -f ($localSize / 1MB), ($remoteSize / 1MB))
    curl.exe -L -C - --retry 5 --retry-delay 5 --connect-timeout 30 -o $bin "$repo/model.bin"
}

Write-Step "Smoke test"
$env:PYTHONUTF8 = "1"
& $py -c @"
import torch, cv2, mediapipe, yt_dlp
from clipping import engine
print('   torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
compute = 'float16' if device == 'cuda' else 'int8'
engine.WhisperModel(r'$modelDir', device=device, compute_type=compute)
print('   Whisper model loads on', device)
"@
if ($LASTEXITCODE -ne 0) { throw "Smoke test failed" }
$nvenc = ffmpeg -hide_banner -encoders | Select-String "h264_nvenc"
if ($nvenc) { Write-Host "   FFmpeg NVENC (GPU video encoder) available" }
else { Write-Host "   FFmpeg NVENC not available - clips will render on the CPU (slower)" }

Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host "Next: put your Gemini key in .env, open a NEW terminal in this folder, then run:"
Write-Host '   .\clipper --url "https://www.youtube.com/watch?v=VIDEO_ID" --clips 3'
