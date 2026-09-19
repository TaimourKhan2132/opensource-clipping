"""
Local wrapper around main.py for this Windows setup.

- Applies sensible defaults (any flag you pass overrides them).
- The pipeline always saves the source as video_asli.mp4 and yt-dlp skips files
  that already exist, so a new URL would silently re-clip the previous video.
  When the URL changes, the old source, subtitles and AI response are deleted.
- After a successful run, this run's clips/thumbnails/metadata are copied to
  outputs/runs/<date>_<video-id>/ so runs don't overwrite each other.

Usage: clipper --url "VIDEO_URL" [main.py options]
"""
import glob
import os
import re
import shutil
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "outputs")
LAST_URL_FILE = os.path.join(BASE, ".last_source_url")

# setup.cmd downloads the model here; otherwise faster-whisper fetches it by name
LOCAL_WHISPER = os.path.join(BASE, "models", "faster-whisper-large-v3-turbo")
WHISPER_MODEL = LOCAL_WHISPER if os.path.isfile(os.path.join(LOCAL_WHISPER, "model.bin")) else "large-v3-turbo"

DEFAULTS = [
    "--font-style", "DEFAULT",          # HORMOZI's Montserrat download is broken (renders as Arial)
    "--whisper-model", WHISPER_MODEL,   # turbo: ~half the size of large-v3, fits a 6 GB GPU
    "--source-height", "1080",          # keeps downloads small on a slow connection
    "--bgm-mood", "chill",              # calm music instead of the AI's per-clip pick
    "--track-mode", "smooth",           # lag-free camera path: faces stay centered, fewer jumps
    "--no-hook",                        # no 3s teaser + TV-static intro
]


def _arg_value(args, *names):
    for i, a in enumerate(args):
        for n in names:
            if a == n and i + 1 < len(args):
                return args[i + 1]
            if a.startswith(n + "="):
                return a.split("=", 1)[1]
    return None


def _video_slug(url):
    m = re.search(r"(?:v=|youtu\.be/|shorts/|video/|reel/|file/d/)([\w-]{6,})", url)
    slug = m.group(1) if m else url.rstrip("/").split("/")[-1]
    return re.sub(r"[^\w-]", "_", slug)[:40] or "video"


def main():
    args = sys.argv[1:]
    url = _arg_value(args, "--url", "-u")
    if url is None or "-h" in args or "--help" in args:
        os.execv(sys.executable, [sys.executable, os.path.join(BASE, "main.py"), *args])

    last_url = open(LAST_URL_FILE, encoding="utf-8").read().strip() if os.path.exists(LAST_URL_FILE) else None
    if url != last_url:
        stale = glob.glob(os.path.join(BASE, "video_asli*")) + [os.path.join(OUT, "gemini_response.json")]
        for path in stale:
            if os.path.isfile(path):
                os.remove(path)
        if last_url:
            print(f"🧹 New source URL — removed the previous video's source files.")
    # Record the URL before running, so an interrupted download of the same video resumes
    with open(LAST_URL_FILE, "w", encoding="utf-8") as f:
        f.write(url)

    started = time.time()
    code = subprocess.call([sys.executable, os.path.join(BASE, "main.py"), *DEFAULTS, *args], cwd=BASE)
    if code != 0:
        sys.exit(code)

    produced = [
        p for p in glob.glob(os.path.join(OUT, "*"))
        if os.path.isfile(p) and os.path.getmtime(p) >= started - 1
        and p.lower().endswith((".mp4", ".jpg", ".json"))
    ]
    if produced:
        run_dir = os.path.join(OUT, "runs", time.strftime("%Y-%m-%d_%H%M") + "_" + _video_slug(url))
        os.makedirs(run_dir, exist_ok=True)
        for p in produced:
            shutil.copy2(p, run_dir)
        print(f"\n📁 This run's clips are saved in: {run_dir}")


if __name__ == "__main__":
    main()
