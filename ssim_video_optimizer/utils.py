# utils.py

import logging
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Sequence, Optional

from tqdm import tqdm

# ────────────────────────────────────────────────
# BASIC SIMPLE COMMAND RUNNER
# ────────────────────────────────────────────────

def run_cmd(cmd: Sequence[Any], capture_output: bool = False, timeout: Optional[int] = None) -> subprocess.CompletedProcess[str]:
    """
    Basic command runner used for ffprobe calls, SSIM evaluation,
    sample encoding, etc., where progress is NOT needed.
    """
    logging.debug("Running command: %s", " ".join(str(c) for c in cmd))
    return subprocess.run(
        cmd,
        check=True,
        stdout=(subprocess.PIPE if capture_output else subprocess.DEVNULL),
        stderr=(subprocess.PIPE if capture_output else subprocess.DEVNULL),
        text=True,
        timeout=timeout
    )


# ────────────────────────────────────────────────
# TIMESTAMP PARSER (for -progress out_time)
# ────────────────────────────────────────────────

def parse_timestamp(ts: str) -> float:
    """Convert out_time=HH:MM:SS.micro to seconds."""
    parts = ts.split(':')
    if len(parts) != 3:
        return 0.0
    h, m, s = parts
    try:
        return int(h) * 3600 + int(m) * 60 + float(s)
    except ValueError:
        return 0.0


# ────────────────────────────────────────────────
# PROGRESS-ENABLED FFMPEG RUNNER
# ────────────────────────────────────────────────

def run_ffmpeg_progress(cmd: Sequence[Any], total_duration: float, desc: str = "Processing"):
    """
    Run FFmpeg with real-time progress using:
        ffmpeg -progress pipe:1 -nostats

    total_duration: seconds (float)
    desc: label for tqdm
    """

    # Inject -progress pipe:1 -nostats before first "-i"
    progress_cmd: List[str] = []
    inserted = False
    for token in cmd:
        if not inserted and token == "-i":
            progress_cmd += ["-progress", "pipe:1", "-nostats"]
            inserted = True
        progress_cmd.append(token)

    logging.debug("Running FFmpeg (with progress): %s", " ".join(progress_cmd))

    # Capture stderr to a temp file so we can surface the reason on failures.
    log_file = tempfile.NamedTemporaryFile(
        prefix="ffmpeg_progress_",
        suffix=".log",
        delete=False,
        mode="w+",
        encoding="utf-8"
    )
    log_path = log_file.name

    process = subprocess.Popen(
        progress_cmd,
        stdout=subprocess.PIPE,
        stderr=log_file,
        text=True,
        bufsize=1
    )

    pbar = tqdm(total=total_duration, desc=desc, unit="s")
    last_time = 0.0

    if process.stdout:
        for line in process.stdout:
            line = line.strip()

            if line.startswith("out_time="):
                ts = line.split("=", 1)[1]
                sec = parse_timestamp(ts)
                if sec > last_time:
                    pbar.update(sec - last_time)
                    last_time = sec

            elif line == "progress=end":
                break

    process.wait()
    pbar.close()
    log_file.flush()
    log_file.close()

    if process.returncode != 0:
        logging.error("FFmpeg failed (code=%s). Full log: %s", process.returncode, log_path)
        err = subprocess.CalledProcessError(process.returncode, progress_cmd)
        err.ffmpeg_log = log_path  # type: ignore[attr-defined]
        raise err
    else:
        # Clean up the log on success
        try:
            os.remove(log_path)
        except OSError:
            pass


# ────────────────────────────────────────────────
# AUDIO STREAM LOGIC
# ────────────────────────────────────────────────

def build_audio_options(streams: List[Dict[str, Any]]) -> List[str]:
    """
    Build FFmpeg audio options for all audio streams.
    - If stream is AAC → copy
    - Otherwise → re-encode to AAC at 64 kbps per channel
    """
    opts: List[str] = []
    for i, s in enumerate(streams):
        codec = s.get('codec_name', '')
        ch = int(s.get('channels') or 2)
        if codec != 'aac':
            bitrate = 64 * ch
            opts += [
                f'-c:a:{i}', 'aac',
                f'-b:a:{i}', f'{bitrate}k',
                f'-ac:{i}', str(ch)
            ]
        else:
            opts += [
                f'-c:a:{i}', 'copy'
            ]
    return opts


# ────────────────────────────────────────────────
# LOGGING SETUP
# ────────────────────────────────────────────────

def setup_logging(verbose: bool, log_file: Optional[str] = None):
    handlers: List[logging.Handler] = []
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))
    if verbose:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=handlers
    )


# ────────────────────────────────────────────────
# FFMPEG FILTER DETECTION (for libvmaf, ssim, etc.)
# ────────────────────────────────────────────────

_filter_text_cache: Optional[str] = None

def _ffmpeg_filters_text() -> str:
    global _filter_text_cache
    if _filter_text_cache is None:
        # -hide_banner avoids tons of noise
        res = run_cmd(['ffmpeg', '-hide_banner', '-filters'], capture_output=True)
        _filter_text_cache = res.stdout or ""
    return _filter_text_cache or ""


def has_filter(name: str) -> bool:
    """
    Return True if ffmpeg has a filter with `name` in its filter list.
    This is a simple substring search, good enough for 'ssim', 'libvmaf', etc.
    """
    text = _ffmpeg_filters_text()
    return name in text
