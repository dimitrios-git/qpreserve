# utils.py

import subprocess
import logging
import re
from tqdm import tqdm


# ────────────────────────────────────────────────
# BASIC SIMPLE COMMAND RUNNER (used where progress NOT needed)
# ────────────────────────────────────────────────

def run_cmd(cmd, capture_output=False):
    """
    Original run_cmd used for ffprobe calls, SSIM evaluation,
    sample encoding, and any non-progress ffmpeg invocation.
    """
    logging.debug(f"Running command: %s", " ".join(str(c) for c in cmd))
    return subprocess.run(
        cmd,
        check=True,
        stdout=(subprocess.PIPE if capture_output else subprocess.DEVNULL),
        stderr=(subprocess.PIPE if capture_output else subprocess.DEVNULL),
        text=True
    )


# ────────────────────────────────────────────────
# TIMESTAMP PARSER
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
# PROGRESS-ENABLED FFMPEG RUNNER (Option B implementation)
# ────────────────────────────────────────────────

def run_ffmpeg_progress(cmd, total_duration: float, desc="Processing"):
    """
    Run FFmpeg with real-time progress using:
        ffmpeg -progress pipe:1 -nostats

    total_duration: seconds (float)
    desc: label for tqdm
    """

    # Inject before the first "-i"
    progress_cmd = []
    inserted = False
    for token in cmd:
        if not inserted and token == "-i":
            progress_cmd += ["-progress", "pipe:1", "-nostats"]
            inserted = True
        progress_cmd.append(token)

    logging.debug(f"Running FFmpeg (with progress): {' '.join(progress_cmd)}")

    process = subprocess.Popen(
        progress_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1
    )

    pbar = tqdm(total=total_duration, desc=desc, unit="s")
    last_time = 0.0

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

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, progress_cmd)


# ────────────────────────────────────────────────
# AUDIO STREAM LOGIC (FROM ORIGINAL FILE)
# ────────────────────────────────────────────────

def build_audio_options(streams: list) -> list:
    """
    Build FFmpeg audio options for all audio streams.
    - If stream is AAC → copy
    - Otherwise → re-encode to AAC at 64 kbps per channel
    """
    opts = []
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
# LOGGING SETUP (FROM ORIGINAL FILE)
# ────────────────────────────────────────────────

def setup_logging(verbose: bool, log_file: str = None):
    handlers = []
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    if verbose:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=handlers
    )
