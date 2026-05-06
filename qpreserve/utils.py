# utils.py

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from typing import Any, Dict, IO, List, Mapping, Optional, Sequence, cast

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

    progress_cmd = _inject_ffmpeg_progress(cmd)
    logging.debug("Running FFmpeg (with progress): %s", " ".join(progress_cmd))

    log_file, log_path = _open_ffmpeg_progress_log()
    process = _spawn_ffmpeg_progress(progress_cmd, log_file)
    pbar, last_time = _start_progress_bar(total_duration, desc)
    _consume_progress_output(process, pbar, last_time)
    process.wait()
    pbar.close()
    log_file.flush()
    log_file.close()
    _finalize_ffmpeg_progress(process, progress_cmd, log_path)


def _inject_ffmpeg_progress(cmd: Sequence[Any]) -> List[str]:
    # Inject -progress pipe:1 -nostats before first "-i"
    progress_cmd: List[str] = []
    inserted = False
    for token in cmd:
        if not inserted and token == "-i":
            progress_cmd += ["-progress", "pipe:1", "-nostats"]
            inserted = True
        progress_cmd.append(token)
    return progress_cmd


def _open_ffmpeg_progress_log() -> tuple[IO[str], str]:
    # Capture stderr to a temp file so we can surface the reason on failures.
    log_file = tempfile.NamedTemporaryFile(
        prefix="ffmpeg_progress_",
        suffix=".log",
        delete=False,
        mode="w+",
        encoding="utf-8"
    )
    return log_file, log_file.name


def _spawn_ffmpeg_progress(
    progress_cmd: Sequence[Any],
    log_file: IO[str],
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        progress_cmd,
        stdout=subprocess.PIPE,
        stderr=log_file,
        text=True,
        bufsize=1
    )


def _start_progress_bar(total_duration: float, desc: str) -> tuple[tqdm[Any], float]:
    pbar = tqdm(
        total=total_duration,
        desc=desc,
        unit="s",
        bar_format="{l_bar}{bar}| {n:.2f}/{total:.2f} {unit}",
    )
    return pbar, 0.0


def _consume_progress_output(process: subprocess.Popen[str], pbar: tqdm[Any], last_time: float) -> None:
    if process.stdout:
        for line in process.stdout:
            line = line.strip()

            if line.startswith("out_time="):
                ts = line.split("=", 1)[1]
                sec = parse_timestamp(ts)
                if sec > last_time:
                    pbar.update(sec - last_time)
                    last_time = sec
            elif line.startswith("out_time_us="):
                # Some FFmpeg builds report progress primarily via out_time_us.
                raw = line.split("=", 1)[1]
                try:
                    sec = int(raw) / 1_000_000.0
                except ValueError:
                    sec = 0.0
                if sec > last_time:
                    pbar.update(sec - last_time)
                    last_time = sec
            elif line.startswith("out_time_ms="):
                # Keep compatibility with builds that emit out_time_ms.
                raw = line.split("=", 1)[1]
                try:
                    val = int(raw)
                except ValueError:
                    val = 0
                # FFmpeg has historically used microseconds in this field.
                sec = val / 1_000_000.0
                if sec > last_time:
                    pbar.update(sec - last_time)
                    last_time = sec

            elif line == "progress=end":
                break


def _finalize_ffmpeg_progress(
    process: subprocess.Popen[str],
    progress_cmd: Sequence[Any],
    log_path: str,
) -> None:
    if process.returncode != 0:
        logging.error("FFmpeg failed (code=%s). Full log: %s", process.returncode, log_path)
        err = subprocess.CalledProcessError(process.returncode, progress_cmd)
        err.ffmpeg_log = log_path  # type: ignore[attr-defined]
        raise err

    # Clean up the log on success
    try:
        os.remove(log_path)
    except OSError:
        pass


# ────────────────────────────────────────────────
# CODEC HELPERS
# ────────────────────────────────────────────────

def normalize_video_codec(video_codec: str) -> str:
    codec = (video_codec or "h264").lower()
    if codec == "hevc":
        return "h265"
    return codec


def nvenc_encoder_for(video_codec: str) -> str:
    return "hevc_nvenc" if normalize_video_codec(video_codec) == "h265" else "h264_nvenc"


def nvenc_pix_fmt_for(video_codec: str) -> str:
    return "p010le" if normalize_video_codec(video_codec) == "h265" else "yuv420p"


def cpu_encoder_for(video_codec: str) -> str:
    return "libx265" if normalize_video_codec(video_codec) == "h265" else "libx264"


def output_codec_tag(video_codec: str) -> str:
    return "hevc" if normalize_video_codec(video_codec) == "h265" else "avc"


_MAX_FILENAME_BYTES = 255


def check_output_filename_length(
    stem: str,
    out_ext: str,
    encoder_tag: str,
    resolution_label: str,
    fps_int: int,
) -> None:
    """Raise ValueError before encoding if any output filename would exceed 255 bytes.

    Assumes worst-case 2-digit QP, which is accurate in the vast majority of cases.
    """
    final_suffix = f" [{encoder_tag} {resolution_label}{fps_int} qp 51]{out_ext}"
    baseline_suffix = f" [baseline qp 51]{out_ext}"
    worst_suffix = max(final_suffix, baseline_suffix, key=lambda s: len(s.encode()))
    total = len(stem.encode()) + len(worst_suffix.encode())
    if total > _MAX_FILENAME_BYTES:
        over_by = total - _MAX_FILENAME_BYTES
        raise ValueError(
            f"Output filename would be {over_by} byte(s) too long for the filesystem "
            f"(limit {_MAX_FILENAME_BYTES} bytes). Rename the source file to be at least "
            f"{over_by} byte(s) shorter and rerun."
        )


# ────────────────────────────────────────────────
# AUDIO STREAM LOGIC
# ────────────────────────────────────────────────

DEFAULT_LOUDNORM_FILTER = "loudnorm=I=-23:TP=-2:LRA=11"


def build_audio_options(
    streams: List[Dict[str, Any]],
    normalize: bool = True,
    add_stereo_downmix: bool = False,
    loudnorm_filter: str = DEFAULT_LOUDNORM_FILTER,
) -> List[str]:
    """
    Build FFmpeg audio options for all audio streams.
    - If stream is AAC → copy
    - Otherwise → re-encode to AAC at 64 kbps per channel
    - If normalize is enabled → apply loudnorm and re-encode to AAC
    - If add_stereo_downmix is enabled → add a stereo AAC downmix for each
      multichannel stream (channels > 2), alongside the original stream
    """
    opts_maps: List[str] = []
    opts_streams: List[str] = []
    for i, s in enumerate(streams):
        opts_streams += _audio_opts_for_stream(i, s, normalize, loudnorm_filter)

    downmix_info: List[tuple[int, Optional[str]]] = []
    if add_stereo_downmix:
        downmix_count = 0
        for i, s in enumerate(streams):
            ch = int(s.get('channels') or 2)
            if ch > 2:
                opts_maps += ['-map', f'0:a:{i}']
                out_index = len(streams) + downmix_count
                opts_streams += _audio_opts_for_downmix(out_index, normalize, loudnorm_filter)
                language = _stream_language(s)
                if language:
                    opts_streams += [f'-metadata:s:a:{out_index}', f'language={language}']
                downmix_info.append((out_index, language))
                downmix_count += 1

    dispositions = _audio_default_dispositions(streams, downmix_info)
    return opts_maps + opts_streams + dispositions


def _audio_opts_for_stream(
    index: int,
    stream: Dict[str, Any],
    normalize: bool,
    loudnorm_filter: str,
) -> List[str]:
    codec = stream.get('codec_name', '')
    ch = int(stream.get('channels') or 2)
    layout = _resolve_channel_layout(stream.get('channel_layout'), ch)
    bitrate = 64 * ch

    if normalize:
        opts = [
            f'-filter:a:{index}', loudnorm_filter,
            f'-c:a:{index}', 'aac',
            f'-b:a:{index}', f'{bitrate}k',
            f'-ac:{index}', str(ch),
            f'-ar:a:{index}', '48000',
        ]
        if layout:
            opts += [f'-channel_layout:a:{index}', layout]
        return opts

    if codec != 'aac':
        opts = [
            f'-c:a:{index}', 'aac',
            f'-b:a:{index}', f'{bitrate}k',
            f'-ac:{index}', str(ch),
            f'-ar:a:{index}', '48000',
        ]
        if layout:
            opts += [f'-channel_layout:a:{index}', layout]
        return opts

    return [f'-c:a:{index}', 'copy']


def _audio_opts_for_downmix(
    index: int,
    normalize: bool,
    loudnorm_filter: str,
) -> List[str]:
    bitrate = 64 * 2
    opts: List[str] = []
    downmix_filter = "aformat=channel_layouts=stereo"
    if normalize:
        downmix_filter = f"{loudnorm_filter},{downmix_filter}"
    opts += [f'-filter:a:{index}', downmix_filter]
    opts += [
        f'-c:a:{index}', 'aac',
        f'-b:a:{index}', f'{bitrate}k',
        f'-ac:{index}', '2',
        f'-ar:a:{index}', '48000',
        f'-channel_layout:a:{index}', 'stereo',
    ]
    return opts


def _resolve_channel_layout(layout: Any, channels: int) -> str | None:
    if isinstance(layout, str) and layout:
        return layout
    if channels == 1:
        return "mono"
    if channels == 2:
        return "stereo"
    if channels == 6:
        return "5.1"
    if channels == 8:
        return "7.1"
    return None


def _stream_language(stream: Dict[str, Any]) -> Optional[str]:
    tags = stream.get('tags')
    if not isinstance(tags, dict):
        return None
    tags_map = cast(Mapping[str, Any], tags)
    language = tags_map.get('language') or tags_map.get('LANGUAGE')
    if not isinstance(language, str) or not language:
        return None
    return language


def _is_english_language(language: Optional[str]) -> bool:
    if not language:
        return False
    lang = language.strip().lower()
    return (
        lang in {"eng", "en", "english"} or
        lang.startswith("en-") or
        lang.startswith("en_")
    )


def _audio_default_dispositions(
    streams: List[Dict[str, Any]],
    downmix_info: List[tuple[int, Optional[str]]],
) -> List[str]:
    total_audio = len(streams) + len(downmix_info)
    if total_audio == 0:
        return []

    candidates: List[tuple[int, Optional[str]]] = []
    for i, s in enumerate(streams):
        ch = int(s.get('channels') or 2)
        if ch == 2:
            candidates.append((i, _stream_language(s)))
    candidates.extend(downmix_info)

    if not candidates:
        return []

    preferred_index: Optional[int] = None
    for idx, language in candidates:
        if _is_english_language(language):
            preferred_index = idx
            break
    if preferred_index is None:
        preferred_index = candidates[0][0]

    opts: List[str] = []
    for i in range(total_audio):
        opts += [f'-disposition:a:{i}', '0']
    opts += [f'-disposition:a:{preferred_index}', 'default']
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
# FFMPEG FILTER DETECTION
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
    This is a simple substring search, good enough for 'ssim', etc.
    """
    text = _ffmpeg_filters_text()
    return name in text
