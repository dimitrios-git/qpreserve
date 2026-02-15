# sampling.py
import os
import tempfile
from pathlib import Path
from typing import List

from .probes import probe_video_duration
from .utils import run_cmd

from tqdm import tqdm

def _normalize_video_codec(video_codec: str) -> str:
    codec = (video_codec or "h264").lower()
    if codec == "hevc":
        return "h265"
    return codec


def _nvenc_encoder_for(video_codec: str) -> str:
    return "hevc_nvenc" if _normalize_video_codec(video_codec) == "h265" else "h264_nvenc"


def _nvenc_pix_fmt_for(video_codec: str) -> str:
    return "p010le" if _normalize_video_codec(video_codec) == "h265" else "yuv420p"


def make_safe_symlink(input_file: str) -> str:
    """
    Create a temp symlink to `input_file` in a dir with a safe filename.
    Returns the symlink path.
    """
    tmpdir = tempfile.mkdtemp(prefix="ssim_safe_")
    ext = Path(input_file).suffix
    safe_name = f"video{ext}"
    link_path = Path(tmpdir) / safe_name
    os.symlink(input_file, link_path)
    return str(link_path)


def _detect_scene_times(input_file: str, threshold: float) -> list[float]:
    """
    Use ffmpeg's scene detection to find timestamps where the scene
    changes significantly. Returns a list of pts_time values (floats).
    """
    filter_expr = f"select=gt(scene\\,{threshold}),showinfo"
    res = run_cmd(
        [
            'ffmpeg', '-v', 'info', '-i', input_file,
            # Only analyze video; some files have broken audio streams that
            # cause ffmpeg to bail out before running the filter graph.
            '-map', '0:v:0', '-an',
            '-vf', filter_expr,
            '-f', 'null', '-'
        ],
        capture_output=True
    )

    times: List[float] = []
    for line in res.stderr.splitlines():
        # Typical showinfo line contains 'showinfo' and 'pts_time:'
        if 'showinfo' in line and 'pts_time:' in line:
            try:
                # Find 'pts_time:' and parse the float that follows
                part = line.split('pts_time:')[1].split()[0]
                t = float(part)
                times.append(t)
            except (ValueError, IndexError):
                continue
    return times


def _pick_from_candidates(candidates: List[float], count: int, duration: float, clip_len: float) -> list[float]:
    """
    Given a sorted list of candidate times, pick `count` of them
    spread across the list, enforcing that they fit in [0, duration - clip_len].
    """
    if not candidates:
        return []

    candidates = sorted(t for t in candidates if 0 <= t < max(duration - clip_len, 0))

    if not candidates:
        return []

    if len(candidates) <= count:
        # If too few, just use what we have
        return candidates

    # Spread picks across the candidate list
    step = len(candidates) / float(count)
    picked: List[float] = []
    for i in range(count):
        idx = int(round(i * step))
        if idx >= len(candidates):
            idx = len(candidates) - 1
        t = candidates[idx]
        picked.append(t)

    # Ensure uniqueness and sorted order
    return sorted(set(picked))


def _compute_uniform_times(duration: float, percent: float, count: int) -> tuple[List[float], float]:
    span = duration * percent / 100.0
    step = max((duration - span) / max(count - 1, 1), 0)
    uniform_times: List[float] = [i * step for i in range(count)]
    return uniform_times, span


def _resolve_clip_len(clip_len: float | None, span: float, count: int) -> float:
    if clip_len is None:
        return span / count if count > 0 else 0.0
    return clip_len


def _scene_threshold_for_mode(sampling_mode: str) -> float | None:
    if sampling_mode == 'scene':
        return 0.30
    if sampling_mode == 'motion':
        return 0.10
    return None


def _select_base_times(
    input_file: str,
    sampling_mode: str,
    uniform_times: List[float],
    duration: float,
    clip_len: float,
    count: int,
) -> List[float]:
    if sampling_mode == 'uniform':
        return uniform_times

    threshold = _scene_threshold_for_mode(sampling_mode)
    if threshold is None:
        return uniform_times

    print("Analyzing video for scene changes to select sample points...")
    candidates = _detect_scene_times(input_file, threshold=threshold)
    picked = _pick_from_candidates(candidates, count, duration, clip_len)
    return picked or uniform_times


def _clamp_time(t: float, duration: float, clip_len: float) -> float:
    return max(0.0, min(t, max(duration - clip_len, 0)))


def _filter_and_pad_times(
    base_times: List[float],
    uniform_times: List[float],
    duration: float,
    clip_len: float,
    count: int,
) -> List[float]:
    filtered = _filter_times(base_times, duration, clip_len, count)
    if len(filtered) < count:
        filtered = _pad_times(filtered, uniform_times, duration, clip_len, count)
    return sorted(filtered)


def _filter_times(
    times: List[float],
    duration: float,
    clip_len: float,
    count: int,
) -> List[float]:
    filtered: List[float] = []
    for t in times:
        t_clamped = _clamp_time(t, duration, clip_len)
        if all(abs(t_clamped - prev) >= clip_len for prev in filtered):
            filtered.append(t_clamped)
            if len(filtered) == count:
                break
    return filtered


def _pad_times(
    filtered: List[float],
    uniform_times: List[float],
    duration: float,
    clip_len: float,
    count: int,
) -> List[float]:
    for t in uniform_times:
        t_clamped = _clamp_time(t, duration, clip_len)
        if all(abs(t_clamped - prev) >= clip_len for prev in filtered):
            filtered.append(t_clamped)
            if len(filtered) == count:
                break
    return filtered


def select_sample_times(
    input_file: str,
    percent: float,
    count: int,
    clip_len: float | None = None,
    sampling_mode: str = 'uniform'
) -> list[float]:
    """
    Select sample start times according to the chosen sampling mode.

    Modes:
      - 'uniform': evenly spaced over the duration
      - 'scene': biased toward scene changes (large differences)
      - 'motion': like scene, but with a lower threshold (more sensitive)
    """
    duration = probe_video_duration(input_file)
    if duration <= 0:
        return []

    uniform_times, span = _compute_uniform_times(duration, percent, count)
    clip_len = _resolve_clip_len(clip_len, span, count)

    base_times = _select_base_times(
        input_file=input_file,
        sampling_mode=sampling_mode,
        uniform_times=uniform_times,
        duration=duration,
        clip_len=clip_len,
        count=count,
    )

    return _filter_and_pad_times(
        base_times=base_times,
        uniform_times=uniform_times,
        duration=duration,
        clip_len=clip_len,
        count=count,
    )


def extract_samples(
    input_file: str,
    percent: float,
    count: int,
    sample_qp: int,
    audio_opts: list[str],
    raw_fr: float,
    video_codec: str = "h264",
    sampling_mode: str = 'uniform',
    tmp_root: str | None = None,
) -> tuple[list[str], str]:
    """
    Extract and re-encode sample clips from the given input file,
    according to the chosen sampling_mode.
    """
    duration = probe_video_duration(input_file)
    if duration <= 0:
        return [], ""

    clip_len = duration * percent / 100.0 / max(count, 1)

    times = select_sample_times(
        input_file=input_file,
        percent=percent,
        count=count,
        clip_len=clip_len,
        sampling_mode=sampling_mode
    )

    if not times:
        return [], ""

    tmpdir = tempfile.mkdtemp(prefix="ssim_sample_", dir=tmp_root)
    samples: list[str] = []
    nvenc_encoder = _nvenc_encoder_for(video_codec)
    pix_fmt = _nvenc_pix_fmt_for(video_codec)

    for idx, t in enumerate(tqdm(times, desc="Extracting samples")):
        ext = os.path.splitext(input_file)[1]
        seg = os.path.join(tmpdir, f"seg_{idx}{ext}")
        sample_file = os.path.join(tmpdir, f"sample_{idx}{ext}")

        # Extract the sample segment
        run_cmd([
            'ffmpeg', '-y',
            '-ss', str(t),
            '-i', input_file,
            '-t', str(clip_len),
            '-c', 'copy',
            seg
        ])

        # Encode sample with explicit stream mapping
        run_cmd([
            'ffmpeg', '-y', '-hwaccel', 'cuda', '-i', seg,
            '-map', '0:v', '-map', '0:a?', '-map', '0:s?', '-map_metadata', '0',
            '-r', str(raw_fr), '-g', str(int(max(1, round(raw_fr / 2)))),
            '-bf', '2', '-pix_fmt', pix_fmt, '-c:v', nvenc_encoder,
            '-preset', 'p7', '-rc', 'constqp', '-qp', str(sample_qp)
        ] + audio_opts + ['-c:s', 'copy', sample_file])

        samples.append(sample_file)

    return samples, tmpdir


def extract_sample_segments(
    input_file: str,
    percent: float,
    count: int,
    sampling_mode: str = 'uniform',
    tmp_root: str | None = None,
) -> tuple[list[str], str, float, float]:
    """
    Extract raw sample segments (stream copy) from the given input file.
    Returns (segments, tmpdir, clip_len, duration).
    """
    duration = probe_video_duration(input_file)
    if duration <= 0:
        return [], "", 0.0, 0.0

    clip_len = duration * percent / 100.0 / max(count, 1)

    times = select_sample_times(
        input_file=input_file,
        percent=percent,
        count=count,
        clip_len=clip_len,
        sampling_mode=sampling_mode
    )

    if not times:
        return [], "", 0.0, duration

    tmpdir = tempfile.mkdtemp(prefix="size_sample_", dir=tmp_root)
    segments: list[str] = []
    ext = os.path.splitext(input_file)[1]

    for idx, t in enumerate(tqdm(times, desc="Extracting samples")):
        seg = os.path.join(tmpdir, f"seg_{idx}{ext}")
        run_cmd([
            'ffmpeg', '-y',
            '-ss', str(t),
            '-i', input_file,
            '-t', str(clip_len),
            '-c', 'copy',
            seg
        ])
        segments.append(seg)

    return segments, tmpdir, clip_len, duration
