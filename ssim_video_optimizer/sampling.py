# sampling.py
import os
import tempfile
from pathlib import Path

from .probes import probe_video_duration
from .utils import run_cmd

from tqdm import tqdm


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
            '-vf', filter_expr,
            '-f', 'null', '-'
        ],
        capture_output=True
    )

    times: list[float] = []
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


def _pick_from_candidates(candidates: list[float], count: int, duration: float, clip_len: float) -> list[float]:
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
    picked = []
    for i in range(count):
        idx = int(round(i * step))
        if idx >= len(candidates):
            idx = len(candidates) - 1
        t = candidates[idx]
        picked.append(t)

    # Ensure uniqueness and sorted order
    return sorted(set(picked))


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

    span = duration * percent / 100.0
    step = max((duration - span) / max(count - 1, 1), 0)
    uniform_times = [i * step for i in range(count)]

    if clip_len is None:
        clip_len = span / count if count > 0 else 0.0

    # Always guarantee we have a uniform baseline
    if sampling_mode == 'uniform':
        base_times = uniform_times
    else:
        # Scene-based candidates
        if sampling_mode == 'scene':
            threshold = 0.30
        elif sampling_mode == 'motion':
            threshold = 0.10
        else:
            # Fallback: unknown mode → uniform
            threshold = None

        if threshold is None:
            base_times = uniform_times
        else:
            candidates = _detect_scene_times(input_file, threshold=threshold)
            picked = _pick_from_candidates(candidates, count, duration, clip_len)

            if picked:
                base_times = picked
            else:
                # Fallback if scene detection failed or gave no usable times
                base_times = uniform_times

    # Ensure spacing at least clip_len and clamp within duration
    filtered: list[float] = []
    for t in base_times:
        t_clamped = max(0.0, min(t, max(duration - clip_len, 0)))
        if all(abs(t_clamped - prev) >= clip_len for prev in filtered):
            filtered.append(t_clamped)
            if len(filtered) == count:
                break

    # If we still have fewer than count, pad with uniform times
    if len(filtered) < count:
        for t in uniform_times:
            t_clamped = max(0.0, min(t, max(duration - clip_len, 0)))
            if all(abs(t_clamped - prev) >= clip_len for prev in filtered):
                filtered.append(t_clamped)
                if len(filtered) == count:
                    break

    return sorted(filtered)


def extract_samples(
    input_file: str,
    percent: float,
    count: int,
    sample_qp: int,
    audio_opts: list,
    raw_fr: float,
    sampling_mode: str = 'uniform'
) -> list[str]:
    """
    Extract and re-encode sample clips from the given input file,
    according to the chosen sampling_mode.
    """
    duration = probe_video_duration(input_file)
    if duration <= 0:
        return []

    clip_len = duration * percent / 100.0 / max(count, 1)

    times = select_sample_times(
        input_file=input_file,
        percent=percent,
        count=count,
        clip_len=clip_len,
        sampling_mode=sampling_mode
    )

    tmpdir = tempfile.mkdtemp(prefix="ssim_sample_")
    samples: list[str] = []

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
            '-bf', '2', '-pix_fmt', 'yuv420p', '-c:v', 'h264_nvenc',
            '-preset', 'p7', '-rc', 'constqp', '-qp', str(sample_qp)
        ] + audio_opts + ['-c:s', 'copy', sample_file])

        samples.append(sample_file)

    return samples
