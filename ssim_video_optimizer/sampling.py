# sampling.py
import json
import os
import tempfile
from pathlib import Path

from .probes import probe_video_duration
from .utils import run_cmd


def make_safe_symlink(input_file: str) -> str:
    """
    Create a temp symlink to `input_file` in a dir with a safe filename.
    Returns the symlink path.
    """
    tmpdir = tempfile.mkdtemp(prefix="ssim_safe_")
    ext = Path(input_file).suffix
    # Build the filename as a normal string:
    safe_name = f"video{ext}"
    link_path = Path(tmpdir) / safe_name
    os.symlink(input_file, link_path)
    return str(link_path)

def detect_scenes(input_file: str, threshold: float = 0.6) -> list:
    safe_input = make_safe_symlink(input_file)
    filter_arg = (
        f"movie='{safe_input}',"
        f"select=gt(scene\\,{threshold})"
    )
    cmd = [
        'ffprobe', '-v', 'quiet', '-f', 'lavfi',
        filter_arg,
        '-show_entries', 'frame=best_effort_timestamp_time:frame_tags=lavfi.scene_score',
        '-of', 'json'
    ]
    output = run_cmd(cmd, capture_output=True).stdout
    data = json.loads(output)
    return [
        float(f.get('best_effort_timestamp_time', 0.0))
        for f in data.get('frames', [])
        if 'lavfi.scene_score' in f.get('tags', {})
    ]

def detect_motion(input_file: str, top_n: int = 0) -> list:
    safe_input = make_safe_symlink(input_file)
    filter_arg = (
        f"movie='{safe_input}',"
        "fps=1,"
        "signalstats,"
        "metadata=print:key=lavfi.signalstats.YDIF"
    )
    cmd = [
        'ffprobe', '-v', 'quiet', '-f', 'lavfi',
        filter_arg,
        '-show_entries', 'frame=pkt_pts_time:frame_tags=lavfi.signalstats.YDIF',
        '-of', 'json'
    ]
    output = run_cmd(cmd, capture_output=True).stdout
    data = json.loads(output)

    scores = []
    for frame in data.get('frames', []):
        tags = frame.get('tags', {})
        score = tags.get('lavfi.signalstats.YDIF')
        if score is None:
            continue
        ts = float(frame.get('pkt_pts_time', frame.get('best_effort_timestamp_time', 0.0)))
        scores.append((float(score), ts))

    scores.sort(key=lambda x: x[0], reverse=True)
    times = [t for _, t in scores]
    return times[:top_n] if top_n else times

def select_sample_times(input_file: str, mode: str, percent: float, count: int, clip_len: float = None) -> list:
    duration = probe_video_duration(input_file)
    if mode == 'scene':
        times = detect_scenes(input_file)
    elif mode == 'motion':
        times = detect_motion(input_file)
    else:
        times = []
    if mode == 'uniform' or not times:
        span = duration * percent / 100.0
        step = max((duration - span) / max(count - 1, 1), 0)
        times = [i * step for i in range(count)]
    if clip_len is None:
        clip_len = duration * percent / 100.0 / count
    filtered = []
    for t in times:
        if all(abs(t - prev) >= clip_len for prev in filtered):
            filtered.append(t)
            if len(filtered) == count:
                break
    return (filtered + times)[:count]

def extract_samples(input_file: str, mode: str, percent: float, count: int, sample_qp: int, audio_opts: list, raw_fr: float) -> list:
    duration = probe_video_duration(input_file)
    clip_len = duration * percent / 100.0 / count
    times = select_sample_times(input_file, mode, percent, count, clip_len)
    tmpdir = tempfile.mkdtemp(prefix="ssim_sample_")
    samples = []
    for idx, t in enumerate(times):
        seg = os.path.join(tmpdir, f"seg_{idx}{os.path.splitext(input_file)[1]}")
        sample_file = os.path.join(tmpdir, f"sample_{idx}{os.path.splitext(input_file)[1]}")
        run_cmd(['ffmpeg', '-y', '-i', input_file, '-ss', str(t), '-t', str(clip_len), '-c', 'copy', seg])
        run_cmd([
            'ffmpeg', '-y', '-hwaccel', 'cuda', '-i', seg,
            '-map', '0', '-map_metadata', '0',
            '-r', str(raw_fr), '-g', str(int(max(1, round(raw_fr/2)))),
            '-bf', '2', '-pix_fmt', 'yuv420p', '-c:v', 'h264_nvenc',
            '-preset', 'p7', '-rc', 'constqp', '-qp', str(sample_qp)
        ] + audio_opts + ['-c:s', 'copy', sample_file])
        samples.append(sample_file)
    return samples
