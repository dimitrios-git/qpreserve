# ssim_search.py
import os
from statistics import mean

from .utils import run_cmd

from tqdm import tqdm


def measure_ssim_on_sample(sample_file: str, qp: int, raw_fr: float, gop: int, audio_opts: list) -> float:
    """
    Re-encode a sample at the given QP (with the proper GOP) and measure SSIM against the original.
    """
    ext = os.path.splitext(sample_file)[1]
    temp_out = sample_file.replace(ext, f'_enc{ext}')

    run_cmd([
        'ffmpeg', '-y', '-hwaccel', 'cuda', '-i', sample_file,
        '-map', '0', '-map_metadata', '0',
        '-r', str(raw_fr), '-g', str(gop), '-bf', '2', '-pix_fmt', 'yuv420p',
        '-c:v', 'h264_nvenc', '-preset', 'p7', '-rc', 'constqp', '-qp', str(qp)
    ] + audio_opts + ['-c:s', 'copy', temp_out])

    # Measure SSIM
    res = run_cmd([
        'ffmpeg', '-i', sample_file, '-i', temp_out,
        '-filter_complex', 'ssim', '-f', 'null', '-'
    ], capture_output=True)

    for line in res.stderr.splitlines():
        if 'All:' in line:
            return float(line.split('All:')[1].split()[0])
    return 0.0


def measure_ssim(qp: int, samples: list, raw_fr: float, gop: int, audio_opts: list, metric: str) -> float:
    """
    Compute the chosen SSIM metric (avg/min/max) across all sample clips at a given QP.
    """
    vals = []
    for s in tqdm(samples, desc=f"SSIM@QP{qp}"):
        vals.append(measure_ssim_on_sample(s, qp, raw_fr, gop, audio_opts))

    results = {'avg': mean(vals), 'min': min(vals), 'max': max(vals)}
    print(f"Sample results at QP={qp}: SSIMs={vals} avg={results['avg']:.4f} min={results['min']:.4f} max={results['max']:.4f}")
    return results[metric]


def find_best_qp(samples: list, min_qp: int, max_qp: int, target_ssim: float,
                metric: str, audio_opts: list, raw_fr: float, gop: int) -> int:
    """
    Binary search for the lowest QP between min_qp and max_qp where sample-based SSIM >= target_ssim.
    """
    low, high = min_qp, max_qp

    # Decide starting best based on high-QP SSIM
    best = high if measure_ssim(high, samples, raw_fr, gop, audio_opts, metric) >= target_ssim else low

    pbar = tqdm(desc="QP Binary Search", unit="step")

    while high - low > 1:
        pbar.update(1)
        mid = (low + high) // 2
        if measure_ssim(mid, samples, raw_fr, gop, audio_opts, metric) >= target_ssim:
            best, low = mid, mid
        else:
            high = mid

    pbar.close()
    return best
