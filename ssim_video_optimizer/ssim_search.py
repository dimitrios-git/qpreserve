# ssim_search.py
import logging
import os
import subprocess
from statistics import mean
from typing import List, Sequence, Tuple

from .utils import run_cmd

from tqdm import tqdm


def measure_ssim_on_sample(sample_file: str, qp: int, raw_fr: float, gop: int, audio_opts: List[str]) -> float:
    """
    Re-encode a sample at the given QP (with the proper GOP) and measure SSIM against the original.
    """
    ext = os.path.splitext(sample_file)[1]
    temp_out = sample_file.replace(ext, f'_enc{ext}')

    try:
        run_cmd([
            'ffmpeg', '-y', '-hwaccel', 'cuda', '-i', sample_file,
            '-map', '0:v', '-map', '0:a?', '-map', '0:s?', '-map_metadata', '0',
            '-r', str(raw_fr), '-g', str(gop), '-bf', '2', '-pix_fmt', 'yuv420p',
            '-c:v', 'h264_nvenc', '-preset', 'p7', '-rc', 'constqp', '-qp', str(qp)
        ] + audio_opts + ['-c:s', 'copy', temp_out])
    except Exception as e:
        logging.warning(
            "Sample encode failed for %s at QP=%d (%s); using SSIM=0.",
            sample_file, qp, e
        )
        return 0.0

    # Measure SSIM
    tqdm.write(f"  Measuring SSIM for sample {os.path.basename(sample_file)} at QP={qp}...")
    def _parse_ssim(text: str) -> float | None:
        for line in text.splitlines():
            if 'All:' in line:
                try:
                    return float(line.split('All:')[1].split()[0])
                except (ValueError, IndexError):
                    return None
        return None

    def _measure(cmd_extra: Sequence[str] | None = None) -> Tuple[float | None, subprocess.CompletedProcess[str]]:
        # Keep SSIM probe lightweight to reduce OOM risk.
        cmd: List[str] = [
            'ffmpeg', '-nostdin', '-threads', '1', '-an', '-sn',
            '-i', sample_file,
            '-i', temp_out,
            '-filter_complex', 'ssim',
            '-f', 'null', '-'
        ]
        if cmd_extra:
            # Insert extra options after ffmpeg
            cmd = ['ffmpeg'] + list(cmd_extra) + cmd[1:]
        
        # Use Popen with communicate() to avoid pipe deadlock on large output
        process: subprocess.Popen[str] | None = None
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate(timeout=300)  # 5-minute timeout per sample
            res_local = subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)
        except subprocess.TimeoutExpired:
            if process is not None:
                process.kill()
            logging.warning("SSIM measurement timed out for %s at QP=%d", sample_file, qp)
            return None, subprocess.CompletedProcess(cmd, 1, "", "Timeout")
        
        val = _parse_ssim(res_local.stderr or "")
        if val is None:
            val = _parse_ssim(res_local.stdout or "")
        return val, res_local

    try:
        val, res = _measure()
    except Exception as e:
        logging.warning(
            "SSIM measurement failed for %s at QP=%d (%s); retrying with verbose log.",
            sample_file, qp, e
        )
        val = None
        res = None

    if val is None or val <= 0:
        try:
            val_fb, res_fb = _measure(cmd_extra=['-v', 'info'])
        except Exception as e:
            logging.warning(
                "Fallback SSIM measurement failed for %s at QP=%d (%s); using SSIM=0.",
                sample_file, qp, e
            )
            val_fb = None
            res_fb = None

        if val_fb is not None and val_fb > 0:
            return val_fb

        # Persist logs to help diagnose recurring parse failures
        import tempfile
        log_file = tempfile.NamedTemporaryFile(prefix="ssim_measure_", suffix=".log", delete=False, mode="w", encoding="utf-8")
        if res and res.stderr:
            log_file.write("PRIMARY STDERR:\n")
            log_file.write(res.stderr)
            log_file.write("\n")
        if res and res.stdout:
            log_file.write("PRIMARY STDOUT:\n")
            log_file.write(res.stdout)
            log_file.write("\n")
        if res_fb and res_fb.stderr:
            log_file.write("FALLBACK STDERR:\n")
            log_file.write(res_fb.stderr)
            log_file.write("\n")
        if res_fb and res_fb.stdout:
            log_file.write("FALLBACK STDOUT:\n")
            log_file.write(res_fb.stdout)
            log_file.write("\n")
        log_path = log_file.name
        log_file.close()

        logging.warning(
            "Could not parse SSIM for sample %s at QP=%d; using SSIM=0. See log: %s",
            sample_file, qp, log_path
        )
        return 0.0

    return val


def measure_ssim(qp: int, samples: List[str], raw_fr: float, gop: int, audio_opts: List[str], metric: str) -> float:
    """
    Compute the chosen SSIM metric (avg/min/max) across all sample clips at a given QP.
    """
    vals: List[float] = []
    pbar = tqdm(samples, desc=f"SSIM@QP{qp}", leave=False)
    for s in pbar:
        vals.append(measure_ssim_on_sample(s, qp, raw_fr, gop, audio_opts))
    pbar.close()

    results: dict[str, float] = {'avg': mean(vals), 'min': min(vals), 'max': max(vals)}
    tqdm.write(
        f"Sample results at QP={qp}: SSIMs={vals} "
        f"avg={results['avg']:.4f} min={results['min']:.4f} max={results['max']:.4f}"
    )
    # If a sample failed (SSIM=0), avoid letting a zero force an overly low QP.
    EPSILON = 1e-9
    if abs(results['min']) < EPSILON and results['max'] > EPSILON:
        tqdm.write(f"QP={qp}: min SSIM is 0; using max SSIM ({results['max']:.4f}) for decision.")
        return results['max']
    if abs(results['max']) < EPSILON and results['min'] > EPSILON:
        tqdm.write(f"QP={qp}: max SSIM is 0; using min SSIM ({results['min']:.4f}) for decision.")
        return results['min']
    if abs(results['min']) < EPSILON and abs(results['max']) < EPSILON:
        tqdm.write(f"QP={qp}: all sample SSIMs are 0; treating as failure.")
        return 0.0

    return results[metric]


def find_best_qp(samples: List[str], min_qp: int, max_qp: int, target_ssim: float,
                metric: str, audio_opts: List[str], raw_fr: float, gop: int) -> int:
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
