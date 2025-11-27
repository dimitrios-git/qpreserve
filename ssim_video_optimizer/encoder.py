# encoder.py

import logging
import os
import subprocess
import tempfile
from typing import Dict, List, Sequence, Any, Tuple

from .utils import run_ffmpeg_progress
from .probes import (
    probe_video_duration,
    detect_hdr,
    probe_video_stream_info,
)


def measure_full_ssim(input_file: str, encoded_file: str) -> float | None:
    """
    Measure full-file SSIM between input_file (reference) and encoded_file (distorted).
    Returns None if parsing fails or SSIM is invalid.
    """

    def _run_ffmpeg(extra: Sequence[str] | None = None) -> tuple[str, subprocess.CompletedProcess[str], str | None]:
        """
        Run ffmpeg, streaming output to a temp file (not memory) to avoid huge RAM use
        on very long videos. Returns captured log text (from temp file) and the process.
        """
        log_file = tempfile.NamedTemporaryFile(prefix="ssim_ffmpeg_", suffix=".log", delete=False, mode="w+", encoding="utf-8")
        log_path = log_file.name

        # Limit threads and skip audio/sub decoding to reduce RAM/CPU; SIGKILLs likely mean OOM.
        cmd: List[str] = [
            'ffmpeg', '-nostdin', '-threads', '1', '-filter_threads', '1',
            '-an', '-sn', '-v', 'warning'
        ]
        if extra:
            cmd += extra
        cmd += [
            '-i', input_file, '-i', encoded_file,
            '-filter_complex', 'ssim',
            '-f', 'null', '-'
        ]

        res_local = subprocess.run(
            cmd,
            check=False,
            stdout=log_file,
            stderr=log_file,
            text=True
        )

        log_file.flush()
        log_file.seek(0)
        log_text = log_file.read()
        log_file.close()

        # Only delete the temp log on success; keep it for diagnostics otherwise.
        if res_local.returncode == 0:
            try:
                os.remove(log_path)
            except OSError:
                pass
            log_path_ret: str | None = None
        else:
            log_path_ret = log_path

        return log_text, res_local, log_path_ret

    def _parse_ssim(text: str) -> float | None:
        for line in text.splitlines():
            if 'All:' in line:
                try:
                    return float(line.split('All:')[1].split()[0])
                except (ValueError, IndexError):
                    return None
        return None

    def _measure(extra: Sequence[str] | None = None) -> Tuple[float | None, subprocess.CompletedProcess[str], str, str | None]:
        log_text, res_local, log_path = _run_ffmpeg(extra=extra)
        val_local = _parse_ssim(log_text)
        return val_local, res_local, log_text, log_path

    val = None
    log_primary = ""
    log_fb = ""
    log_path_primary: str | None = None
    log_path_fb: str | None = None

    try:
        val, _, log_primary, log_path_primary = _measure()
    except Exception as e:
        logging.warning(
            "Full-file SSIM command failed for %s (%s); retrying verbose. Log: %s",
            encoded_file, e, log_path_primary or "n/a"
        )

    if val is None or val <= 0:
        try:
            val_fb, _, log_fb, log_path_fb = _measure(extra=['-v', 'info'])
            if val_fb is not None and val_fb > 0:
                return val_fb
        except Exception as e:
            logging.warning(
                "Fallback full-file SSIM failed for %s (%s). Primary log: %s Fallback log: %s",
                encoded_file, e, log_path_primary or "n/a", log_path_fb or "n/a"
            )

    if val is not None and val > 0:
        return val

    # Persist logs to aid debugging
    log_file = tempfile.NamedTemporaryFile(prefix="full_ssim_", suffix=".log", delete=False, mode="w", encoding="utf-8")
    if log_primary:
        log_file.write("PRIMARY LOG:\n")
        log_file.write(log_primary)
        log_file.write("\n")
    if log_fb:
        log_file.write("FALLBACK LOG:\n")
        log_file.write(log_fb)
        log_file.write("\n")
    if log_path_primary:
        log_file.write(f"PRIMARY LOG PATH (kept): {log_path_primary}\n")
    if log_path_fb:
        log_file.write(f"FALLBACK LOG PATH (kept): {log_path_fb}\n")
    log_path = log_file.name
    log_file.close()

    logging.warning("Could not parse full-file SSIM for %s; treating as unavailable. See log: %s", encoded_file, log_path)
    return None


def build_hdr_sdr_filter(hdr: Dict[str, Any]) -> str | None:
    """
    Builds HDR->SDR tonemapping filter chain for ffmpeg, if needed.

    Handles:
      - PQ (HDR10, transfer=smpte2084)
      - HLG (transfer=arib-std-b67)
      - Fallback for other bt2020-tagged HDR-like streams
    """

    tr = hdr["transfer"]

    # PQ (HDR10)
    if tr == "smpte2084":
        return (
            "zscale=primaries=bt2020:transfer=smpte2084:matrix=bt2020nc,"
            "zscale=t=linear:npl=100,"
            "tonemap=hable,"
            "zscale=primaries=bt709:transfer=bt709:matrix=bt709"
        )

    # HLG
    if tr == "arib-std-b67":
        return (
            "zscale=primaries=bt2020:transfer=arib-std-b67:matrix=bt2020nc,"
            "zscale=t=linear:npl=100,"
            "tonemap=hable,"
            "zscale=primaries=bt709:transfer=bt709:matrix=bt709"
        )

    # Fallback for weird HDR tagging but bt2020-like space
    if hdr["is_hdr"]:
        return (
            "zscale=primaries=bt2020:transfer=smpte2084:matrix=bt2020nc,"
            "zscale=t=linear:npl=100,"
            "tonemap=hable,"
            "zscale=primaries=bt709:transfer=bt709:matrix=bt709"
        )

    return None


def _is_low_res(path: str) -> bool:
    """
    Decide if a video should be treated as 'low-res' for encoder safety.
    Here: max(width, height) <= 720.
    """
    info = probe_video_stream_info(path)
    w = info["width"]
    h = info["height"]
    max_dim = max(w, h)
    return max_dim <= 720


def encode_baseline(input_file: str, output_dir: str | None = None) -> str:
    """
    Step 0 + 1: Create the "baseline" file from the ORIGINAL source.

    - Detect HDR and apply HDR->SDR tonemapping if needed.
    - Normalize to SDR BT.709, yuv420p, SAR=1.
    - For low-res sources (<=720p):
        * Use libx264 lossless-ish baseline (QP=0, preset=veryslow).
        * Avoid all NVENC bugs / crashes on SD.
    - For higher resolutions:
        * Use H.264 NVENC in constqp mode:
            - QP=0
            - preset=p7
            - bf=2
    """

    base, ext = os.path.splitext(os.path.basename(input_file))
    # Normalize container: keep mp4, otherwise use mkv to safely hold codecs.
    out_ext = ".mp4" if ext.lower() == ".mp4" else ".mkv"
    filename = f"{base} [baseline qp 0]{out_ext}"
    output = os.path.join(output_dir, filename) if output_dir else filename

    total_duration = probe_video_duration(input_file)
    low_res = _is_low_res(input_file)

    if low_res:
        logging.info("Low-resolution source detected → baseline via h264_nvenc (QP=0).")
    else:
        logging.info("High-resolution source → baseline via h264_nvenc (QP=0).")

    # HDR detection + tonemap
    hdr_info = detect_hdr(input_file)
    filter_chain = build_hdr_sdr_filter(hdr_info)

    # Base command: map only v/a/s, ignore data/tmcd/etc.
    base_cmd = [
        'ffmpeg', '-y',
        '-i', input_file,
        '-map', '0:v',
        '-map', '0:a?',
        '-map', '0:s?',
        '-map_metadata', '0',
    ]

    if filter_chain:
        print("HDR detected → applying HDR→SDR tone mapping")
        base_cmd += ['-vf', filter_chain]

    base_cmd += [
        '-pix_fmt', 'yuv420p',
        '-color_primaries', 'bt709',
        '-color_trc', 'bt709',
        '-colorspace', 'bt709',
    ]

    nvenc_opts = [
        '-c:v', 'h264_nvenc',
        '-preset', 'p7',
        '-rc', 'constqp',
        '-qp', '0',
        '-bf', '2',
    ]

    # GPU baseline (force NVENC even for low-res per user request)
    cmd = base_cmd + nvenc_opts
    cmd += [
        '-c:a', 'copy',
        '-c:s', 'copy',
        output
    ]

    run_ffmpeg_progress(cmd, total_duration, desc="Baseline Encode")
    return output


def encode_final(
    input_file: str,
    qp: int,
    audio_opts: list[str],
    raw_fr: float,
    gop: int,
    return_ssim: bool = False,
    output_dir: str | None = None,
    output_base: str | None = None,
    output_ext: str | None = None,
) -> str | tuple[str, float | None]:
    """
    Final encode (from the baseline file), with FFmpeg progress and optional SSIM.

    - input_file here is the BASELINE file (already normalized to SDR, bt709).
    - Uses h264_nvenc -qp {qp} for all baselines (low-res included).
    """

    base_in, ext_in = os.path.splitext(os.path.basename(input_file))
    base = output_base or base_in
    # Normalize container: keep mp4, otherwise mkv (even if caller passed e.g. .avi)
    ext_hint = output_ext or ext_in
    ext = ".mp4" if ext_hint.lower() == ".mp4" else ".mkv"
    total_duration = probe_video_duration(input_file)
    low_res = _is_low_res(input_file)

    encoder_tag = "h264_nvenc"
    filename = f"{base} [{encoder_tag} qp {qp}]{ext}"
    output = os.path.join(output_dir, filename) if output_dir else filename

    common_opts = [
        '-map', '0:v',
        '-map', '0:a?',
        '-map', '0:s?',
        '-map_metadata', '0',
        '-r', str(raw_fr),
        '-g', str(gop),
        '-bf', '2',
        '-pix_fmt', 'yuv420p',
    ]

    if low_res:
        logging.info("Low-resolution baseline → final encode via h264_nvenc (QP=%d).", qp)
    else:
        logging.info("High-resolution baseline → final encode via h264_nvenc (QP=%d).", qp)

    cmd = [
        'ffmpeg', '-y',
        '-hwaccel', 'cuda',
        '-i', input_file,
    ] + common_opts + [
        '-c:v', 'h264_nvenc',
        '-preset', 'p7',
        '-rc', 'constqp',
        '-qp', str(qp),
    ] + audio_opts + ['-c:s', 'copy', output]
    run_ffmpeg_progress(cmd, total_duration, desc=f"Final File Encode (QP={qp})")

    if return_ssim:
        ssim_val = measure_full_ssim(input_file, output)
        if ssim_val is not None:
            print(f"Full-file SSIM at QP {qp}: {ssim_val:.4f}")
        else:
            print(f"Full-file SSIM at QP {qp}: unavailable")
        return output, ssim_val

    return output
