# encoder.py

import logging
import os
import subprocess
import tempfile
from typing import Dict, List, Sequence, Any, Tuple

from .utils import (
    run_ffmpeg_progress,
    normalize_video_codec,
    nvenc_encoder_for,
    cpu_encoder_for,
    output_codec_tag,
)
from .probes import (
    probe_video_duration,
    detect_hdr,
    probe_video_stream_info,
)


def _is_10bit_pix_fmt(pix_fmt: str) -> bool:
    p = (pix_fmt or "").lower()
    return "10" in p or "p010" in p


def _build_resolved_color_meta(hdr_info: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve output color metadata and mark whether any unknown field was filled."""
    prim = str(hdr_info.get("primaries") or "").lower()
    tr = str(hdr_info.get("transfer") or "").lower()
    mat = str(hdr_info.get("matrix") or "").lower()
    filled_unknown = False

    unknown_tokens = {"", "unknown", "unspecified", "undefined", "undef", "n/a", "na"}
    tr_aliases = {
        "bt470bg": "gamma28",
        "bt470m": "gamma22",
        "bt601": "smpte170m",
        "unknown": "",
        "unspecified": "",
    }
    tr = tr_aliases.get(tr, tr)

    if prim in unknown_tokens:
        prim = "bt2020" if hdr_info.get("is_hdr") else "bt709"
        filled_unknown = True
    if tr in unknown_tokens:
        tr = "smpte2084" if hdr_info.get("is_hdr") else "bt709"
        filled_unknown = True
    if mat in unknown_tokens:
        mat = "bt2020nc" if hdr_info.get("is_hdr") else "bt709"
        filled_unknown = True

    return {
        "range": "tv",
        "primaries": prim,
        "transfer": tr,
        "matrix": mat,
        "filled_unknown": filled_unknown,
    }


def _build_color_args_from_meta(meta: Dict[str, Any]) -> list[str]:
    return [
        "-color_range", str(meta.get("range", "tv")),
        "-color_primaries", str(meta.get("primaries", "bt709")),
        "-color_trc", str(meta.get("transfer", "bt709")),
        "-colorspace", str(meta.get("matrix", "bt709")),
    ]


def _build_color_bsf(codec_norm: str, meta: Dict[str, Any]) -> str | None:
    """
    Build a bitstream metadata filter that fills missing VUI color fields.
    Applied only for HEVC and only when we inferred unknown source tags.
    """
    if codec_norm != "h265":
        return None
    if not bool(meta.get("filled_unknown")):
        return None

    prim_map = {
        "bt709": 1,
        "bt470bg": 5,
        "smpte170m": 6,
        "smpte240m": 7,
        "film": 8,
        "bt2020": 9,
    }
    tr_map = {
        "bt709": 1,
        "gamma22": 4,
        "gamma28": 5,
        "smpte170m": 6,
        "smpte240m": 7,
        "linear": 8,
        "log": 9,
        "log_sqrt": 10,
        "iec61966-2-4": 11,
        "bt1361e": 12,
        "iec61966-2-1": 13,
        "bt2020-10": 14,
        "bt2020-12": 15,
        "smpte2084": 16,
        "smpte428": 17,
        "arib-std-b67": 18,
    }
    mat_map = {
        "rgb": 0,
        "bt709": 1,
        "fcc": 4,
        "bt470bg": 5,
        "smpte170m": 6,
        "smpte240m": 7,
        "ycgco": 8,
        "bt2020nc": 9,
        "bt2020c": 10,
    }

    range_code = 1 if str(meta.get("range", "tv")).lower() == "pc" else 0
    prim_code = prim_map.get(str(meta.get("primaries", "")).lower())
    tr_code = tr_map.get(str(meta.get("transfer", "")).lower())
    mat_code = mat_map.get(str(meta.get("matrix", "")).lower())
    if prim_code is None or tr_code is None or mat_code is None:
        return None

    return (
        "hevc_metadata="
        f"video_full_range_flag={range_code}:"
        f"colour_primaries={prim_code}:"
        f"transfer_characteristics={tr_code}:"
        f"matrix_coefficients={mat_code}"
    )


def _build_ssim_filter_chain(raw_fr: float | None) -> str:
    # Normalize timestamps (and optionally FPS) to avoid large frame sync buffers
    # that can blow up RAM on long or VFR sources.
    if raw_fr and raw_fr > 0:
        fr_str = f"{raw_fr:.6f}".rstrip("0").rstrip(".")
        return (
            f"[0:v]fps={fr_str},setpts=PTS-STARTPTS[ref];"
            f"[1:v]fps={fr_str},setpts=PTS-STARTPTS[dist];"
            "[ref][dist]ssim=stats_file=-"
        )
    return (
        "[0:v]setpts=PTS-STARTPTS[ref];"
        "[1:v]setpts=PTS-STARTPTS[dist];"
        "[ref][dist]ssim=stats_file=-"
    )


def _run_ssim_ffmpeg(
    input_file: str,
    encoded_file: str,
    filter_chain: str,
    extra: Sequence[str] | None = None,
    start: float | None = None,
    duration: float | None = None,
) -> tuple[str, str, subprocess.CompletedProcess[str]]:
    """
    Run ffmpeg, streaming output to a temp file (not memory) to avoid huge RAM use
    on very long videos. Returns captured log text (from temp file) and the process.
    """
    # Skip audio/sub decoding to reduce RAM/CPU. Let ffmpeg decide thread usage.
    cmd: List[str] = [
        'ffmpeg', '-hide_banner', '-nostats', '-loglevel', 'error',
        '-nostdin',
        '-an', '-sn'
    ]
    if extra:
        cmd += extra
    if start is not None:
        cmd += ['-ss', f"{start:.3f}"]
    if duration is not None:
        cmd += ['-t', f"{duration:.3f}"]
    cmd += ['-i', input_file]
    if start is not None:
        cmd += ['-ss', f"{start:.3f}"]
    if duration is not None:
        cmd += ['-t', f"{duration:.3f}"]
    cmd += [
        '-i', encoded_file,
        '-filter_complex', filter_chain,
        '-f', 'null', '-'
    ]

    res_local = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    return res_local.stdout or "", res_local.stderr or "", res_local


def _parse_ssim(text: str) -> float | None:
    values: List[float] = []
    for line in text.splitlines():
        if 'All:' in line:
            try:
                values.append(float(line.split('All:')[1].split()[0]))
            except (ValueError, IndexError):
                continue
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return sum(values) / len(values)


def _measure_ssim_once(
    input_file: str,
    encoded_file: str,
    filter_chain: str,
    extra: Sequence[str] | None = None,
    start: float | None = None,
    duration: float | None = None,
) -> Tuple[float | None, subprocess.CompletedProcess[str], str, str]:
    stdout_text, stderr_text, res_local = _run_ssim_ffmpeg(
        input_file=input_file,
        encoded_file=encoded_file,
        filter_chain=filter_chain,
        extra=extra,
        start=start,
        duration=duration,
    )
    val_local = _parse_ssim(stdout_text)
    if val_local is None:
        val_local = _parse_ssim(stderr_text)
    return val_local, res_local, stdout_text, stderr_text


def _persist_full_ssim_logs(
    encoded_file: str,
    start: float | None,
    duration: float | None,
    log_primary: str,
    log_primary_err: str,
    log_fb: str,
    log_fb_err: str,
) -> str:
    log_file = tempfile.NamedTemporaryFile(
        prefix="full_ssim_",
        suffix=".log",
        delete=False,
        mode="w",
        encoding="utf-8"
    )
    if start is not None or duration is not None:
        log_file.write(f"WINDOW: start={start} duration={duration}\n")
    if log_primary:
        log_file.write("PRIMARY LOG:\n")
        log_file.write(log_primary)
        log_file.write("\n")
    if log_primary_err:
        log_file.write("PRIMARY STDERR:\n")
        log_file.write(log_primary_err)
        log_file.write("\n")
    if log_fb:
        log_file.write("FALLBACK LOG:\n")
        log_file.write(log_fb)
        log_file.write("\n")
    if log_fb_err:
        log_file.write("FALLBACK STDERR:\n")
        log_file.write(log_fb_err)
        log_file.write("\n")
    log_path = log_file.name
    log_file.close()

    logging.warning(
        "Could not parse full-file SSIM for %s; treating as unavailable. See log: %s",
        encoded_file, log_path
    )
    return log_path


def _measure_ssim_window(
    input_file: str,
    encoded_file: str,
    filter_chain: str,
    start: float | None = None,
    duration: float | None = None,
) -> float | None:
    val = None
    log_primary = ""
    log_primary_err = ""
    log_fb = ""
    log_fb_err = ""

    try:
        val, _, log_primary, log_primary_err = _measure_ssim_once(
            input_file=input_file,
            encoded_file=encoded_file,
            filter_chain=filter_chain,
            start=start,
            duration=duration,
        )
    except Exception as e:
        logging.warning(
            "Full-file SSIM command failed for %s (%s); retrying verbose.",
            encoded_file, e
        )

    if val is None or val <= 0:
        try:
            val_fb, _, log_fb, log_fb_err = _measure_ssim_once(
                input_file=input_file,
                encoded_file=encoded_file,
                filter_chain=filter_chain,
                extra=['-v', 'info'],
                start=start,
                duration=duration,
            )
            if val_fb is not None and val_fb > 0:
                return val_fb
        except Exception as e:
            logging.warning(
                "Fallback full-file SSIM failed for %s (%s).",
                encoded_file, e
            )

    if val is not None and val > 0:
        return val

    _persist_full_ssim_logs(
        encoded_file=encoded_file,
        start=start,
        duration=duration,
        log_primary=log_primary,
        log_primary_err=log_primary_err,
        log_fb=log_fb,
        log_fb_err=log_fb_err,
    )
    return None


def _measure_ssim_in_chunks(
    input_file: str,
    encoded_file: str,
    filter_chain: str,
    chunk_seconds: float,
) -> float | None:
    duration_total = probe_video_duration(input_file)
    if duration_total <= chunk_seconds:
        return None

    acc = 0.0
    weight = 0.0
    start = 0.0
    while start < duration_total:
        window = min(chunk_seconds, duration_total - start)
        val_window = _measure_ssim_window(
            input_file=input_file,
            encoded_file=encoded_file,
            filter_chain=filter_chain,
            start=start,
            duration=window,
        )
        if val_window is None:
            return None
        acc += val_window * window
        weight += window
        start += window

    if weight > 0:
        return acc / weight
    return None


def measure_full_ssim(
    input_file: str,
    encoded_file: str,
    raw_fr: float | None = None,
    chunk_seconds: float | None = None,
) -> float | None:
    """
    Measure full-file SSIM between input_file (reference) and encoded_file (distorted).
    Returns None if parsing fails or SSIM is invalid.
    """
    print("Measuring full-file SSIM...")
    filter_chain = _build_ssim_filter_chain(raw_fr)

    if chunk_seconds and chunk_seconds > 0:
        chunked = _measure_ssim_in_chunks(
            input_file=input_file,
            encoded_file=encoded_file,
            filter_chain=filter_chain,
            chunk_seconds=chunk_seconds,
        )
        if chunked is not None:
            return chunked

    return _measure_ssim_window(
        input_file=input_file,
        encoded_file=encoded_file,
        filter_chain=filter_chain,
    )


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


def _build_bt709_tag_filter() -> str:
    """
    Force SDR BT.709 color metadata onto frames so outputs have complete tags.
    """
    return "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv"


def _merge_filters(*filters: str | None) -> str | None:
    parts = [f for f in filters if f]
    if not parts:
        return None
    return ",".join(parts)


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


def _get_resolution_label(path: str) -> str:
    """
    Return a standardized resolution label based on the video's actual dimensions.
    Maps to common standards: 480p, 720p, 1080p, 1440p, 2160p (4K), etc.
    
    Uses the larger dimension (usually height for landscape, width for portrait)
    to determine the closest standard resolution.
    """
    info = probe_video_stream_info(path)
    w = info["width"]
    h = info["height"]
    
    # Use the smaller dimension for classification (usually height in landscape)
    # This handles both landscape and portrait correctly
    min_dim = min(w, h)
    
    # Map to standard resolutions with some tolerance
    if min_dim <= 360:
        return "360p"
    elif min_dim <= 480:
        return "480p"
    elif min_dim <= 576:
        return "576p"  # PAL SD
    elif min_dim <= 720:
        return "720p"
    elif min_dim <= 1080:
        return "1080p"
    elif min_dim <= 1440:
        return "1440p"  # 2K
    elif min_dim <= 2160:
        return "2160p"  # 4K UHD
    elif min_dim <= 4320:
        return "4320p"  # 8K
    else:
        return f"{min_dim}p"  # Fallback for unusual resolutions


def encode_baseline(
    input_file: str,
    output_dir: str | None = None,
    qp: int = 15,
    extra_vf: str | None = None,
    video_codec: str = "h264",
) -> str:
    """
    Step 0 + 1: Create the "baseline" file from the ORIGINAL source.

    - Detect HDR and apply HDR->SDR tonemapping if needed.
    - Normalize to SDR BT.709, yuv420p, SAR=1.
    - Optional extra video filter (e.g., crop) before tagging.
    - Attempt H.264 NVENC in constqp mode:
        - QP=baseline
        - preset=p7
        - bf=2
    - If NVENC fails, fall back to libx264 lossless-ish baseline:
        - QP=baseline
        - preset=veryslow
    """

    base, ext = os.path.splitext(os.path.basename(input_file))
    # Normalize container: keep mp4, otherwise use mkv to safely hold codecs.
    out_ext = ".mp4" if ext.lower() == ".mp4" else ".mkv"
    filename = f"{base} [baseline qp {qp}]{out_ext}"
    output = os.path.join(output_dir, filename) if output_dir else filename

    total_duration = probe_video_duration(input_file)
    low_res = _is_low_res(input_file)

    codec_norm = normalize_video_codec(video_codec)
    nvenc_encoder = nvenc_encoder_for(codec_norm)
    cpu_encoder = cpu_encoder_for(codec_norm)
    if low_res:
        logging.info("Low-resolution source detected → baseline via %s (QP=%d).", nvenc_encoder, qp)
    else:
        logging.info("High-resolution source → baseline via %s (QP=%d).", nvenc_encoder, qp)

    hdr_info = detect_hdr(input_file)
    color_meta = _build_resolved_color_meta(hdr_info)
    source_vinfo = probe_video_stream_info(input_file)
    source_pix_fmt = str(source_vinfo.get("pix_fmt", "") or "")
    bsf_filter: str | None = None
    if codec_norm == "h264":
        hdr_filter = build_hdr_sdr_filter(hdr_info)
        tag_filter = _build_bt709_tag_filter()
        filter_chain = _merge_filters(hdr_filter, extra_vf, tag_filter)
        format_args = [
            '-pix_fmt', 'yuv420p',
            '-color_range', 'tv',
            '-color_primaries', 'bt709',
            '-color_trc', 'bt709',
            '-colorspace', 'bt709',
        ]
    else:
        filter_chain = extra_vf
        pix_fmt = 'p010le' if _is_10bit_pix_fmt(source_pix_fmt) else 'yuv420p'
        format_args = ['-pix_fmt', pix_fmt] + _build_color_args_from_meta(color_meta)
        bsf_filter = _build_color_bsf(codec_norm, color_meta)

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
        if hdr_info["is_hdr"]:
            print("HDR detected → applying HDR→SDR tone mapping")
        base_cmd += ['-vf', filter_chain]

    base_cmd += format_args

    nvenc_opts = [
        '-c:v', nvenc_encoder,
        '-preset', 'p7',
        '-rc', 'constqp',
        '-qp', str(qp),
        '-bf', '2',
    ]

    # GPU baseline (force NVENC even for low-res per user request)
    cmd = base_cmd + nvenc_opts
    if bsf_filter:
        cmd += ['-bsf:v', bsf_filter]
    cmd += [
        '-c:a', 'copy',
        '-c:s', 'copy',
        output
    ]

    try:
        run_ffmpeg_progress(cmd, total_duration, desc="Baseline Encode")
    except subprocess.CalledProcessError as err:
        logging.warning("NVENC baseline failed (%s). Falling back to %s baseline QP.", err, cpu_encoder)
        cpu_cmd = base_cmd + [
            '-c:v', cpu_encoder,
            '-preset', 'veryslow',
            '-qp', str(qp),
        ]
        if bsf_filter:
            cpu_cmd += ['-bsf:v', bsf_filter]
        cpu_cmd += [
            '-c:a', 'copy',
            '-c:s', 'copy',
            output
        ]
        run_ffmpeg_progress(cpu_cmd, total_duration, desc="Baseline Encode (CPU)")
    return output


def encode_final(
    input_file: str,
    qp: int,
    audio_opts: list[str],
    raw_fr: float,
    gop: int,
    video_codec: str = "h264",
    return_ssim: bool = False,
    ssim_chunk_seconds: float | None = None,
    output_dir: str | None = None,
    output_base: str | None = None,
    output_ext: str | None = None,
    extra_vf: str | None = None,
    output_resolution_label: str | None = None,
) -> str | tuple[str, float | None]:
    """
    Final encode (from the baseline file), with FFmpeg progress and optional SSIM.

    - input_file here is the BASELINE file (already normalized to SDR, bt709).
    - Uses selected NVENC codec -qp {qp} for all baselines.
    """

    base_in, _ = os.path.splitext(os.path.basename(input_file))
    base = output_base or base_in
    # Default to MKV for final optimized output; allow safe overrides only.
    if output_ext:
        ext = output_ext if output_ext.startswith(".") else f".{output_ext}"
        safe_exts = {".mkv", ".mp4"}
        if ext.lower() not in safe_exts:
            logging.warning(
                "Output extension %s is not supported for final encode; using .mkv instead.",
                ext
            )
            ext = ".mkv"
    else:
        ext = ".mkv"
    total_duration = probe_video_duration(input_file)
    low_res = _is_low_res(input_file)
    codec_norm = normalize_video_codec(video_codec)
    nvenc_encoder = nvenc_encoder_for(codec_norm)
    hdr_info = detect_hdr(input_file)
    color_meta = _build_resolved_color_meta(hdr_info)
    source_vinfo = probe_video_stream_info(input_file)
    source_pix_fmt = str(source_vinfo.get("pix_fmt", "") or "")

    # Get resolution label and format framerate
    resolution = output_resolution_label or _get_resolution_label(input_file)
    fps_int = int(round(raw_fr))
    
    encoder_tag = output_codec_tag(codec_norm)
    # New format: [codec resolution qp XX].source.ext
    filename = f"{base} [{encoder_tag} {resolution}{fps_int} qp {qp}].source{ext}"
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output = os.path.join(output_dir, filename)
    else:
        output = filename

    vf_opts: list[str] = []
    color_opts: list[str] = []
    bsf_opts: list[str] = []
    pix_fmt = "yuv420p"
    if codec_norm == "h264":
        filter_chain = _merge_filters(extra_vf, _build_bt709_tag_filter())
        if filter_chain:
            vf_opts = ['-vf', filter_chain]
        color_opts = [
            '-color_range', 'tv',
            '-color_primaries', 'bt709',
            '-color_trc', 'bt709',
            '-colorspace', 'bt709',
        ]
    else:
        if extra_vf:
            vf_opts = ['-vf', extra_vf]
        pix_fmt = 'p010le' if _is_10bit_pix_fmt(source_pix_fmt) else 'yuv420p'
        color_opts = _build_color_args_from_meta(color_meta)
        bsf_filter = _build_color_bsf(codec_norm, color_meta)
        if bsf_filter:
            bsf_opts = ['-bsf:v', bsf_filter]

    common_opts = [
        '-map', '0:v',
        '-map', '0:a?',
        '-map', '0:s?',
        '-map_metadata', '0',
        '-r', str(raw_fr),
        '-g', str(gop),
        '-bf', '2',
    ] + vf_opts + ['-pix_fmt', pix_fmt] + color_opts

    if low_res:
        logging.info("Low-resolution baseline → final encode via %s (QP=%d).", nvenc_encoder, qp)
    else:
        logging.info("High-resolution baseline → final encode via %s (QP=%d).", nvenc_encoder, qp)

    cmd = [
        'ffmpeg', '-y',
        '-hwaccel', 'cuda',
        '-i', input_file,
    ] + common_opts + [
        '-c:v', nvenc_encoder,
        '-preset', 'p7',
        '-rc', 'constqp',
        '-qp', str(qp),
    ] + audio_opts + bsf_opts + ['-c:s', 'copy', output]
    run_ffmpeg_progress(cmd, total_duration, desc=f"Final File Encode (QP={qp})")

    if return_ssim:
        ssim_val = measure_full_ssim(
            input_file,
            output,
            raw_fr=raw_fr,
            chunk_seconds=ssim_chunk_seconds,
        )
        if ssim_val is not None:
            print(f"Full-file SSIM at QP {qp}: {ssim_val:.4f}")
        else:
            print(f"Full-file SSIM at QP {qp}: unavailable")
        return output, ssim_val

    return output
