# encoder.py
import os
from .utils import run_cmd, run_ffmpeg_progress
from .probes import probe_video_duration, detect_hdr


def measure_full_ssim(input_file: str, encoded_file: str) -> float:
    res = run_cmd([
        'ffmpeg', '-i', input_file, '-i', encoded_file,
        '-filter_complex', 'ssim', '-f', 'null', '-'
    ], capture_output=True)

    for line in res.stderr.splitlines():
        if 'All:' in line:
            return float(line.split('All:')[1].split()[0])
    return 0.0


def build_hdr_sdr_filter(hdr: dict) -> str | None:
    """
    Builds HDR->SDR tonemapping filter chain for ffmpeg, if needed.
    """

    prim = hdr["primaries"]
    tr = hdr["transfer"]
    mat = hdr["matrix"]

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

    # Fallback for weirdly tagged HDR
    if hdr["is_hdr"]:
        return (
            "zscale=primaries=bt2020:transfer=smpte2084:matrix=bt2020nc,"
            "zscale=t=linear:npl=100,"
            "tonemap=hable,"
            "zscale=primaries=bt709:transfer=bt709:matrix=bt709"
        )

    return None


def encode_baseline(input_file: str, output_dir: str | None = None) -> str:
    """
    Step 0 + 1:
        - Detect HDR
        - Tone-map if needed
        - Create a near-lossless baseline (QP 0, H.264 NVENC)
        - Use real FFmpeg progress
    """

    base, ext = os.path.splitext(os.path.basename(input_file))
    filename = f"{base} [baseline qp 0]{ext}"
    output = os.path.join(output_dir, filename) if output_dir else filename

    total_duration = probe_video_duration(input_file)

    hdr_info = detect_hdr(input_file)
    filter_chain = build_hdr_sdr_filter(hdr_info)

    cmd = ['ffmpeg', '-y', '-i', input_file, '-map', '0', '-map_metadata', '0']

    if filter_chain:
        print("HDR detected → applying HDR→SDR tone mapping")
        cmd += ['-vf', filter_chain]

    cmd += [
        '-pix_fmt', 'yuv420p',
        '-color_primaries', 'bt709',
        '-color_trc', 'bt709',
        '-colorspace', 'bt709',
        '-c:v', 'h264_nvenc',
        '-preset', 'p7',
        '-rc', 'constqp',
        '-qp', '0',
        '-bf', '2',
        '-c:a', 'copy',
        '-c:s', 'copy',
        output
    ]

    run_ffmpeg_progress(cmd, total_duration, desc="Baseline Encode")
    return output


def encode_final(input_file: str, qp: int, audio_opts: list, raw_fr: float, gop: int,
                 return_ssim: bool = False, output_dir: str | None = None):
    """
    Final encode (from baseline) with FFmpeg progress.
    """

    base, ext = os.path.splitext(os.path.basename(input_file))
    filename = f"{base} [h264_nvenc qp {qp}]{ext}"
    output = os.path.join(output_dir, filename) if output_dir else filename

    total_duration = probe_video_duration(input_file)

    cmd = [
        'ffmpeg', '-y', '-hwaccel', 'cuda', '-i', input_file,
        '-map', '0', '-map_metadata', '0',
        '-r', str(raw_fr), '-g', str(gop), '-bf', '2',
        '-pix_fmt', 'yuv420p',
        '-c:v', 'h264_nvenc',
        '-preset', 'p7',
        '-rc', 'constqp',
        '-qp', str(qp)
    ] + audio_opts + ['-c:s', 'copy', output]

    run_ffmpeg_progress(cmd, total_duration, desc=f"Final Encode (QP={qp})")

    if return_ssim:
        ssim_val = measure_full_ssim(input_file, output)
        print(f"Full-file SSIM at QP {qp}: {ssim_val:.4f}")
        return output, ssim_val

    return output
