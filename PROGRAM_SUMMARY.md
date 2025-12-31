# SSIM Video Optimizer – Code Overview

## Purpose

- CLI tool to re-encode videos with NVIDIA NVENC H.264 while hitting a target perceptual quality threshold (currently SSIM; VMAF placeholder).  
- Automates sample selection, QP search, and final encode with progress feedback, handling HDR→SDR tonemapping when needed.

## End-to-End Flow (`ssim_video_optimizer/cli.py`)

- Validate input exists, probe basic video info, and warn/confirm on problematic sources (resolutions >4096 or 10-bit pixel formats).  
- Set up logging and scratch directories (env `SSIM_SCRATCH_DIR` or `--scratch-dir`), warning if free space looks low.  
- **Baseline encode:** `encoder.encode_baseline` creates a normalized copy at QP 0 using `h264_nvenc` (container mp4→mp4 else mkv). If HDR is detected, it tonemaps to SDR BT.709, yuv420p because the pipeline forces 8-bit H.264; NVENC does not support 10-bit H.264 HDR (only HEVC does), so HDR metadata would be invalid otherwise. Making this optional would require a “keep HDR / no tonemap” flag plus an alternate HDR-capable path (e.g., HEVC 10-bit).  
- **Sample extraction:** `sampling.extract_samples` selects start times (`uniform` | `scene` | `motion`) and cuts clips totaling `--sample-percent` across `--sample-count` segments, then re-encodes each sample at `--sample-qp` with NVENC.  
- **QP search on samples:** `ssim_search.find_best_qp` runs a binary search between `--min-qp`/`--max-qp`, re-encoding each sample per QP and measuring SSIM against the sample source. Aggregates via `--metric` (`avg|min|max`) and chooses the lowest QP meeting `--ssim`.  
- **Final encode:** `encoder.encode_final` re-encodes the full baseline at the chosen QP (preserving original container hint, copying subtitles, encoding/copying audio per `build_audio_options`). Shows FFmpeg progress.  
- **Full-file SSIM descent (optional):** Unless `--skip-full-ssim`, iterates downward from the sample-derived QP, measuring full-video SSIM each time and keeping the highest-SSIM encode that meets the threshold or the first available SSIM.  
- Moves the accepted output next to the source file and cleans scratch directories.

## Key Modules

- `encoder.py`: Baseline/final NVENC pipelines; HDR detection hookup; full-file SSIM measurement with log preservation on parse failures.  
- `sampling.py`: Scene/motion detection via FFmpeg `showinfo`; spacing logic to avoid overlap; sample extraction + NVENC sample encode.  
- `ssim_search.py`: Sample re-encode + SSIM measurement (lightweight ffmpeg runs) and binary search to pick QP.  
- `probes.py`: FFprobe wrappers for duration, framerate, HDR metadata, stream info.  
- `utils.py`: Command runner, FFmpeg progress integration (uses `-progress` + tqdm), audio option builder (copy AAC else encode AAC @64 kbps/channel), filter availability helper.  
- `batch.sh`: Finds video files recursively, skips already converted/tagged outputs, and runs `ssim-video-optimizer` per file; removes tiny outputs and continues on errors.

## Assumptions and Notable Behaviors

- Requires FFmpeg with CUDA/NVENC and `ssim` filter; VMAF path is stubbed even if `libvmaf` exists.  
- All encodes use H.264 NVENC (QP mode, preset p7, `bf=2`); low-res files are still forced through NVENC per current design.  
- Baseline and final outputs normalize to yuv420p, BT.709, SAR=1; subtitles copied, metadata preserved.  
- Full-file SSIM logs persist on parse failures to aid debugging; temp files/dirs are otherwise cleaned.  
- Containers: mp4 inputs remain mp4; everything else defaults to mkv to safely hold streams.

## Entry Points

- Python: `python -m ssim_video_optimizer` or `ssim_video_optimizer.cli:main`.  
- CLI script: `ssim-video-optimizer` (installed via packaging).  
- Batch helper: `ssim_video_optimizer/batch.sh` for directory-wide processing.

## Next Features and Plans (ideas)

- **Presets for non-experts:** One-click profiles such as PS4/Plex-safe H.264 (level/profile/vbv caps), Universal H.264, and “Keep HDR via HEVC” with sane audio/sub defaults.
- **HDR choice:** Toggle between tonemap-to-SDR (H.264 8-bit) and keep-HDR (HEVC 10-bit) with automatic path selection based on source and target device.
- **FFmpeg bundling/checks:** Windows-friendly packaging with a bundled static FFmpeg (or first-run download) so no PATH setup is required; clear error if missing.
- **Hardware/codec flexibility:** Detect NVENC/QuickSync/VAAPI; fall back to x264. Optional HEVC/AV1 outputs for capable devices while keeping H.264 as the default.
- **Simple/Advanced UI modes:** Beginner view with only file/folder picker + preset; advanced panel exposing target SSIM, QP bounds, sampling mode, scratch dir.
- **GUI packaging:** PyInstaller-style single-file installer/zip with embedded Python deps and bundled FFmpeg; first-run probe for ffmpeg/ffprobe with a one-click download fallback.
- **Robust batch UX:** Per-file status, progress, ETA, and end-of-run summary; resume/skip failed items; keep concise logs for troubleshooting.
- **Multi-preset runs:** Allow selecting multiple presets in one pass; reuse the same baseline and samples, then emit multiple outputs without re-running sampling/SSIM.
- **Multi-resolution ladder:** Optional outputs at 480p/720p/1080p/2160p (Plex-style ladder) using the same sampling/SSIM decisions where possible.
- **Audio/sub handling:** Auto AAC with optional stereo downmix, sensible defaults for subs, and warnings when containers cannot hold certain subtitle types (fallback to mkv when needed).
- **Quality metrics expansion:** Optional VMAF when available; configurable sample length/count; pluggable metrics while keeping SSIM default.
- **GUI stack:** PySide6 for a native-feeling cross-platform UI; aligns with Qt docs/examples and works with PyInstaller for Windows bundling.

## GUI First Mini Action Plan

- Minimal PySide6 window: file/folder picker, scratch/output picker, preset selector, Run button, log panel.
- Start with one “Test preset” mirroring current CLI defaults (SSIM target, QP bounds, sampling mode); map UI choices to CLI args and show the generated command.
- Execute CLI in a background thread/process; stream stdout/stderr to the log panel; show per-item states (Queued/Running/Done/Failed).
- Validate prerequisites upfront: check ffmpeg/ffprobe availability and input existence; surface friendly errors.
- Keep launchable via `python -m ...` initially; later add PyInstaller packaging and FFmpeg bundling.
