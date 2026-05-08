# QPreserve

A command-line tool to find the optimal H.264 or H.265/HEVC encoding quality for any video by targeting a user-specified SSIM threshold.  
It samples your video (via scene changes, motion peaks, or uniform intervals), measures SSIM on those clips across a QP range, and does a binary search to identify the lowest QP that still meets your quality goal—then applies that to the full file.

## Key Features

- **Automated SSIM-guided QP search** — Samples representative segments and runs a binary search over QP values to hit a target SSIM.
- **Flexible sampling modes** — Choose between uniform intervals, FFprobe scene-change detection, or motion peaks for smarter clip selection.
- **CUDA-accelerated H.264 / HEVC encoding** — Uses NVIDIA's NVENC for fast re-encoding.
- **Audio passthrough or re-encode** — Automatically copies or converts audio streams to AAC at matching bitrates/channels.
- **Optional stereo downmix** — `--add-stereo-downmix` adds an AAC stereo downmix alongside each multichannel stream while keeping the originals.
- **Audio-only processing** — `--add-stereo-downmix-copy-video` skips the SSIM pipeline and only processes audio while copying the video stream.
- **Batch mode** — Cluster a directory of similar videos and encode one representative per cluster.
- **Custom output directory** — Write all output files to a separate directory, with relative path reconstruction in batch mode and optional suffix-free filenames for clean resumable runs.

## Requirements

- **Python** ≥ 3.10
- **FFmpeg** with CUDA/NVENC support (`ffmpeg` and `ffprobe` must be on your `PATH`)
- **NVIDIA GPU** with NVENC support
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`

## Installation

### Using uv (recommended)

```bash
uv pip install .
```

Or run directly without installing:

```bash
uv run qpreserve <input> [options]
```

### Using pip

```bash
pip install .
```

## Usage

### Basic

```bash
qpreserve input.mkv
```

Re-encodes `input.mkv` targeting the default SSIM of 0.986, writing the output alongside the source file.

### Target a specific quality

```bash
qpreserve input.mkv --ssim 0.992
```

### Choose a sampling strategy

```bash
qpreserve input.mkv --sampling-mode scene     # scene-change detection
qpreserve input.mkv --sampling-mode motion    # motion-peak based
qpreserve input.mkv --sampling-mode uniform   # evenly spaced intervals
```

### Resize while encoding

```bash
qpreserve input.mkv --resize 1080p
```

### Audio-only (copy video stream)

```bash
qpreserve input.mkv --add-stereo-downmix-copy-video
```

Adds a stereo AAC downmix for each multichannel audio track without re-encoding video.

### Batch mode

```bash
qpreserve /path/to/videos/ --batch-auto
```

Treats the input as a directory, clusters similar videos by quality, and encodes one representative per cluster.  
Add `--batch-dry-run` to preview planned actions without encoding.

If you expect the output to be smaller than the source (e.g. re-encoding old H.264 sources), add `--batch-size-guard` to keep outputs within the source file size:

```bash
qpreserve /path/to/videos/ --batch-auto --batch-size-guard
```

The guard operates in two layers:

1. **Pre-encode (ladder scan)** — After the expected-QP ladder is built, if the recommended QP would produce a larger output, the guard first scans the already-measured ladder rows for the highest-quality QP that fits within the source size. Only if no row in the current ladder fits does it drop to a lower quality tier and rebuild the ladder.
2. **Post-encode (actual size)** — After the final encode completes, the actual output size is verified against the source. For the cluster representative, if the output is still larger, the guard drops to the next lower tier and re-encodes. For peer members (which use the representative's fixed QP), oversized outputs are discarded with a warning.

Note: this guard is intentionally off by default. Transcoding across codecs (e.g. HEVC → H.264) commonly produces a larger output and that is expected behaviour.

### Output directory

```bash
qpreserve input.mkv --output-dir /path/to/output/
```

Writes the encoded file to the specified directory instead of alongside the source. In batch mode, the relative path of each source file within the input directory is recreated under the output directory:

```text
/media/videos/movies/film.mkv  →  /output/movies/film [H265 1080p24 qp 28].mkv
/media/videos/series/ep01.mkv  →  /output/series/ep01 [H265 1080p24 qp 28].mkv
```

Add `--no-suffix` to omit the quality/codec tag from filenames (requires `--output-dir`):

```bash
qpreserve /media/videos/ --batch-auto --output-dir /output/ --no-suffix
```

```text
/media/videos/movies/film.mkv  →  /output/movies/film.mkv
/media/videos/series/ep01.mkv  →  /output/series/ep01.mkv
```

When `--no-suffix` is active, restarting an interrupted run automatically skips any file whose output already exists in the output directory — making batch jobs safely resumable.

## Options Reference

| Option | Default | Description |
| ------ | ------- | ----------- |
| `--ssim` | `0.986` | Target SSIM threshold (0–1, higher = better quality) |
| `--codec` | `h264_nvenc` | Video codec (`h264_nvenc`, `hevc_nvenc`) |
| `--min-qp` | `6` | Minimum QP during binary search |
| `--max-qp` | `40` | Maximum QP during binary search |
| `--sampling-mode` | `auto` | Clip selection strategy (`uniform`, `scene`, `motion`, `auto`) |
| `--sample-percent` | `auto` | Percentage of video duration used for sampling |
| `--sample-count` | `auto` | Number of sample clips to extract |
| `--resize` | — | Resize to a standard label (`720p`, `1080p`, `4k`, …) |
| `--target-fps` | — | Downsample to target framerate (`24`, `30`, `60`, `120`) |
| `--add-stereo-downmix` | off | Add a stereo AAC downmix alongside each multichannel stream |
| `--add-stereo-downmix-copy-video` | off | Copy video stream; process audio only |
| `--no-full-ssim` | off | Skip full-file SSIM verification (faster, less precise) |
| `--batch-auto` | off | Directory input: cluster and batch-encode |
| `--batch-dry-run` | off | Print planned batch actions without encoding |
| `--batch-size-guard` | off | Keep outputs within source size: scan existing ladder first, then drop tier; verify actual size after encode |
| `--output-dir` | — | Write outputs to this directory; batch mode recreates relative paths |
| `--no-suffix` | off | Omit quality/codec suffix from filenames (requires `--output-dir`); enables resume on restart |
| `--log-file` | — | Write log output to a file |
| `-v` / `--verbose` | off | Enable verbose logging |

Run `qpreserve --help` for the complete option list.

## How It Works

1. **Baseline encode** — Encodes the source at a low QP to establish a perceptual reference.
2. **Sampling** — Extracts short representative clips using the chosen sampling strategy.
3. **QP binary search** — Measures SSIM on each sample at candidate QP values and converges on the highest QP that still meets the target.
4. **Final encode** — Re-encodes the full file at the selected QP, then optionally verifies SSIM on the result.

## License

MIT — see [LICENSE](LICENSE).
