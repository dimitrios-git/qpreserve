# SSIM Video Optimizer

A command-line tool to find the optimal H.264 encoding quality for any video by targeting a user-specified SSIM threshold.  
It samples your video (via scene changes, motion peaks, or uniform intervals), measures SSIM on those clips across a QP range, and does a binary search to identify the lowest QP that still meets your quality goal—then applies that to the full file.

## Key Features

- **Automated SSIM-guided QP search**  
  Samples representative segments and runs a binary search over QP values to hit a target SSIM (default 0.98).

- **Flexible sampling modes**  
  Choose between uniform intervals, FFprobe scene-change detection, or motion peaks for smarter clip selection.

- **CUDA-accelerated H.264 encoding**  
  Uses NVIDIA’s NVENC for faster re-encoding.

- **Audio passthrough or re-encode**  
  Automatically copies or converts audio streams to AAC at matching bitrates/channels.

- **Zero-dependency install**  
  Just FFmpeg (with CUDA support) and Python; use `uv` to run and manage the project environment.
