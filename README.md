# QPreserve

A command-line tool to find the optimal H.264 encoding quality for any video by targeting a user-specified SSIM threshold.  
It samples your video (via scene changes, motion peaks, or uniform intervals), measures SSIM on those clips across a QP range, and does a binary search to identify the lowest QP that still meets your quality goal—then applies that to the full file.

## Key Features

- **Automated SSIM-guided QP search**  
  Samples representative segments and runs a binary search over QP values to hit a target SSIM.

- **Flexible sampling modes**  
  Choose between uniform intervals, FFprobe scene-change detection, or motion peaks for smarter clip selection.

- **CUDA-accelerated H.264 encoding**  
  Uses NVIDIA’s NVENC for faster re-encoding.

- **Audio passthrough or re-encode**  
  Automatically copies or converts audio streams to AAC at matching bitrates/channels.

- **Optional stereo downmix**  
  With `--add-stereo-downmix`, adds an AAC stereo downmix for each multichannel audio stream while keeping originals.

- **Audio-only processing (copy video)**  
  Use `--add-stereo-downmix-copy-video` to skip the SSIM pipeline and only process audio while copying the video stream.

- **Zero-dependency install**  
  Just FFmpeg (with CUDA support) and Python; use `uv` to run and manage the project environment.
