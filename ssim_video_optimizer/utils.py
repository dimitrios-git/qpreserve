# utils.py
import subprocess
import logging


def run_cmd(cmd, capture_output=False):
    logging.info(f"Running command: %s", " ".join(str(c) for c in cmd))
    return subprocess.run(
        cmd, check=True,
        stdout=(subprocess.PIPE if capture_output else subprocess.DEVNULL),
        stderr=(subprocess.PIPE if capture_output else subprocess.DEVNULL),
        text=True
    )

def build_audio_options(streams: list) -> list:
    opts = []
    for i, s in enumerate(streams):
        codec = s.get('codec_name', '')
        ch = int(s.get('channels') or 2)
        if codec != 'aac':
            bitrate = 64 * ch  # Default bitrate based on channels
            opts += [
                f'-c:a:{i}', 'aac',
                f'-b:a:{i}', f'{bitrate}k',
                f'-ac:{i}', str(ch)
            ]
        else:
            opts += [
                f'-c:a:{i}', 'copy'
            ]
    return opts

def setup_logging(verbose: bool, log_file: str = None):
    handlers = []
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    if verbose:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=handlers)
