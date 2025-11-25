# probes.py
import json
from .utils import run_cmd


def probe_video_framerate(input_file: str) -> float:
    res = run_cmd([
        'ffprobe', '-v', 'quiet',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'json',
        input_file
    ], capture_output=True)

    data = json.loads(res.stdout)
    rate = data['streams'][0]['r_frame_rate']
    num, den = rate.split('/')
    return float(num) / float(den)


def probe_video_duration(input_file: str) -> float:
    res = run_cmd([
        'ffprobe', '-v', 'quiet',
        '-show_entries', 'format=duration',
        '-of', 'json',
        input_file
    ], capture_output=True)

    data = json.loads(res.stdout)
    return float(data['format']['duration'])


def probe_audio_streams(input_file: str) -> list:
    res = run_cmd([
        'ffprobe', '-v', 'quiet',
        '-show_entries', 'stream=index,codec_type,codec_name,channels',
        '-select_streams', 'a',
        '-of', 'json',
        input_file
    ], capture_output=True)
    data = json.loads(res.stdout)
    return data.get('streams', [])


def detect_hdr(input_file: str) -> dict:
    """
    Returns:
        {
            "is_hdr": True/False,
            "primaries": "...",
            "transfer": "...",
            "matrix": "..."
        }
    """

    res = run_cmd([
        'ffprobe', '-v', 'quiet',
        '-select_streams', 'v:0',
        '-show_entries',
        'stream=color_primaries,color_transfer,color_space',
        '-of', 'json',
        input_file
    ], capture_output=True)

    data = json.loads(res.stdout)
    st = data['streams'][0]

    prim = st.get('color_primaries', '').lower()
    tr = st.get('color_transfer', '').lower()
    mat = st.get('color_space', '').lower()

    is_hdr = (
        prim in ('bt2020', 'smpte2085') or
        tr in ('smpte2084', 'arib-std-b67') or
        mat in ('bt2020nc', 'bt2020c')
    )

    return {
        "is_hdr": is_hdr,
        "primaries": prim,
        "transfer": tr,
        "matrix": mat
    }


def probe_video_stream_info(input_file: str) -> dict:
    """
    Return width, height, pix_fmt of the first video stream.
    """
    res = run_cmd([
        'ffprobe', '-v', 'quiet',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,pix_fmt',
        '-of', 'json',
        input_file
    ], capture_output=True)

    data = json.loads(res.stdout)
    st = data['streams'][0]

    return {
        "width": int(st.get('width', 0) or 0),
        "height": int(st.get('height', 0) or 0),
        "pix_fmt": st.get('pix_fmt', '')
    }
