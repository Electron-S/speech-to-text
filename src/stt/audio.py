"""입력 경로 정규화와 16kHz mono wav 변환."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from . import ffutil

# 'C:\Users\...' / 'D:/rec/a.m4a' 처럼 Windows 형식으로 붙여넣은 경로
_WIN_PATH = re.compile(r"^[A-Za-z]:[\\/]")

SAMPLE_RATE = 16000


def normalize_path(raw: str) -> Path:
    """붙여넣은 경로를 WSL에서 열 수 있는 Path로 바꾼다.

    핸드폰에서 옮긴 파일은 대개 Windows 탐색기에서 경로를 복사해 오므로
    'C:\\Users\\...' 와 따옴표가 그대로 섞여 들어온다.
    """
    s = raw.strip().strip('"').strip("'")
    if _WIN_PATH.match(s):
        if shutil.which("wslpath"):
            try:
                out = ffutil.run(["wslpath", "-u", s]).stdout.strip()
                if out:
                    return Path(out)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                pass
        # wslpath가 없거나 실패하면 직접 변환: C:\a\b -> /mnt/c/a/b
        drive = s[0].lower()
        rest = s[3:].replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}")
    return Path(s).expanduser()


def probe(path: Path) -> tuple[float, str]:
    """(길이초, 코덱명). 오디오로 읽을 수 없으면 ValueError."""
    try:
        info = ffutil.ffprobe_json(path)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        raise ValueError(f"오디오로 읽을 수 없는 파일입니다: {path.name}") from e
    audio = next(
        (s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None
    )
    if audio is None:
        raise ValueError(f"오디오 스트림이 없습니다: {path.name}")
    duration = float(info.get("format", {}).get("duration") or audio.get("duration") or 0)
    return duration, audio.get("codec_name", "?")


def to_wav16k(src: Path, dst: Path) -> Path:
    """어떤 입력이든 16kHz mono 16bit wav로. whisper와 화자분리가 같은 파일을 본다."""
    ffutil.run([
        "ffmpeg", "-y", "-v", "error",
        "-i", str(src),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_s16le", "-f", "wav",
        str(dst),
    ])
    return dst


def read_wav_mono(path: Path):
    """16kHz mono wav -> (float32 numpy 배열, sample_rate). 화자분리 입력용."""
    import wave

    import numpy as np

    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"16bit mono wav가 아닙니다: {path}")
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype("float32") / 32768.0
    return samples, rate
