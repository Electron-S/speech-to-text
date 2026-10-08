"""ffmpeg/ffprobe subprocess 래퍼.

subprocess I/O를 utf-8로 못박아 Windows(기본 cp949)에서도 같은 동작을 하게 한다.
autocut/src/autocut/ffutil.py 와 같은 패턴이며, 이 도구에 필요한 부분만 남겼다.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


class FFmpegNotFound(RuntimeError):
    pass


def require_ffmpeg() -> None:
    if all(shutil.which(exe) for exe in ("ffmpeg", "ffprobe")):
        return
    try:
        # sudo 없이 쓰는 경로: pip로 받은 static 바이너리
        import static_ffmpeg  # type: ignore[import-not-found]
        static_ffmpeg.add_paths()
    except ImportError:
        pass
    for exe in ("ffmpeg", "ffprobe"):
        if shutil.which(exe) is None:
            raise FFmpegNotFound(
                f"'{exe}' 를 찾을 수 없습니다. 설치: pip install static-ffmpeg "
                f"(또는 sudo apt install -y ffmpeg)"
            )


# 손상된 입력으로 ffmpeg가 무한 대기하는 것을 막는 안전망.
DEFAULT_TIMEOUT_S = 3600


def run(cmd: list[str], *, capture: bool = True, check: bool = True,
        timeout: float | None = DEFAULT_TIMEOUT_S) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=capture,
        check=check,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def ffprobe_json(path: Path | str) -> dict:
    proc = run([
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path),
    ])
    return json.loads(proc.stdout)
