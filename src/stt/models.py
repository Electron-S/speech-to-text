"""화자분리 모델 파일 다운로드/캐시.

faster-whisper 모델은 huggingface_hub이 알아서 캐시하지만, sherpa-onnx 모델은
GitHub 릴리스에 올라온 파일을 직접 받아야 한다. 둘 다 토큰이 필요 없다.
"""
from __future__ import annotations

import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

_BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"

# pyannote segmentation 3.0 을 ONNX로 export 한 것 (k2-fsa 배포, 게이트 없음)
SEG_URL = f"{_BASE}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
SEG_MEMBER = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
# 3D-Speaker 화자 임베딩. 중국어 코퍼스로 학습됐지만 임베딩은 언어 비의존이라
# 한국어에도 그대로 쓴다(무슨 말을 하는지가 아니라 목소리를 본다).
EMB_URL = (f"{_BASE}/speaker-recongition-models/"
           "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx")

MIN_BYTES = 100_000  # 받다 끊긴 파일/에러 HTML을 모델로 착각하지 않게


def cache_dir() -> Path:
    root = os.environ.get("STT_MODEL_DIR")
    path = Path(root) if root else Path.home() / ".cache" / "stt-models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download(url: str, dst: Path) -> Path:
    """.part 로 받아서 다 받은 뒤에만 제자리로 옮긴다(중간에 끊겨도 캐시가 안 썩음)."""
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"[모델] 내려받는 중: {dst.name}")
    with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as f:
        shutil.copyfileobj(resp, f)
    if tmp.stat().st_size < MIN_BYTES:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"모델 다운로드가 비정상적으로 작습니다: {url}")
    tmp.replace(dst)
    return dst


def segmentation_model() -> Path:
    out = cache_dir() / "pyannote-segmentation-3-0.onnx"
    if out.exists() and out.stat().st_size >= MIN_BYTES:
        return out
    archive = cache_dir() / "seg.tar.bz2"
    _download(SEG_URL, archive)
    with tarfile.open(archive, "r:bz2") as tar:
        member = tar.getmember(SEG_MEMBER)
        src = tar.extractfile(member)
        if src is None:
            raise RuntimeError(f"압축 안에 {SEG_MEMBER} 가 없습니다")
        tmp = out.with_suffix(".part")
        with src, tmp.open("wb") as f:
            shutil.copyfileobj(src, f)
        tmp.replace(out)
    archive.unlink(missing_ok=True)
    return out


def embedding_model() -> Path:
    out = cache_dir() / "3dspeaker-eres2net-base-16k.onnx"
    if out.exists() and out.stat().st_size >= MIN_BYTES:
        return out
    return _download(EMB_URL, out)
