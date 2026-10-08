"""faster-whisper 전사. cuda -> cpu, batched -> 단건 순으로 폴백한다.

폴백을 두는 이유: 노트북 GPU(8GB)에서 large-v3 + 배치는 다른 프로그램이 VRAM을
쓰고 있으면 도중에 OOM으로 죽는다. 그때 전사를 처음부터 다시 시도해서라도
결과를 내는 게 낫다.
"""
from __future__ import annotations

import ctypes
import os
import struct
import sysconfig
import tempfile
import wave
from pathlib import Path

_model_cache: dict[tuple[str, str], object] = {}
_cuda_libs_loaded = False


def _preload_cuda_libs() -> None:
    """pip로 받은 CUDA 라이브러리를 미리 프로세스에 올린다.

    ctranslate2는 libcublas/libcudnn을 dlopen(soname)으로 찾는데, torch가 없는
    venv에서는 nvidia-* 패키지의 so가 로더 검색 경로에 없어서
    'Library libcublas.so.12 is not found or cannot be loaded' 로 GPU가 통째로
    죽는다(그러면 CPU 폴백이 걸려 몇 배 느려진다). LD_LIBRARY_PATH는 프로세스
    시작 시점에 읽히므로 지금 바꿔도 늦고, RTLD_GLOBAL로 직접 올려두면 이후
    dlopen이 이미 올라온 객체를 soname으로 찾아 쓴다.
    """
    global _cuda_libs_loaded
    if _cuda_libs_loaded:
        return
    _cuda_libs_loaded = True

    candidates: list[Path] = []
    seen: set[str] = set()
    for key in ("platlib", "purelib"):
        root = Path(sysconfig.get_paths().get(key, "")) / "nvidia"
        if not root.is_dir():
            continue
        for so in sorted(root.glob("*/lib/lib*.so.*")):
            # 필요한 것만: cublas(행렬곱)와 cudnn(인코더 conv). 나머지는 건드리지 않는다.
            if not so.name.startswith(("libcublas", "libcudnn")):
                continue
            if so.name in seen:
                continue
            seen.add(so.name)
            candidates.append(so)

    # 서로 의존하므로(cublas->cublasLt, cudnn->cudnn_*) 실패한 것만 몇 번 다시 시도한다
    pending = candidates
    for _ in range(3):
        failed = []
        for so in pending:
            try:
                ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                failed.append(so)
        if not failed:
            break
        pending = failed


def _silence_wav() -> str:
    """CUDA 경로를 미리 찔러보는 1초 무음 wav. 호출자가 unlink 한다."""
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    f.close()  # wave 모듈이 파일 객체를 닫지 않으므로 이름으로 다시 연다
    with wave.open(f.name, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<16000h", *([0] * 16000)))
    return f.name


def _load_model(model_size: str, device: str):
    from faster_whisper import WhisperModel

    if device == "cuda":
        _preload_cuda_libs()
    key = (model_size, device)
    if key in _model_cache:
        return _model_cache[key]

    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    if device == "cuda":
        # CUDA 초기화를 지금 강제해서 sm/드라이버 오류가 전사 중간이 아니라 여기서 터지게 한다
        smoke = _silence_wav()
        try:
            list(model.transcribe(smoke, language="ko")[0])
        finally:
            os.unlink(smoke)
    _model_cache[key] = model
    return model


def _run(model_size: str, device: str, batch_size: int, wav: str, language: str | None,
         duration: float, vad: bool, vad_threshold: float) -> tuple[list[dict], str]:
    from faster_whisper import BatchedInferencePipeline

    model = _load_model(model_size, device)
    runner = model
    kwargs: dict = {}
    if batch_size > 1:
        runner = BatchedInferencePipeline(model=model)
        kwargs["batch_size"] = batch_size

    label = f"{model_size} @ {device}/{'float16' if device == 'cuda' else 'int8'}"
    print(f"[전사] {label}" + (f", batch {batch_size}" if batch_size > 1 else ""), flush=True)

    if vad:
        kwargs["vad_filter"] = True
        kwargs["vad_parameters"] = {
            "threshold": vad_threshold,
            "min_silence_duration_ms": 500,
        }
    else:
        kwargs["vad_filter"] = False

    segments, info = runner.transcribe(
        wav,
        language=language,
        word_timestamps=True,
        # 긴 녹음에서 직전 문장을 계속 물고 같은 말을 무한 반복하는 환각을 막는다
        condition_on_previous_text=False,
        **kwargs,
    )

    out: list[dict] = []
    next_mark = 10
    for seg in segments:
        out.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": (seg.text or "").strip(),
            "words": [
                {"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word}
                for w in (seg.words or [])
                if w.start is not None and w.end is not None
            ],
        })
        if duration > 0:
            pct = min(99, int(seg.end * 100 / duration))
            if pct >= next_mark:
                next_mark = pct + 10
                print(f"[전사] {pct}%", flush=True)
    return out, getattr(info, "language", language or "?")


def _release(model_size: str, device: str) -> None:
    """캐시에서 모델을 내리고 VRAM을 실제로 돌려받는다.

    GPU에서 CPU로 폴백할 때 앞의 모델을 확실히 놓아주지 않으면 large-v3 두 벌이
    동시에 올라가 8GB 카드에서 OOM이 난다(그러면 폴백이 폴백을 부른다).
    """
    import gc

    _model_cache.pop((model_size, device), None)
    gc.collect()


def _plans(device: str, batch_size: int, vad: bool) -> list[tuple[str, int]]:
    """시도할 (장치, 배치) 조합을 순서대로. 앞이 실패하면 다음으로 내려간다."""
    if not vad and batch_size > 1:
        # 배치 파이프라인은 VAD가 준 구간을 나눠 처리하는 구조라, VAD 없이는
        # 'No clip timestamps found' 로 무조건 실패한다. 조용히 단건으로 내린다.
        batch_size = 1
    batch_size = max(1, batch_size)

    if device == "cpu":
        return [("cpu", 1)]
    if device == "cuda":
        return [("cuda", batch_size), ("cuda", 1)] if batch_size > 1 else [("cuda", 1)]
    if batch_size > 1:
        return [("cuda", batch_size), ("cuda", 1), ("cpu", 1)]
    return [("cuda", 1), ("cpu", 1)]


def transcribe(wav: str, *, model_size: str = "large-v3", device: str = "auto",
               language: str | None = "ko", batch_size: int = 4,
               duration: float = 0.0, vad: bool = True,
               vad_threshold: float = 0.5) -> tuple[list[dict], str]:
    """(세그먼트 목록, 인식된 언어). 세그먼트에는 단어별 타임스탬프가 들어 있다."""
    plans = _plans(device, batch_size, vad)
    last = ""
    for i, (dev, batch) in enumerate(plans):
        try:
            return _run(model_size, dev, batch, wav, language, duration, vad, vad_threshold)
        except Exception as e:  # ctranslate2는 상황에 따라 여러 종류의 RuntimeError를 던진다
            # 예외 객체를 들고 있으면 traceback이 _run의 프레임을 붙잡아 GPU 모델이
            # 풀리지 않는다. 메시지만 남기고 예외는 이 블록에서 버린다.
            last = f"{type(e).__name__}: {e}"
            if i == len(plans) - 1:
                break
            nxt_dev = plans[i + 1][0]
            if dev == "cuda" and nxt_dev == "cuda":
                print(f"[전사] 배치 실패 — batch 1로 다시 시도합니다 ({last[:160]})", flush=True)
            elif dev == "cuda":
                print(f"[전사] GPU 사용 불가 — CPU로 폴백합니다. 느리면 --model small 을 "
                      f"쓰세요 ({last[:160]})", flush=True)
            else:
                print(f"[전사] 실패 — 다음 방법으로 시도합니다 ({last[:160]})", flush=True)
            if dev != nxt_dev:
                _release(model_size, dev)
    raise RuntimeError(f"전사에 실패했습니다: {last}")
