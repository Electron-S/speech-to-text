"""결과를 txt / timeline.txt / json 으로 쓴다.

파일 쓰기에는 항상 encoding을 명시한다. Windows에서 open()의 기본값은 cp949라
명시하지 않으면 한글이 깨지거나 UnicodeEncodeError로 죽는다. 기본값을 utf-8-sig
(BOM 있는 UTF-8)로 두는 이유도 같다 — 결과 txt는 Windows 폴더에 떨어져 메모장·
워드로 열릴 텐데, BOM이 없으면 예전 도구들이 cp949로 읽어 한글을 깨뜨린다.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from contextlib import ExitStack
from pathlib import Path

DEFAULT_ENCODING = "utf-8-sig"
PARAGRAPH_GAP_S = 2.5


def fmt_time(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def speaker_label(speaker: int | None) -> str:
    return f"화자{speaker + 1}" if speaker is not None else ""


def plain_text(utterances, *, paragraph_gap: float = PARAGRAPH_GAP_S) -> str:
    """화자·시간 없는 본문. 화자가 바뀌거나 침묵이 길 때만 문단을 나눈다."""
    paragraphs: list[list[str]] = []
    prev = None
    for u in utterances:
        new_para = (
            prev is None
            or u.speaker != prev.speaker
            or u.start - prev.end >= paragraph_gap
        )
        if new_para:
            paragraphs.append([])
        paragraphs[-1].append(u.text)
        prev = u
    return "\n\n".join(" ".join(p) for p in paragraphs if p) + "\n"


def timeline_text(utterances, *, header: str | None = None) -> str:
    lines: list[str] = []
    if header:
        lines += [header, ""]
    for u in utterances:
        label = speaker_label(u.speaker)
        prefix = f"[{fmt_time(u.start)}]"
        lines.append(f"{prefix} {label}: {u.text}" if label else f"{prefix} {u.text}")
    return "\n".join(lines) + "\n"


def json_payload(utterances, *, source: Path, duration: float, model: str,
                 language: str) -> dict:
    return {
        "source": str(source),
        "duration_sec": round(duration, 3),
        "model": model,
        "language": language,
        "speakers": len({u.speaker for u in utterances if u.speaker is not None}),
        "utterances": [
            {
                "start": u.start,
                "end": u.end,
                "speaker": u.speaker,
                "speaker_label": speaker_label(u.speaker) or None,
                "text": u.text,
            }
            for u in utterances
        ],
    }


def write_outputs(files: dict[Path, tuple[str, str]]) -> None:
    """모든 결과를 준비한 뒤 교체하고, 저장 실패 시 이전 상태로 되돌린다."""
    with ExitStack() as stack:
        prepared = []
        for path, (text, encoding) in files.items():
            # 같은 파일시스템에 준비해야 replace가 원자적으로 동작한다.
            tmp = Path(stack.enter_context(tempfile.TemporaryDirectory(
                prefix=".stt-", dir=path.parent)))
            staged = tmp / "new"
            staged.write_text(text, encoding=encoding, newline="\n")
            backup = None
            if path.exists():
                backup = tmp / "previous"
                shutil.copy2(path, backup)
            prepared.append((path, staged, backup))

        published = []
        try:
            for path, staged, backup in prepared:
                staged.replace(path)
                published.append((path, backup))
        except Exception:
            for path, backup in reversed(published):
                if backup is None:
                    path.unlink()
                else:
                    backup.replace(path)
            raise


def write_text(path: Path, text: str, *, encoding: str = DEFAULT_ENCODING) -> Path:
    write_outputs({path: (text, encoding)})
    return path


def write_json(path: Path, payload: dict) -> Path:
    write_text(path, json_text(payload), encoding="utf-8")
    return path


def json_text(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
