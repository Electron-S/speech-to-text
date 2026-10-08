"""stt CLI — m4a 녹음 파일을 같은 폴더에 txt로 풀어놓는다."""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path
from typing import Annotated, Optional

import typer

from . import audio, ffutil, merge, render, transcribe as tr
from .diarize import DEFAULT_THRESHOLD

# Windows 콘솔(기본 cp949)에서 한글 로그가 깨지거나 죽지 않게 못박는다
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

app = typer.Typer(add_completion=False, help=__doc__)


def _outputs(src: Path, out_dir: Path | None, want_json: bool) -> dict[str, Path]:
    # with_suffix()를 쓰면 '2026.09.09 회의.m4a' 의 '.09 회의' 를 확장자로 오해한다.
    # 녹음 파일명에 날짜가 흔하니 이름은 문자열로 직접 만든다.
    folder = out_dir or src.parent
    stem = src.stem
    paths = {
        "text": folder / f"{stem}.txt",
        "timeline": folder / f"{stem}.timeline.txt",
    }
    if want_json:
        paths["json"] = folder / f"{stem}.json"
    return paths


def _warn_if_little_speech(utterances, duration: float, vad: bool) -> None:
    """VAD가 말소리를 버렸을 가능성을 알린다.

    배경음악이 크거나 마이크가 멀면 silero VAD가 사람 말을 잡음으로 보고 통째로
    버린다(실측: BGM 위에 얹힌 말은 앞부분이 잘려나갔다). 조용한 녹음이라 원래
    말이 적은 것과 구분할 수 없으니, 판단은 사용자에게 넘기고 사실만 알린다.
    """
    if not vad or duration < 30:
        return
    speech = sum(u.end - u.start for u in utterances)
    ratio = speech / duration if duration else 1.0
    if ratio < 0.25:
        print(f"[참고] 말소리로 인식된 구간이 전체의 {ratio * 100:.0f}%뿐입니다. "
              f"배경음이 크거나 목소리가 작았다면 --no-vad 로 다시 시도해 보세요.",
              file=sys.stderr)


def _process(src: Path, *, model: str, language: str | None, device: str, batch: int,
             diarize: bool, speakers: int, threshold: float, encoding: str,
             out_dir: Path | None, want_json: bool, force: bool,
             vad: bool, vad_threshold: float) -> bool:
    if not src.exists():
        raise FileNotFoundError(f"파일이 없습니다: {src}")

    outputs = _outputs(src, out_dir, want_json)
    existing = [p for p in outputs.values() if p.exists()]
    if existing and not force:
        # 이미 만들어둔(손으로 고쳤을 수도 있는) 결과를 말없이 덮어쓰지 않는다
        names = ", ".join(p.name for p in existing)
        print(f"[건너뜀] 결과가 이미 있습니다: {names}  (덮어쓰려면 --force)", file=sys.stderr)
        return False

    duration, codec = audio.probe(src)
    print(f"\n=== {src.name}  ({render.fmt_time(duration)}, {codec}) ===", flush=True)
    started = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="stt-") as tmp:
        wav = audio.to_wav16k(src, Path(tmp) / "audio.wav")
        segments, detected = tr.transcribe(
            str(wav), model_size=model, device=device, language=language,
            batch_size=batch, duration=duration, vad=vad, vad_threshold=vad_threshold,
        )
        spans = []
        if diarize:
            from . import diarize as dz
            spans = dz.diarize(wav, num_speakers=speakers, threshold=threshold)

    utterances = merge.build_utterances(segments, spans)
    if not utterances:
        print("[경고] 인식된 말이 없습니다(무음이거나 볼륨이 너무 작습니다)", file=sys.stderr)
        return False

    _warn_if_little_speech(utterances, duration, vad)
    n_speakers = merge.speaker_count(utterances)
    header = (f"# {src.name}  |  {render.fmt_time(duration)}  |  {model}/{detected}"
              + (f"  |  화자 {n_speakers}명" if n_speakers else ""))
    files = {
        outputs["text"]: (render.plain_text(utterances), encoding),
        outputs["timeline"]: (render.timeline_text(utterances, header=header), encoding),
    }
    if "json" in outputs:
        files[outputs["json"]] = (render.json_text(render.json_payload(
            utterances, source=src, duration=duration, model=model, language=detected)),
            "utf-8")
    render.write_outputs(files)

    elapsed = time.monotonic() - started
    speed = f", 녹음 길이의 {duration / elapsed:.1f}배속" if elapsed > 0 and duration else ""
    print(f"[완료] {len(utterances)}개 발화"
          + (f", 화자 {n_speakers}명" if n_speakers else ", 화자분리 없음")
          + f", {elapsed:.0f}초 소요{speed}")
    for path in outputs.values():
        print(f"       -> {path}")
    return True


@app.command()
def main(
    paths: Annotated[list[str], typer.Argument(
        help="녹음 파일 경로. Windows 경로(C:\\...)도 그대로 붙여넣을 수 있습니다")],
    model: Annotated[str, typer.Option("--model", "-m",
        help="large-v3(정확) / medium / small(빠름)")] = "large-v3",
    lang: Annotated[str, typer.Option("--lang", "-l",
        help="ko 고정 권장. auto 로 두면 whisper가 추정")] = "ko",
    speakers: Annotated[int, typer.Option("--speakers", "-s",
        help="화자 수를 알면 지정(0이면 자동 추정)")] = 0,
    diarize: Annotated[bool, typer.Option("--diarize/--no-diarize",
        help="화자 구분. 혼자 말한 메모면 끄는 게 빠릅니다")] = True,
    threshold: Annotated[float, typer.Option("--threshold",
        help="화자 자동 추정 민감도. 낮추면 화자를 더 잘게 나눕니다")] = DEFAULT_THRESHOLD,
    vad: Annotated[bool, typer.Option("--vad/--no-vad",
        help="무음 구간 건너뛰기. 배경음이 크거나 목소리가 작아 말이 잘려나가면 끄세요")] = True,
    vad_threshold: Annotated[float, typer.Option("--vad-threshold",
        help="말소리 판정 기준. 낮추면 작은 목소리도 살립니다(0~1)")] = 0.5,
    device: Annotated[str, typer.Option("--device", help="auto / cuda / cpu")] = "auto",
    batch: Annotated[int, typer.Option("--batch", help="GPU 배치 크기(1이면 배치 끔)")] = 4,
    encoding: Annotated[str, typer.Option("--encoding",
        help="출력 txt 인코딩. Windows 메모장 대비 기본은 BOM 있는 utf-8-sig")] = "utf-8-sig",
    out: Annotated[Optional[Path], typer.Option("--out",
        help="결과를 따로 모을 폴더(기본: 입력 파일과 같은 폴더)")] = None,
    json_out: Annotated[bool, typer.Option("--json/--no-json",
        help="기계용 json도 함께 저장")] = False,
    force: Annotated[bool, typer.Option("--force", "-f",
        help="이미 있는 결과를 덮어쓴다")] = False,
) -> None:
    """녹음 파일을 텍스트로 옮긴다. 결과는 입력 파일과 같은 폴더에 생긴다."""
    ffutil.require_ffmpeg()
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    targets = [audio.normalize_path(p) for p in paths]
    ok = failed = 0
    for src in targets:
        try:
            if _process(src, model=model, language=None if lang == "auto" else lang,
                        device=device, batch=batch, diarize=diarize, speakers=speakers,
                        threshold=threshold, encoding=encoding, out_dir=out,
                        want_json=json_out, force=force, vad=vad,
                        vad_threshold=vad_threshold):
                ok += 1
        except (ValueError, FileNotFoundError) as e:  # 경로가 틀렸거나 오디오가 아니다
            failed += 1
            print(f"[실패] {src.name}: {e}", file=sys.stderr)
        except Exception as e:
            failed += 1
            print(f"[실패] {src.name}: {type(e).__name__}: {e}", file=sys.stderr)

    if len(targets) > 1:
        print(f"\n{ok}/{len(targets)}개 파일 변환 완료")
    # 이미 변환돼 있어서 건너뛴 것은 실패가 아니다(스크립트에서 재실행할 수 있어야 한다)
    if failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
