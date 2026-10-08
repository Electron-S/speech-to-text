import json

import pytest
from typer.testing import CliRunner

from stt import cli
from stt.cli import _warn_if_little_speech
from stt.merge import Utterance


def _warn(capsys, utterances, duration, vad=True) -> str:
    _warn_if_little_speech(utterances, duration, vad)
    return capsys.readouterr().err


def test_warns_when_almost_nothing_was_transcribed(capsys):
    """BGM 위에 얹힌 말을 VAD가 버리는 실측 사례를 사용자가 알아채게 한다."""
    err = _warn(capsys, [Utterance(0, 5, 0, "짧은 한 마디")], duration=600)
    assert "--no-vad" in err


def test_no_warning_when_speech_fills_the_recording(capsys):
    ut = [Utterance(0, 250, 0, "가"), Utterance(250, 500, 1, "나")]
    assert _warn(capsys, ut, duration=600) == ""


def test_no_warning_when_vad_is_already_off(capsys):
    assert _warn(capsys, [Utterance(0, 5, 0, "가")], duration=600, vad=False) == ""


def test_no_warning_for_short_clips(capsys):
    """20초 메모에 한 마디만 있는 건 이상한 일이 아니다."""
    assert _warn(capsys, [Utterance(0, 2, 0, "가")], duration=20) == ""


def test_zero_duration_does_not_divide_by_zero(capsys):
    assert _warn(capsys, [], duration=0) == ""


def test_missing_file_is_an_error_not_a_skip(tmp_path):
    """경로를 잘못 붙여넣은 것은 조용히 넘길 일이 아니다."""
    import pytest
    from stt.cli import _process
    with pytest.raises(FileNotFoundError):
        _process(tmp_path / "없는 녹음.m4a", model="small", language="ko", device="cpu",
                 batch=1, diarize=False, speakers=0, threshold=0.5, encoding="utf-8",
                 out_dir=None, want_json=False, force=False, vad=True, vad_threshold=0.5)


def test_existing_results_are_not_overwritten(tmp_path, capsys):
    """손으로 고쳐둔 전사본을 자동 재생성으로 날리지 않는다."""
    from stt.cli import _process
    src = tmp_path / "회의.m4a"
    src.write_bytes(b"not really audio")
    (tmp_path / "회의.txt").write_text("손으로 고친 내용", encoding="utf-8")
    ran = _process(src, model="small", language="ko", device="cpu", batch=1, diarize=False,
                   speakers=0, threshold=0.5, encoding="utf-8", out_dir=None,
                   want_json=False, force=False, vad=True, vad_threshold=0.5)
    assert ran is False
    assert "--force" in capsys.readouterr().err
    assert (tmp_path / "회의.txt").read_text(encoding="utf-8") == "손으로 고친 내용"


@pytest.mark.parametrize("text", ["한글", "English only"])
def test_save_failure_can_be_retried_without_force(tmp_path, monkeypatch, text):
    # 영문 본문도 한글 파일명이 들어간 timeline 저장에서 실패할 수 있다.
    src = tmp_path / "회의.m4a"
    src.write_bytes(b"audio placeholder")
    monkeypatch.setattr(cli.ffutil, "require_ffmpeg", lambda: None)
    monkeypatch.setattr(cli.audio, "probe", lambda src: (1, "aac"))
    monkeypatch.setattr(cli.audio, "to_wav16k", lambda src, dst: dst)
    monkeypatch.setattr(cli.tr, "transcribe", lambda *a, **kw: (
        [{"start": 0, "end": 1, "text": text, "words": []}], "ko"))
    runner = CliRunner()
    args = [str(src), "--no-diarize", "--json"]
    failed = runner.invoke(cli.app, args + ["--encoding", "ascii"])
    assert failed.exit_code == 1
    assert set(tmp_path.iterdir()) == {src}

    retried = runner.invoke(cli.app, args)
    assert retried.exit_code == 0, retried.output
    assert "[완료]" in retried.output
    assert src.with_suffix(".txt").read_text(encoding="utf-8-sig").strip() == text
    assert text in src.with_suffix(".timeline.txt").read_text(encoding="utf-8-sig")
    payload = json.loads(src.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["utterances"][0]["text"] == text
