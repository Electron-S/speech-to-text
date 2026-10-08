from pathlib import Path

import pytest

from stt import render
from stt.merge import Utterance


def test_fmt_time_uses_hours_minutes_seconds():
    assert render.fmt_time(0) == "00:00:00"
    assert render.fmt_time(83) == "00:01:23"
    assert render.fmt_time(3661.9) == "01:01:01"


def test_speaker_label_is_one_based():
    assert render.speaker_label(0) == "화자1"
    assert render.speaker_label(2) == "화자3"
    assert render.speaker_label(None) == ""


def test_timeline_has_time_and_speaker():
    ut = [Utterance(0, 2, 0, "안녕하세요"), Utterance(83, 90, 1, "네 반갑습니다")]
    out = render.timeline_text(ut)
    assert out.splitlines() == ["[00:00:00] 화자1: 안녕하세요", "[00:01:23] 화자2: 네 반갑습니다"]


def test_timeline_without_speakers_omits_the_label():
    out = render.timeline_text([Utterance(5, 6, None, "혼자 말한 메모")])
    assert out.strip() == "[00:00:05] 혼자 말한 메모"


def test_timeline_header_goes_first():
    out = render.timeline_text([Utterance(0, 1, 0, "가")], header="# 회의.m4a")
    assert out.splitlines()[0] == "# 회의.m4a"


def test_plain_text_breaks_paragraph_on_speaker_change():
    ut = [Utterance(0, 2, 0, "제 얘기입니다"), Utterance(2.1, 4, 1, "제 답입니다")]
    assert render.plain_text(ut) == "제 얘기입니다\n\n제 답입니다\n"


def test_plain_text_joins_same_speaker_within_a_paragraph():
    ut = [Utterance(0, 2, 0, "첫 문장."), Utterance(2.2, 4, 0, "둘째 문장.")]
    assert render.plain_text(ut) == "첫 문장. 둘째 문장.\n"


def test_plain_text_breaks_paragraph_after_a_long_silence():
    ut = [Utterance(0, 2, 0, "먼저"), Utterance(20, 21, 0, "한참 뒤")]
    assert render.plain_text(ut) == "먼저\n\n한참 뒤\n"


def test_plain_text_has_no_timestamps_or_labels():
    out = render.plain_text([Utterance(83, 90, 1, "본문만 있어야 한다")])
    assert "[" not in out and "화자" not in out


def test_written_txt_defaults_to_utf8_with_bom(tmp_path):
    """Windows 메모장이 cp949로 오해하지 않도록 기본은 BOM 있는 UTF-8."""
    path = render.write_text(tmp_path / "결과.txt", "한글 본문\n")
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert path.read_text(encoding="utf-8-sig") == "한글 본문\n"


def test_encoding_can_be_switched_off_bom(tmp_path):
    path = render.write_text(tmp_path / "결과.txt", "한글\n", encoding="utf-8")
    assert not path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_written_txt_uses_lf_even_though_python_default_is_platform_dependent(tmp_path):
    path = render.write_text(tmp_path / "a.txt", "가\n나\n")
    assert b"\r\n" not in path.read_bytes()


def test_json_payload_keeps_korean_readable_and_has_no_bom(tmp_path):
    ut = [Utterance(0, 2, 0, "한글 발화")]
    payload = render.json_payload(ut, source=tmp_path / "회의.m4a", duration=2.0,
                                  model="large-v3", language="ko")
    assert payload["speakers"] == 1
    assert payload["utterances"][0]["speaker_label"] == "화자1"
    path = render.write_json(tmp_path / "회의.json", payload)
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert "한글 발화".encode() in raw


def test_json_speaker_count_zero_without_diarization(tmp_path):
    payload = render.json_payload([Utterance(0, 1, None, "가")], source=tmp_path / "a.m4a",
                                  duration=1.0, model="small", language="ko")
    assert payload["speakers"] == 0
    assert payload["utterances"][0]["speaker_label"] is None


def test_paragraph_gap_is_configurable():
    ut = [Utterance(0, 2, 0, "먼저"), Utterance(4, 5, 0, "조금 뒤")]
    assert render.plain_text(ut, paragraph_gap=1.0) == "먼저\n\n조금 뒤\n"
    assert render.plain_text(ut, paragraph_gap=5.0) == "먼저 조금 뒤\n"


@pytest.mark.parametrize("existing", [False, True])
def test_failed_encoding_preserves_previous_state(tmp_path, existing):
    path = tmp_path / "결과.txt"
    if existing:
        path.write_bytes(b"previous")
    with pytest.raises(UnicodeEncodeError):
        render.write_text(path, "한글", encoding="ascii")
    assert path.exists() is existing
    if existing:
        assert path.read_bytes() == b"previous"
    assert set(tmp_path.iterdir()) == ({path} if existing else set())


@pytest.mark.parametrize("existing", [False, True])
def test_later_replace_failure_rolls_back_all_outputs(tmp_path, monkeypatch, existing):
    paths = [tmp_path / name for name in ("회의.txt", "회의.timeline.txt", "회의.json")]
    if existing:
        for path in paths:
            path.write_bytes(b"previous")
    replace = Path.replace

    def fail_last_output(self, target):
        if target == paths[-1]:
            raise OSError("simulated output failure")
        return replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_last_output)
    with pytest.raises(OSError, match="simulated output failure"):
        render.write_outputs({p: ("new", "utf-8") for p in paths})
    assert set(tmp_path.iterdir()) == (set(paths) if existing else set())
    if existing:
        assert all(p.read_bytes() == b"previous" for p in paths)
