from pathlib import Path

import pytest

from stt import audio
from stt.cli import _outputs


@pytest.mark.parametrize("raw,expected", [
    (r"C:\Users\cyyoo\Recordings\회의.m4a", "/mnt/c/Users/cyyoo/Recordings/회의.m4a"),
    (r"D:\rec\a.m4a", "/mnt/d/rec/a.m4a"),
    ("C:/Users/cyyoo/a.m4a", "/mnt/c/Users/cyyoo/a.m4a"),
])
def test_windows_paths_become_wsl_paths(raw, expected):
    assert str(audio.normalize_path(raw)) == expected


def test_quotes_and_spaces_are_stripped():
    got = audio.normalize_path('  "C:\\Users\\cyyoo\\내 녹음\\회의 1.m4a"  ')
    assert str(got) == "/mnt/c/Users/cyyoo/내 녹음/회의 1.m4a"


def test_posix_paths_pass_through():
    assert str(audio.normalize_path("/mnt/c/rec/회의.m4a")) == "/mnt/c/rec/회의.m4a"
    assert str(audio.normalize_path("./회의.m4a")) == "회의.m4a"


def test_home_is_expanded():
    assert str(audio.normalize_path("~/rec/a.m4a")).startswith("/home/")


def test_outputs_land_next_to_the_input():
    src = Path("/mnt/c/Users/cyyoo/내 녹음/회의 1.m4a")
    out = _outputs(src, None, want_json=False)
    assert out["text"] == Path("/mnt/c/Users/cyyoo/내 녹음/회의 1.txt")
    assert out["timeline"] == Path("/mnt/c/Users/cyyoo/내 녹음/회의 1.timeline.txt")
    assert "json" not in out


def test_dotted_filename_keeps_its_stem():
    """'2026.09.09 회의.m4a' 처럼 점이 섞인 이름에서 확장자만 바뀌어야 한다."""
    out = _outputs(Path("/tmp/2026.09.09 회의.m4a"), None, want_json=True)
    assert out["text"].name == "2026.09.09 회의.txt"
    assert out["timeline"].name == "2026.09.09 회의.timeline.txt"
    assert out["json"].name == "2026.09.09 회의.json"


def test_out_dir_overrides_the_location(tmp_path):
    out = _outputs(Path("/mnt/c/rec/회의.m4a"), tmp_path, want_json=False)
    assert out["text"] == tmp_path / "회의.txt"


def test_read_wav_mono_rejects_stereo(tmp_path):
    import wave
    path = tmp_path / "s.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00" * 64)
    with pytest.raises(ValueError):
        audio.read_wav_mono(path)


def test_read_wav_mono_returns_float_samples(tmp_path):
    import struct
    import wave
    path = tmp_path / "m.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<3h", 0, 32767, -32768))
    samples, rate = audio.read_wav_mono(path)
    assert rate == 16000
    assert len(samples) == 3
    assert -1.01 <= float(samples.min()) and float(samples.max()) <= 1.0
