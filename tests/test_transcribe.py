from stt.transcribe import _plans


def test_auto_tries_gpu_batched_then_gpu_single_then_cpu():
    assert _plans("auto", 4, vad=True) == [("cuda", 4), ("cuda", 1), ("cpu", 1)]


def test_vad_off_never_uses_the_batched_pipeline():
    """배치 파이프라인은 VAD 구간 없이는 'No clip timestamps found'로 반드시 실패한다."""
    assert _plans("auto", 4, vad=False) == [("cuda", 1), ("cpu", 1)]
    assert _plans("cuda", 8, vad=False) == [("cuda", 1)]


def test_explicit_cpu_does_not_touch_the_gpu():
    assert _plans("cpu", 4, vad=True) == [("cpu", 1)]


def test_explicit_cuda_never_falls_back_to_cpu():
    """--device cuda 는 GPU로 돌리겠다는 뜻이니 조용히 CPU로 내려가지 않는다."""
    assert _plans("cuda", 4, vad=True) == [("cuda", 4), ("cuda", 1)]


def test_batch_one_skips_the_batched_attempt():
    assert _plans("auto", 1, vad=True) == [("cuda", 1), ("cpu", 1)]


def test_zero_or_negative_batch_is_treated_as_one():
    assert _plans("auto", 0, vad=True) == [("cuda", 1), ("cpu", 1)]
