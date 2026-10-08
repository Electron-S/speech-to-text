"""sherpa-onnx 화자분리 (CPU/onnxruntime, HF 토큰 불필요).

autocut의 음높이(F0) 기반 화자 구분은 쓰지 않는다. 그 방식은 성인남↔아이처럼
한 옥타브 이상 벌어진 2인에게만 동작하고 3명이면 두 명으로 뭉개진다고
autocut/src/autocut/speakers.py 주석에 저자가 직접 적어뒀다. 회의 녹음에는 맞지 않는다.
여기서는 pyannote segmentation(ONNX) + 화자 임베딩 클러스터링으로 인원 수를 세지 않고도 나눈다.

인원을 자동 추정할 때 sherpa-onnx의 클러스터링을 그대로 쓰지 않는다. FastClustering은
complete linkage라서(두 무리 사이 *모든* 쌍이 기준 안이어야 합친다) 구간이 수백 개인
1시간 회의에서는 튀는 임베딩 하나가 합병을 막아 같은 사람이 계속 쪼개진다.
실측: 59분 회의가 화자 202명, 53분 회의가 130명. threshold를 올려 보면 1.1에서 19명,
1.2에서 6명, 1.3에서 2명으로 무너져서 쓸 만한 값이 없었다.
그래서 sherpa는 잘게 쪼갠 구간을 얻는 데만 쓰고, 구간마다 임베딩을 다시 뽑아
average linkage로 묶은 뒤 말한 양이 아주 적은 무리를 가장 가까운 화자에 흡수한다.
참석자 수를 아는 회의(5명, 6명)에서 정확히 맞고, 6~10명 규모 회의 두 건은 12명이 나왔다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import audio, models

# sherpa 1차 분할용. 일부러 잘게 쪼개서 한 구간에 두 사람이 섞이지 않게 한다.
PRESPLIT_THRESHOLD = 0.5
# 재클러스터링(average linkage, 코사인 거리) 기본값. 0.55보다 올리면 5명 회의에서
# 2.7분 말한 사람이 다른 사람에 합쳐지기 시작하고, 내리면 6명 회의가 7명이 된다.
DEFAULT_THRESHOLD = 0.55
# 이보다 짧은 구간은 임베딩이 불안정해서 클러스터링에 넣지 않고 나중에 라벨만 붙인다.
MIN_EMBED_S = 0.5
# 말한 총량이 이보다 적은 무리는 독립 화자로 치지 않는다. 1시간 회의에서는 30초,
# 짧은 녹음에서는 전체 말소리의 5%까지 낮아져서 1분짜리 2인 대화도 두 명으로 남는다.
MIN_SPEAKER_S = 30.0
MIN_SPEAKER_SHARE = 0.05


@dataclass(frozen=True)
class SpeakerSpan:
    start: float
    end: float
    speaker: int


def diarize(wav: Path, *, num_speakers: int = -1, threshold: float = DEFAULT_THRESHOLD,
            num_threads: int = 4, progress: bool = True) -> list[SpeakerSpan]:
    """화자 구간 목록. num_speakers<=0 이면 threshold로 인원을 자동 추정한다."""
    import sherpa_onnx

    fixed = bool(num_speakers and num_speakers > 0)
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(models.segmentation_model()),
            ),
            num_threads=num_threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(models.embedding_model()),
            num_threads=num_threads,
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers if fixed else -1,
            threshold=PRESPLIT_THRESHOLD,
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise RuntimeError("화자분리 설정이 유효하지 않습니다(모델 경로 확인)")

    sd = sherpa_onnx.OfflineSpeakerDiarization(config)
    samples, rate = audio.read_wav_mono(wav)
    if rate != sd.sample_rate:
        raise ValueError(f"화자분리 모델은 {sd.sample_rate}Hz를 요구합니다(입력 {rate}Hz)")

    last = [-1]

    def on_progress(done: int, total: int) -> int:
        if progress and total:
            pct = int(done * 100 / total)
            if pct >= last[0] + 10:
                last[0] = pct
                print(f"[화자] {pct}%", flush=True)
        return 0

    result = sd.process(samples, callback=on_progress if progress else None)
    spans = [
        SpeakerSpan(start=round(s.start, 3), end=round(s.end, 3), speaker=int(s.speaker))
        for s in result.sort_by_start_time()
    ]
    if fixed or not spans:
        # 인원을 알려주면 그 수로 자른 sherpa 결과를 그대로 쓴다(사용자가 검증해 온 경로)
        return spans

    if progress:
        print(f"[화자] 1차 {len({s.speaker for s in spans})}개 묶음 -> 재분류 중", flush=True)
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(models.embedding_model()), num_threads=num_threads))
    return recluster(spans, _embed_spans(extractor, samples, rate, spans), threshold=threshold)


def _embed_spans(extractor, samples, rate: int, spans: list[SpeakerSpan]) -> dict:
    """MIN_EMBED_S 이상인 구간의 화자 임베딩 {구간 번호: 벡터}."""
    import numpy as np

    out = {}
    for i, s in enumerate(spans):
        a, b = int(s.start * rate), int(s.end * rate)
        if b - a < int(MIN_EMBED_S * rate):
            continue
        stream = extractor.create_stream()
        stream.accept_waveform(rate, samples[a:b])
        stream.input_finished()
        if extractor.is_ready(stream):
            out[i] = np.asarray(extractor.compute(stream), dtype=np.float64)
    return out


def recluster(spans: list[SpeakerSpan], embeddings: dict, *,
              threshold: float = DEFAULT_THRESHOLD) -> list[SpeakerSpan]:
    """구간 임베딩으로 화자를 다시 묶고, 임베딩이 없는 짧은 구간에도 라벨을 붙인다."""
    import numpy as np

    if not embeddings:
        return spans
    idx = sorted(embeddings)
    X = np.stack([embeddings[i] for i in idx])
    weights = np.array([spans[i].end - spans[i].start for i in idx])
    labels = dict(zip(idx, cluster_speakers(X, weights, threshold=threshold).tolist()))

    # 짧은 구간은 1차 분할에서 같은 묶음이던 구간들의 다수결(말한 길이 기준)을 따르고,
    # 그 묶음에 임베딩 있는 구간이 하나도 없으면 시간상 가장 가까운 구간을 따른다.
    votes: dict[int, dict[int, float]] = {}
    for i, label in labels.items():
        tally = votes.setdefault(spans[i].speaker, {})
        tally[label] = tally.get(label, 0.0) + spans[i].end - spans[i].start
    mids = np.array([(spans[i].start + spans[i].end) / 2 for i in idx])

    out = []
    for i, s in enumerate(spans):
        if i in labels:
            label = labels[i]
        elif s.speaker in votes:
            label = max(votes[s.speaker].items(), key=lambda kv: kv[1])[0]
        else:
            label = labels[idx[int(np.argmin(np.abs(mids - (s.start + s.end) / 2)))]]
        out.append(SpeakerSpan(s.start, s.end, label))
    return out


def cluster_speakers(X, weights, *, threshold: float = DEFAULT_THRESHOLD,
                     min_speaker_s: float = MIN_SPEAKER_S,
                     min_share: float = MIN_SPEAKER_SHARE):
    """임베딩 행렬 -> 0부터 시작하는 화자 번호 배열.

    average linkage(구간 길이 가중) 코사인 거리로 threshold까지 합친 뒤, 말한 총량이
    min(min_speaker_s, 전체의 min_share) 미만인 무리는 가장 가까운 큰 무리에 붙인다.
    """
    import numpy as np

    X = np.asarray(X, dtype=np.float64)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    w = np.asarray(weights, dtype=np.float64)
    n = len(X)
    if n == 0:
        return np.zeros(0, dtype=int)

    # Lance-Williams 갱신으로 무리 간 평균 거리를 유지한다. 비활성 행/열은 inf로 막는다.
    D = 1.0 - X @ X.T
    np.fill_diagonal(D, np.inf)
    size = w.copy()
    root = np.arange(n)
    for _ in range(n - 1):
        k = int(np.argmin(D))
        i, j = divmod(k, n)
        if D[i, j] > threshold:
            break
        merged = (size[i] * D[i] + size[j] * D[j]) / (size[i] + size[j])
        D[i], D[:, i] = merged, merged
        D[i, i] = np.inf
        D[j], D[:, j] = np.inf, np.inf
        size[i] += size[j]
        root[root == j] = i
    _, labels = np.unique(root, return_inverse=True)

    totals = np.bincount(labels, weights=w)
    floor = min(min_speaker_s, min_share * w.sum())
    big = np.flatnonzero(totals >= floor)
    if len(big) == 0:  # 아무도 기준을 못 넘으면 가장 많이 말한 무리 하나만 남긴다
        big = np.array([int(np.argmax(totals))])
    centroids = np.stack([(X[labels == c] * w[labels == c, None]).sum(0) for c in big])
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    small = ~np.isin(labels, big)
    labels[small] = big[np.argmax(X[small] @ centroids.T, axis=1)]
    _, labels = np.unique(labels, return_inverse=True)
    return labels
