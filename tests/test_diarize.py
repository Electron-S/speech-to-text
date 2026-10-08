import numpy as np

from stt.diarize import SpeakerSpan, cluster_speakers, recluster

DIM = 64


def unit(v):
    return v / np.linalg.norm(v)


def voices(rng, center, count, noise=0.094):
    """center 주변 임베딩. noise=0.094면 중심과 코사인 ~0.8, 서로 간 거리 ~0.36."""
    return np.stack([center + rng.normal(0, noise, DIM) for _ in range(count)])


def test_long_meeting_noise_does_not_split_one_voice_into_many():
    # 1시간 회의처럼 한 사람당 구간이 수백 개면 튀는 쌍이 반드시 생긴다.
    # complete linkage는 여기서 수십 명으로 쪼갰고, average linkage는 두 명으로 남아야 한다.
    rng = np.random.default_rng(0)
    a, b = unit(rng.normal(size=DIM)), unit(rng.normal(size=DIM))
    X = np.vstack([voices(rng, a, 200), voices(rng, b, 200)])
    labels = cluster_speakers(X, np.full(400, 3.0))
    assert len(set(labels[:200])) == 1
    assert len(set(labels[200:])) == 1
    assert labels[0] != labels[200]


def test_voice_that_barely_spoke_is_absorbed_into_nearest_speaker():
    rng = np.random.default_rng(1)
    a, b = unit(rng.normal(size=DIM)), unit(rng.normal(size=DIM))
    # a 쪽에 가깝지만 따로 묶일 만큼은 먼 목소리가 6초만 말했다
    near_a = unit(0.4 * a + 0.92 * unit(rng.normal(size=DIM)))
    X = np.vstack([voices(rng, a, 100), voices(rng, b, 100), voices(rng, near_a, 3)])
    w = np.concatenate([np.full(200, 3.0), np.full(3, 2.0)])
    labels = cluster_speakers(X, w)
    assert len(set(labels)) == 2
    assert set(labels[200:]) == {labels[0]}


def test_short_recording_keeps_two_speakers_under_thirty_seconds():
    # 40초짜리 2인 대화: 30초 고정 기준이면 둘 다 지워져 한 명이 된다
    rng = np.random.default_rng(2)
    a, b = unit(rng.normal(size=DIM)), unit(rng.normal(size=DIM))
    X = np.vstack([voices(rng, a, 5), voices(rng, b, 5)])
    labels = cluster_speakers(X, np.full(10, 4.0))
    assert len(set(labels)) == 2


def test_labels_are_dense_from_zero():
    rng = np.random.default_rng(3)
    centers = [unit(rng.normal(size=DIM)) for _ in range(3)]
    X = np.vstack([voices(rng, c, 30) for c in centers])
    labels = cluster_speakers(X, np.full(90, 3.0))
    assert sorted(set(labels.tolist())) == [0, 1, 2]


def test_cluster_speakers_empty():
    assert len(cluster_speakers(np.zeros((0, DIM)), np.zeros(0))) == 0


def test_recluster_labels_short_spans_from_presplit_group_or_nearest_time():
    rng = np.random.default_rng(4)
    a, b = unit(rng.normal(size=DIM)), unit(rng.normal(size=DIM))
    va, vb = voices(rng, a, 2), voices(rng, b, 2)
    spans = [
        SpeakerSpan(0.0, 3.0, 7),    # a
        SpeakerSpan(3.0, 3.2, 7),    # 짧음: 1차 묶음 7 -> a
        SpeakerSpan(4.0, 7.0, 9),    # b
        SpeakerSpan(8.0, 11.0, 7),   # a
        SpeakerSpan(12.0, 15.0, 9),  # b
        SpeakerSpan(15.1, 15.3, 42), # 짧고 1차 묶음도 처음 보는 번호 -> 시간상 가까운 b
    ]
    emb = {0: va[0], 2: vb[0], 3: va[1], 4: vb[1]}
    out = recluster(spans, emb)
    who = [s.speaker for s in out]
    assert who[0] == who[1] == who[3]
    assert who[2] == who[4] == who[5]
    assert who[0] != who[2]
    assert [(s.start, s.end) for s in out] == [(s.start, s.end) for s in spans]


def test_recluster_without_embeddings_keeps_spans():
    spans = [SpeakerSpan(0.0, 0.3, 0)]
    assert recluster(spans, {}) == spans
