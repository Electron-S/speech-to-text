from dataclasses import dataclass

import pytest

from stt.merge import SpeakerIndex, build_utterances, speaker_count


@dataclass(frozen=True)
class Span:
    start: float
    end: float
    speaker: int


def seg(start, end, text, words=None):
    return {"start": start, "end": end, "text": text, "words": words or []}


def word(start, end, text):
    return {"start": start, "end": end, "word": text}


def test_speaker_index_picks_largest_overlap():
    idx = SpeakerIndex([Span(0, 10, 0), Span(10, 20, 1)])
    assert idx.at(1, 2) == 0
    assert idx.at(11, 12) == 1
    # 경계에 걸친 단어는 더 많이 겹치는 쪽으로
    assert idx.at(9.0, 10.4) == 0
    assert idx.at(9.6, 11.0) == 1


def test_speaker_index_outside_any_span_is_unknown():
    idx = SpeakerIndex([Span(5, 6, 0)])
    assert idx.at(0, 1) is None
    assert idx.at(100, 101) is None
    assert SpeakerIndex([]).at(0, 1) is None


@pytest.mark.parametrize("start,end,expected", [
    (5, 6, 0),  # 짧은 중첩 구간이 끝나도 긴 구간은 계속 유효하다
    (5, 5, 0),  # 길이 0인 단어도 동일하게 처리한다
    (1, 2, 0),  # 겹침이 같으면 작은 화자 번호를 선택한다
    (10, 11, None),  # 끝점은 구간에 포함되지 않는다
    (-1, 0, None),  # 시작점에 닿기만 하는 단어는 겹치지 않는다
])
def test_nested_speaker_spans_keep_the_containing_speaker(start, end, expected):
    idx = SpeakerIndex([Span(3, 4, 1), Span(0, 10, 0), Span(1, 2, 1)])
    assert idx.at(start, end) == expected


def test_overlapping_spans_still_choose_largest_total_overlap():
    idx = SpeakerIndex([Span(0, 10, 0), Span(1, 2, 2), Span(3, 4, 2),
                        Span(8, 11, 1), Span(11, 13, 1)])
    assert idx.at(9, 12) == 1


def test_nested_spans_preserve_speaker_changes_within_a_segment():
    segments = [seg(5, 12, "계속 말합니다 다음 화자입니다", [
        word(5, 6, "계속 말합니다"), word(11, 12, " 다음 화자입니다"),
    ])]
    spans = [Span(0, 10, 7), Span(1, 2, 2), Span(3, 4, 2), Span(10, 13, 2)]
    out = build_utterances(segments, spans)
    assert [(u.start, u.end, u.speaker, u.text) for u in out] == [
        (5, 6, 0, "계속 말합니다"), (11, 12, 1, "다음 화자입니다"),
    ]


def test_zero_length_word_still_matches_containing_span():
    idx = SpeakerIndex([Span(0, 10, 3)])
    assert idx.at(4.0, 4.0) == 3


def test_single_speaker_segment_keeps_whisper_sentence():
    """한 화자뿐인 세그먼트는 단어를 다시 이어붙이지 않는다 (안녕/하세요 조각 문제)."""
    segments = [seg(0, 2, "안녕하세요 반갑습니다",
                    [word(0, 1, "안녕"), word(1, 1.5, "하세요"), word(1.5, 2, " 반갑습니다")])]
    out = build_utterances(segments, [Span(0, 5, 0)])
    assert len(out) == 1
    assert out[0].text == "안녕하세요 반갑습니다"
    assert out[0].speaker == 0


def test_segment_with_two_speakers_is_split_at_the_turn():
    segments = [seg(0, 4, "네 그렇습니다 아니요 다릅니다", [
        word(0.0, 0.5, "네"), word(0.5, 1.5, " 그렇습니다"),
        word(2.0, 2.5, " 아니요"), word(2.5, 3.5, " 다릅니다"),
    ])]
    spans = [Span(0, 1.8, 0), Span(1.9, 4.0, 1)]
    out = build_utterances(segments, spans)
    assert [(u.speaker, u.text) for u in out] == [
        (0, "네 그렇습니다"),
        (1, "아니요 다릅니다"),
    ]


def test_same_speaker_short_gap_merges_across_segments():
    segments = [seg(0, 2, "첫 문장입니다"), seg(2.5, 4, "이어서 말합니다")]
    out = build_utterances(segments, [Span(0, 5, 0)])
    assert len(out) == 1
    assert out[0].text == "첫 문장입니다 이어서 말합니다"
    assert (out[0].start, out[0].end) == (0, 4)


def test_long_silence_breaks_utterance_even_for_same_speaker():
    segments = [seg(0, 2, "먼저"), seg(30, 31, "한참 뒤에")]
    out = build_utterances(segments, [Span(0, 60, 0)])
    assert [u.text for u in out] == ["먼저", "한참 뒤에"]


def test_speakers_are_renumbered_by_first_appearance():
    """클러스터 번호는 발화 순서와 무관하므로 먼저 말한 사람이 화자1이 되게 한다."""
    segments = [seg(0, 1, "제가 먼저"), seg(5, 6, "제가 나중")]
    spans = [Span(0, 2, 7), Span(4, 7, 2)]
    out = build_utterances(segments, spans)
    assert [u.speaker for u in out] == [0, 1]


def test_unknown_speaker_inherits_from_previous_utterance():
    segments = [seg(0, 1, "처음"), seg(5, 6, "구간 밖")]
    out = build_utterances(segments, [Span(0, 1.5, 0)])
    assert [(u.speaker, u.text) for u in out] == [(0, "처음"), (0, "구간 밖")]


def test_leading_unknown_borrows_from_the_next_known_speaker():
    segments = [seg(0, 1, "구간 밖"), seg(5, 6, "구간 안")]
    out = build_utterances(segments, [Span(4, 7, 0)])
    assert [u.speaker for u in out] == [0, 0]


def test_no_diarization_leaves_speaker_none():
    segments = [seg(0, 1, "혼자 말한 메모")]
    out = build_utterances(segments, [])
    assert out[0].speaker is None
    assert speaker_count(out) == 0


def test_empty_and_whitespace_segments_are_dropped():
    segments = [seg(0, 1, "   "), seg(1, 2, "실제 내용"), seg(2, 3, "")]
    out = build_utterances(segments, [])
    assert [u.text for u in out] == ["실제 내용"]


def test_long_monologue_is_broken_up():
    segments = [seg(i, i + 1, "가" * 120) for i in range(10)]
    out = build_utterances(segments, [Span(0, 20, 0)], max_chars=300)
    assert len(out) > 1
    assert all(len(u.text) <= 500 for u in out)


def test_speaker_count_counts_distinct_speakers():
    segments = [seg(0, 1, "가"), seg(5, 6, "나"), seg(10, 11, "다")]
    spans = [Span(0, 2, 0), Span(4, 7, 1), Span(9, 12, 0)]
    out = build_utterances(segments, spans)
    assert speaker_count(out) == 2


def words_of(*items):
    return [word(s, e, t) for s, e, t in items]


def seg_of(words):
    return seg(words[0]["start"], words[-1]["end"], "".join(w["word"] for w in words).strip(), words)


def test_overlaps_reports_time_per_speaker():
    idx = SpeakerIndex([Span(0, 2, 0), Span(1, 3, 1)])
    assert idx.overlaps(1.5, 2.5) == {0: 0.5, 1: 1.0}
    assert idx.overlaps(5, 6) == {}


# 아래 시각은 0928 AX 회의 실제 전사·화자분리 결과에서 그대로 옮겼다.

def test_word_in_turn_overlap_stays_with_its_phrase():
    """'…거고 그' / '내용입니다.' 처럼 구 끝 단어 하나만 다음 사람에게 넘어가지 않는다."""
    words = words_of((446.78, 447.14, " 관련된"), (447.14, 447.42, " 거고"),
                     (447.42, 447.66, " 그"), (447.66, 449.16, " 내용입니다."),
                     (449.54, 450.22, " 저는"), (450.22, 450.50, " 그냥"))
    spans = [Span(443.02, 448.64, 2), Span(447.62, 449.14, 0), Span(450.10, 451.91, 2)]
    out = build_utterances([seg_of(words)], spans)
    assert [(u.speaker, u.text) for u in out] == [(0, "관련된 거고 그 내용입니다. 저는 그냥")]


def test_squeezed_words_in_three_way_overlap_finish_the_sentence():
    """whisper가 '될 것 같아요.'를 0.16초에 몰아넣고 화자 구간 셋이 겹친 자리."""
    words = words_of((859.96, 860.42, " 계정"), (860.42, 860.84, " 탈퇴를"),
                     (860.84, 861.18, " 하든"), (861.18, 861.80, " 그렇게"),
                     (861.80, 862.28, " 시켜야"), (862.28, 862.34, " 될"),
                     (862.34, 862.44, " 것"), (862.44, 862.44, " 같아요."),
                     (862.44, 863.14, " 그래야"), (863.14, 863.28, " 될"),
                     (863.28, 863.86, " 수도"), (863.86, 864.08, " 있을"),
                     (864.08, 864.20, " 것"), (864.20, 864.22, " 같아요."))
    spans = [Span(857.37, 862.93, 4), Span(861.97, 862.56, 2),
             Span(862.29, 864.45, 0), Span(863.69, 871.32, 4)]
    out = build_utterances([seg_of(words)], spans)
    assert [(u.speaker, u.text) for u in out] == [
        (0, "계정 탈퇴를 하든 그렇게 시켜야 될 것 같아요."),
        (1, "그래야 될 수도 있을 것 같아요."),
    ]


def test_short_interjection_without_overlap_evidence_is_kept():
    """'자동으로 돼?'는 앞사람 구간과 전혀 안 겹치는 실제 끼어들기라 흡수하지 않는다."""
    words = words_of((685.65, 686.05, " 자동으로"), (686.05, 686.25, " 됩니다"),
                     (686.25, 686.63, " 아"), (686.63, 687.07, " 자동으로"),
                     (687.07, 687.23, " 돼?"), (687.29, 687.45, " 네"),
                     (687.45, 688.35, " 그전에"))
    spans = [Span(684.35, 686.40, 0), Span(686.66, 687.37, 2), Span(687.37, 690.76, 0)]
    out = build_utterances([seg_of(words)], spans)
    assert [(u.speaker, u.text) for u in out] == [
        (0, "자동으로 됩니다 아"), (1, "자동으로 돼?"), (0, "네 그전에"),
    ]


def test_word_shared_evenly_joins_the_question_it_belongs_to():
    """'스킬을'은 두 화자에 반반 걸쳐 있어 '등록했어요?' 쪽으로 붙는다."""
    words = words_of((2403.88, 2404.14, " 있습니다."), (2406.22, 2406.98, " 스킬을"),
                     (2406.98, 2407.58, " 등록했어요?"), (2408.36, 2409.16, " 배포는"),
                     (2409.16, 2409.30, " 안"), (2409.30, 2409.56, " 했고요."))
    spans = [Span(2400.43, 2404.68, 10), Span(2406.56, 2407.01, 10),
             Span(2406.57, 2408.67, 9), Span(2408.51, 2416.03, 10)]
    out = build_utterances([seg_of(words)], spans)
    assert [(u.speaker, u.text) for u in out] == [
        (0, "있습니다."), (1, "스킬을 등록했어요?"), (0, "배포는 안 했고요."),
    ]


def test_real_turn_without_punctuation_or_pause_is_kept():
    """whisper가 문장부호를 빠뜨려도 양쪽이 충분히 길면 실제 화자 전환으로 둔다."""
    words = words_of((0.0, 0.5, " 그래서"), (0.5, 1.0, " 이번에"), (1.0, 1.5, " 이렇게"),
                     (1.5, 2.0, " 했고"), (2.0, 2.5, " 그럼"), (2.5, 3.0, " 다음"),
                     (3.0, 3.5, " 주에"), (3.5, 4.0, " 보시죠"))
    spans = [Span(0.0, 2.0, 0), Span(2.0, 4.0, 1)]
    out = build_utterances([seg_of(words)], spans)
    assert [(u.speaker, u.text) for u in out] == [
        (0, "그래서 이번에 이렇게 했고"), (1, "그럼 다음 주에 보시죠"),
    ]
