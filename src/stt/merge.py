"""whisper 세그먼트 + 화자 구간 -> 화자별 발화 목록.

whisper는 한 세그먼트 안에 두 사람 말을 붙여놓는 일이 흔하다. 그래서 세그먼트
단위로 화자를 붙이지 않고 단어 타임스탬프마다 화자를 찾은 뒤, 화자가 바뀌는
지점에서 세그먼트를 쪼갠다.

텍스트는 가능하면 whisper가 만든 세그먼트 문장을 그대로 쓴다. 단어 토큰을 다시
이어붙이면 한국어에서 '안녕' + '하세요' 처럼 어절 중간이 잘린 조각들이 나와,
공백을 넣으면 '안녕 하세요'가 되고 안 넣으면 문장 사이가 붙어버린다. 그래서
한 세그먼트가 한 화자면 세그먼트 문장을 쓰고, 여러 화자가 섞인 세그먼트만
단어 토큰을 이어붙여 나눈다.

단어마다 화자를 따로 정하면 말이 넘어가는 자리에서 단어 한두 개가 옆 사람에게
붙는다('…관련된 거고 그' / '내용입니다.'). 화자분리 구간은 전환 지점에서 서로
겹치고, whisper는 '될 것 같아요.'를 0.16초에 몰아넣기도 해서 그 단어들의 근거가
양쪽에 반반 걸리기 때문이다. 그래서 구두점·쉼 사이(구)에 낀 두 단어 이하 조각은,
옆 화자 구간과도 충분히 겹칠 때만 옆 화자에게 넘긴다(`_smooth_turns`).
겹침이 없는 짧은 끼어들기('자동으로 돼?')는 실제 발언이므로 남긴다.
"""
from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass
from itertools import accumulate


@dataclass(frozen=True)
class Utterance:
    start: float
    end: float
    speaker: int | None  # 0-based. 화자분리를 끄면 None
    text: str


class SpeakerIndex:
    """정렬된 화자 구간에서 [start, end) 와 가장 많이 겹치는 화자를 찾는다."""

    def __init__(self, spans) -> None:
        self.spans = sorted(spans, key=lambda s: (s.start, s.end))
        # 시작 순으로 정렬해도 중첩 구간의 종료 시간은 정렬되지 않는다.
        # 누적 최댓값은 항상 단조 증가하므로 이진 탐색에 사용할 수 있다.
        self.max_ends = list(accumulate((s.end for s in self.spans), max))

    def overlaps(self, start: float, end: float) -> dict[int, float]:
        """화자별로 [start, end) 와 겹치는 시간(초)."""
        if end <= start:  # 길이 0인 단어는 그 시점을 포함하는 구간으로 본다
            end = start + 1e-3
        overlap: dict[int, float] = {}
        # 앞선 모든 구간이 단어 시작 전에 끝난 경우에만 건너뛴다.
        i = bisect_right(self.max_ends, start)
        for j in range(i, len(self.spans)):
            span = self.spans[j]
            if span.start >= end:
                break
            hit = min(span.end, end) - max(span.start, start)
            if hit > 0:
                overlap[span.speaker] = overlap.get(span.speaker, 0.0) + hit
        return overlap

    def at(self, start: float, end: float) -> int | None:
        return _best(self.overlaps(start, end))


def _best(overlap: dict[int, float]) -> int | None:
    if not overlap:
        return None
    return max(overlap.items(), key=lambda kv: (kv[1], -kv[0]))[0]


# 화자 전환으로 자연스러운 자리: 문장부호 뒤, 또는 이만큼 이상 쉰 곳.
# 0928 회의 단어 간격의 90%가 0.08초 이하라 0.3초면 확실한 쉼이다.
TURN_PAUSE_S = 0.3
_SENTENCE_END = re.compile(r"[.?!]\s*$")
# 구 안에서 이 단어 수 이하인 조각만 옆 화자에게 넘길 후보가 된다.
# 3으로 늘리면 '아직은 정답이 없죠' 같은 실제 끼어들기까지 지워졌다.
SHORT_RUN_WORDS = 2
# 옆 화자 구간과 겹친 시간이 원래 화자의 이 비율 이상이어야 넘긴다.
# 0.15로 낮추면 '네, 알겠습니다.' 같은 남의 대답이 질문자에게 흡수됐다.
MIN_EVIDENCE_RATIO = 0.3


def _turn_ok(words: list[dict], k: int) -> bool:
    """words[k-1] 과 words[k] 사이가 화자가 바뀌어도 자연스러운 자리인가."""
    if k <= 0 or k >= len(words):
        return True
    gap = float(words[k]["start"]) - float(words[k - 1]["end"])
    return bool(_SENTENCE_END.search(words[k - 1]["word"])) or gap >= TURN_PAUSE_S


def _smooth_turns(words: list[dict], labels: list, evidence: list[dict[int, float]]) -> list:
    """구(자연스러운 전환 자리 사이) 안의 짧은 화자 조각을 근거가 있으면 옆 화자에 붙인다.

    후보가 여럿이면 원래 화자 근거를 가장 적게 잃는 것부터 옮기고, 조각들이 합쳐지면
    다시 본다. 근거가 모자라면 실제 짧은 발언으로 보고 그대로 둔다.
    """
    labels = list(labels)
    cuts = [k for k in range(len(words) + 1) if _turn_ok(words, k)]
    for lo, hi in zip(cuts, cuts[1:]):
        while True:
            runs: list[list[int]] = []
            for k in range(lo, hi):
                if runs and labels[runs[-1][1] - 1] == labels[k]:
                    runs[-1][1] = k + 1
                else:
                    runs.append([k, k + 1])
            best = None
            for r, (a, b) in enumerate(runs):
                if b - a > SHORT_RUN_WORDS:
                    continue
                own = sum(evidence[k].get(labels[a], 0.0) for k in range(a, b))
                for nb in (r - 1, r + 1):
                    if not 0 <= nb < len(runs):
                        continue
                    target = labels[runs[nb][0]]
                    got = sum(evidence[k].get(target, 0.0) for k in range(a, b))
                    if got >= MIN_EVIDENCE_RATIO * own and (best is None or own - got < best[0]):
                        best = (own - got, a, b, target)
            if best is None:
                break
            _, a, b, target = best
            labels[a:b] = [target] * (b - a)
    return labels


@dataclass
class _Part:
    start: float
    end: float
    speaker: int | None
    text: str
    seg_index: int


def _split_segment(seg: dict, index: SpeakerIndex, seg_index: int) -> list[_Part]:
    text = (seg.get("text") or "").strip()
    words = [w for w in seg.get("words") or [] if (w.get("word") or "").strip()]
    start, end = float(seg["start"]), float(seg["end"])

    if not words:
        if not text:
            return []
        return [_Part(start, end, index.at(start, end), text, seg_index)]

    evidence = [index.overlaps(float(w["start"]), float(w["end"])) for w in words]
    labels = [_best(e) for e in evidence]

    if len({s for s in labels if s is not None}) > 1:
        # 화자가 섞인 세그먼트: 미지정 단어는 직전 화자에 붙여 조각이 잘게 갈리는 것을 막는다
        filled: list[int | None] = []
        for label in labels:
            filled.append(label if label is not None else (filled[-1] if filled else None))
        for i in range(len(filled) - 1, -1, -1):
            if filled[i] is None:
                filled[i] = next((s for s in filled[i + 1:] if s is not None), None)
        labels = _smooth_turns(words, filled, evidence)

    known = {s for s in labels if s is not None}
    if len(known) <= 1:
        # 한 화자(또는 화자 미지정)뿐이면 whisper 문장을 그대로 쓴다
        speaker = next(iter(known)) if known else None
        return [_Part(start, end, speaker, text or "".join(w["word"] for w in words).strip(),
                      seg_index)]

    parts: list[_Part] = []
    run_start = 0
    for i in range(1, len(words) + 1):
        if i == len(words) or labels[i] != labels[run_start]:
            chunk = words[run_start:i]
            chunk_text = "".join(w["word"] for w in chunk).strip()
            if chunk_text:
                parts.append(_Part(
                    start=float(chunk[0]["start"]),
                    end=float(chunk[-1]["end"]),
                    speaker=labels[run_start],
                    text=chunk_text,
                    seg_index=seg_index,
                ))
            run_start = i
    return parts


def build_utterances(segments, spans, *, merge_gap: float = 2.0,
                     max_chars: int = 500) -> list[Utterance]:
    """세그먼트와 화자 구간을 합쳐 발화 목록을 만든다.

    같은 화자가 merge_gap 미만 간격으로 이어 말하면 한 발화로 합치고,
    화자가 바뀌거나 침묵이 길거나 한 발화가 max_chars를 넘으면 끊는다.
    """
    index = SpeakerIndex(spans or [])

    parts: list[_Part] = []
    for i, seg in enumerate(segments):
        parts.extend(_split_segment(seg, index, i))
    if not parts:
        return []

    # 앞뒤 문맥으로 화자 미지정 조각을 채운다(화자분리를 껐으면 전부 None으로 남는다)
    for i, part in enumerate(parts):
        if part.speaker is None:
            prev = next((p.speaker for p in reversed(parts[:i]) if p.speaker is not None), None)
            part.speaker = prev if prev is not None else next(
                (p.speaker for p in parts[i + 1:] if p.speaker is not None), None)

    # 먼저 말한 사람이 화자1이 되도록 번호를 다시 매긴다(클러스터 번호는 순서와 무관)
    order: dict[int, int] = {}
    for part in parts:
        if part.speaker is not None and part.speaker not in order:
            order[part.speaker] = len(order)

    out: list[Utterance] = []
    buf: list[_Part] = []

    def flush() -> None:
        if not buf:
            return
        text = " ".join(p.text for p in buf).strip()
        if text:
            speaker = buf[0].speaker
            out.append(Utterance(
                start=buf[0].start,
                end=buf[-1].end,
                speaker=order[speaker] if speaker is not None else None,
                text=text,
            ))
        buf.clear()

    for part in parts:
        if buf:
            prev = buf[-1]
            chars = sum(len(p.text) + 1 for p in buf)
            if (part.speaker != prev.speaker
                    or part.start - prev.end >= merge_gap
                    or chars >= max_chars):
                flush()
        buf.append(part)
    flush()
    return out


def speaker_count(utterances) -> int:
    return len({u.speaker for u in utterances if u.speaker is not None})
