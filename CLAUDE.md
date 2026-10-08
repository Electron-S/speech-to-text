# speech-to-text

m4a 녹음 → 한국어 텍스트(화자 구분 + 타임스탬프). 결과는 입력 파일과 같은 폴더에 쓴다.
사용법은 README.md, 실행은 `.venv/bin/stt <경로>`, 테스트는 `.venv/bin/python -m pytest`.

## 파이프라인

`cli.py` → `audio.py`(경로 정규화 + 16kHz mono wav) → `transcribe.py`(faster-whisper)
→ `diarize.py`(sherpa-onnx 화자 구간) → `merge.py`(단어↔화자 정렬) → `render.py`(txt/json)

## 이 프로젝트에서 이미 판단이 끝난 것들

**화자 구분은 sherpa-onnx로 한다.** autocut의 음높이(F0) 기반 방식(`autocut/src/autocut/speakers.py`)을
쓰지 않는 이유는 그 파일 주석에 저자가 직접 적어놨다 — 2인 전용이고, 성인남↔아이처럼 옥타브가
벌어져야 동작하고, 3명이면 두 명으로 뭉개진다. pyannote 본체는 HF 게이트 모델이라 토큰이 필요해서
빼놨다. 검증: TTS로 만든 2인 대화(정답 있음)에서 화자 순서 5/5, 시간 오차 0.1초 이내.

**단, 인원 자동 추정은 sherpa 클러스터링을 쓰지 않는다** (`diarize.recluster`). sherpa의
FastClustering은 complete linkage라 1시간 회의에서 같은 사람을 계속 쪼갰다(59분 → 202명,
53분 → 130명). TTS 2인 대화는 짧고 깨끗해서 이 문제가 안 드러났다. sherpa는 잘게 쪼갠
구간만 얻고, 구간 임베딩을 average linkage(0.55)로 다시 묶은 뒤 말한 양이
min(30초, 전체의 5%) 미만인 무리를 가까운 화자에 흡수한다. 실측: 참석자 5명·6명 회의를
정확히 맞히고, 6~10명 규모 AX 회의 두 건은 12명이 나왔다. 0.57부터는 5명 회의에서
실제 한 명이 사라진다. `--speakers N` 경로는 sherpa 결과를 그대로 쓴다.

**CUDA 라이브러리는 직접 preload 해야 한다** (`transcribe.py:_preload_cuda_libs`).
torch가 없는 venv에서는 ctranslate2가 `libcublas.so.12 is not found` 로 GPU를 통째로 못 쓰고
CPU로 폴백해 몇 배 느려진다. LD_LIBRARY_PATH는 프로세스 시작 시 읽히므로 런타임에 바꿔봐야 늦고,
`ctypes.CDLL(..., RTLD_GLOBAL)` 로 미리 올려야 한다. 이걸 지우면 조용히 느려지기만 한다.

**VAD는 켜두되 끌 수 있어야 한다.** 깨끗한 말소리에서는 VAD on/off가 결과를 바꾸지 않는다(실측).
그런데 BGM이 깔린 오디오에서는 VAD가 문장 앞부분을 통째로 버렸고 `--no-vad` 로만 살아났다.
그래서 기본은 on, 말소리 비율이 25% 미만이면 `--no-vad` 를 안내한다.

**후보정은 하지 않는다.** 용어 사전·오탈자 교정·요약은 사용자가 따로 한다. `initial_prompt` 로
사내 용어를 힌트로 주는 기능도 의도적으로 넣지 않았다. 요청이 오기 전에 추가하지 말 것.

**결과 파일을 말없이 덮어쓰지 않는다.** 이미 있으면 건너뛰고 `--force` 를 안내한다.
사용자가 손으로 고쳐둔 전사본이 있을 수 있다.

## 코드 규칙

- 파일을 열 때 `encoding` 을 반드시 명시한다. Windows에서 기본값이 cp949라 한글이 깨진다.
  출력 txt 기본값은 `utf-8-sig` — 결과가 Windows 폴더에 떨어져 메모장으로 열리기 때문.
  subprocess도 `encoding="utf-8", errors="replace"` 로 고정한다(`ffutil.run`).
- 출력 파일명은 문자열로 만든다. `Path.with_suffix()` 는 `2026.09.09 회의.m4a` 의 `.09 회의` 를
  확장자로 오해한다(테스트 있음: `test_dotted_filename_keeps_its_stem`).
- 단어 토큰을 다시 이어붙여 문장을 만들지 않는다. 한국어는 `안녕`+`하세요` 처럼 어절 중간이
  잘려 나오므로, 한 화자짜리 세그먼트는 whisper 문장을 그대로 쓴다(`merge._split_segment`).
- 화자 전환 지점의 단어 한두 개가 옆 사람에게 붙는 문제는 `merge._smooth_turns` 가 다룬다.
  구두점·0.3초 쉼 사이에 낀 2단어 이하 조각을, 옆 화자 구간과 원래 화자의 30% 이상 겹칠
  때만 옆으로 넘긴다. 겹침 조건 없이 문장 단위로 몰아주면 안 된다 — whisper가 한국어
  구두점을 자주 빠뜨려서, 실측 0928 회의의 문장 중간 전환 91곳 중 60곳이 2초 넘게 이어지는
  실제 화자 전환이었다. 기준값을 정한 근거는 해당 상수 주석에 있다.
