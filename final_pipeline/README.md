# RFP RAG 최종 파이프라인 (Python 모듈)

제공된 `RFP_최종_Dense_KiwiBM25_리랭커_GPT5mini (3) (1).ipynb`의 실행 경로를 Python 파일로 분리했습니다. 문서 파싱, 사업 식별, Dense+Kiwi BM25 검색, RRF, 재순위화와 평가 지표를 유지하면서 답변 생성 모델을 Claude Haiku 4.5로 변경했습니다.

## 기본 설정

| 항목 | 값 |
|---|---|
| 입력 | DEV v1.1의 100개 문서 중 `DOC001`, `DOC010`, `DOC020`, `DOC030`, `DOC040`, `DOC050`, `DOC060`, `DOC070`, `DOC080`, `DOC090`의 질문 각 3개, 총 30문항 평가. PDF/HWP/HWPX의 표 Markdown 추출 |
| 청킹 | 700자, 겹침 100자, 구분자 `\n\n` → `\n` → 공백 |
| 임베딩 | `dragonkue/BGE-m3-ko`, 배치 16, 정규화, GPU가 있으면 CUDA |
| 검색 | 사업명을 특정할 수 있으면 해당 사업만, 아니면 전체 문서 |
| 후보 | Dense 20개 + Kiwi BM25 20개 |
| 결합 | RRF `k=60` |
| 재순위화 | `Alibaba-NLP/gte-multilingual-reranker-base`, 배치 8, 최종 문맥 6개 |
| 질문 방식 | 원래 질문을 그대로 검색에 사용 |
| 답변 생성 | `claude-haiku-4-5-20251001`, 최대 출력 4096토큰 |
| 답변 판정·RAGAS 평가 | 비교 기준을 유지하도록 `gpt-5-mini-2025-08-07` 사용 |
| RAGAS | `context_recall`, `faithfulness`, `factual_correctness(mode=f1)`; 작업자 1, 재시도 2, 제한 시간 300초 |

모델과 검색 값을 바꾸려면 `rfp_rag/config.py`를 수정하거나 `run.py`의 옵션을 사용하세요. `--generation-model`에는 Claude 또는 OpenAI 모델 ID를 지정할 수 있습니다. 다른 제공자를 쓴다면 `RagPipeline`에 `embeddings=`, `chat_model=`, `reranker=` 객체를 주입할 수 있습니다. 기본 실행은 위 표와 같습니다.

## 파일 구조

- `rfp_rag/config.py`: 기본 설정과 검증
- `rfp_rag/data.py`: 벤치마크 검증, 문서·표 추출, CSV 메타데이터 연결, 청킹
- `rfp_rag/hwp_parser_fix.py`: 손상된 HWP 하이퍼링크 필드에서 읽기 위치를 전진시키는 파서 보정
- `rfp_rag/pipeline.py`: 사업 식별, 검색기, RRF, 리랭커, 답변 생성
- `rfp_rag/evaluation.py`: 문서 Hit/MRR/NDCG, GPT 판정, 의미 유사도, 문서 길이별 MRR
- `rfp_rag/ragas_eval.py`: RAGAS 세 지표
- `run.py`: 실행 명령

## 실행

별도 Python 환경에서 설치합니다. Colab에서도 패키지를 설치한 뒤 Python 스크립트로 실행할 수 있습니다.

```bash
python -m pip install -r requirements.txt
python run.py show-config
```

`ANTHROPIC_API_KEY`를 환경 변수나 Colab 보안 비밀에 설정합니다. `evaluate`와 `ragas` 명령은 GPT-5 mini로 평가하므로 `OPENAI_API_KEY`도 필요합니다. Colab에서는 먼저 Drive를 마운트합니다. 기본 경로는 원본 노트북과 같은 `/content/drive/MyDrive/data`, `/content/drive/MyDrive/data_list.csv`, `/content/drive/MyDrive/dev_benchmark_v1.1.json`입니다. 로컬에서는 아래처럼 경로를 지정합니다.

```bash
python run.py evaluate --data-dir ./data --metadata-csv ./data_list.csv --benchmark ./dev_benchmark_v1.1.json --out-dir ./results
```

평가에서는 지정한 10개 문서의 질문 30개만 사용합니다. 검색 코퍼스는 기존의 100개 문서를 그대로 사용합니다. 먼저 적은 수의 문항으로 실행을 확인하려면 다음처럼 실행합니다.

```bash
python run.py evaluate --data-dir ./data --metadata-csv ./data_list.csv --benchmark ./dev_benchmark_v1.1.json --limit 3 --skip-ragas
```

`evaluate`는 기본 지표와 RAGAS를 실행합니다. `--skip-ragas`로 기본 평가까지만 저장했다면 다음 명령으로 저장된 30개 답변·문맥을 다시 생성하지 않고 RAGAS만 실행합니다.

```bash
python run.py ragas --out-dir ./results
```

한 질문에 답하려면 사업명이 들어 있는 질문을 입력합니다.

```bash
python run.py answer --question "사업명에 관한 입찰 참가 자격은?" --data-dir ./data --metadata-csv ./data_list.csv --benchmark ./dev_benchmark_v1.1.json
```

예를 들어 임베딩만 바꾸려면 `--embedding-model BAAI/bge-m3`, 일반 BM25를 비교하려면 `--bm25-mode current`를 사용합니다. 그 외 `--chunk-size`, `--chunk-overlap`, `--dense-k`, `--bm25-k`, `--rrf-k`, `--reranker-model`, `--final-context-count`, `--generation-model`, `--generation-max-tokens`, `--judge-model`, `--ragas-model`을 독립적으로 바꿀 수 있습니다. `ragas` 명령은 저장된 `run_config.json`을 읽어 원래 실행의 설정을 이어받습니다.

## 제거한 오류 대응 코드

- 특정 파일을 건너뛰고 나중에 수동으로 합치던 셀을 없앴습니다. 지원 확장자의 파일만 순회하므로 `parsed_results` 폴더가 파일 목록에 섞이지 않습니다. HWP/HWPX를 읽기 전에는 손상된 하이퍼링크 필드에서도 위치를 전진시키도록 파서 메서드를 현재 프로세스에 적용합니다. 원본 문서 파일은 수정하지 않습니다.
- 존재하지 않는 VertexAI 모듈을 `sys.modules`에 가짜로 등록하던 코드를 없앴습니다. RAGAS 0.3.9와 호환되는 LangChain Community 0.3 계열을 설치합니다.
- 중복된 RAGAS 재실행·임시 점수 재구성 셀을 없앴습니다. 기본 평가 결과를 `results/base_scores.csv`에 저장하고, RAGAS는 그 결과와 `ragas_samples.json`을 읽어 한 번만 실행합니다.
- 원본의 LangSmith 업로드는 429 저장 제한으로 일부 실행·채점이 기록되지 않았습니다. 기본 평가는 같은 수식과 GPT 판정 문구로 **로컬 CSV에 먼저 저장**합니다. 따라서 LangSmith 웹 실험 기록은 이 버전에서 만들지 않습니다. 검색·답변·평가 지표의 기본 설정은 그대로입니다.

생성되는 `base_answers.jsonl`과 `ragas_samples.json`에는 문서 내용과 답변이 들어갑니다. `results/`는 Git에서 제외했습니다. 원본 문서, 평가 JSON, CSV, API 키도 GitHub에 올리지 마세요.
