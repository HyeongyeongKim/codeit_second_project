# RFP RAG 최종 파이프라인 (Python 모듈)

제공된 `RFP_최종_Dense_KiwiBM25_리랭커_GPT5mini (3) (1).ipynb`의 실행 경로를 Python 파일로 분리했습니다. 문서 파싱, 사업 식별, Dense+Kiwi BM25 검색, RRF, 재순위화와 답변 생성 모델 Claude Haiku 4.5 설정을 유지합니다. 검색 평가는 정답 문서 순위 대신 근거 청크 순위로 계산합니다.

## 기본 설정

| 항목 | 값 |
|---|---|
| 입력 기본값 | DEV v1.1의 `DOC001`, `DOC010`, `DOC020`, `DOC030`, `DOC040`, `DOC050`, `DOC060`, `DOC070`, `DOC080`, `DOC090` 총 10개 문서. 각 3문항씩 30문항 사용. PDF/HWP/HWPX의 표 Markdown 추출 |
| 청킹 | 700자, 겹침 100자, 구분자 `\n\n` → `\n` → 공백 |
| 임베딩 | `dragonkue/BGE-m3-ko`, 배치 16, 정규화, GPU가 있으면 CUDA |
| 검색 | 사업명을 특정할 수 있으면 해당 사업만, 아니면 전체 문서 |
| 후보 | Dense 20개 + Kiwi BM25 20개 |
| 결합 | RRF `k=60` |
| 재순위화 | `Alibaba-NLP/gte-multilingual-reranker-base`, 배치 8, 최종 문맥 6개 |
| 질문 방식 | 원래 질문을 그대로 검색에 사용 |
| 답변 생성 | `claude-haiku-4-5-20251001`, 최대 출력 4096토큰 |
| 답변 판정 | `gemini-3.1-flash-lite`로 LLM-as-judge 평가 |
| RAGAS 평가 모델 | `gemini-3.1-flash-lite` |
| 평가 지표 | 청크 `Hit@5`, 청크 `MRR`, `LLM-as-Judge`, `Factual Correctness (F1)` 네 가지 |
| RAGAS | `factual_correctness(mode=f1)`; 작업자 1, 재시도 2, 제한 시간 300초 |

사용할 문서 번호는 **`rfp_rag/config.py` 맨 위의 `DEFAULT_DOCUMENT_NUMBERS` 한 목록**에서 바꿉니다. 기본값은 `(1, 10, 20, 30, 40, 50, 60, 70, 80, 90)`입니다. 실행할 때만 바꾸려면 아래 `--document-numbers` 옵션을 쓰세요. 이 선택 하나가 문서 파싱, 청킹, 검색 코퍼스와 임베딩, 답변 생성 문항, 검색·생성·RAGAS 평가에 모두 적용됩니다. 모델과 검색 값은 `rfp_rag/config.py` 또는 `run.py` 옵션으로 바꿀 수 있습니다. `--generation-model`에는 Claude 또는 OpenAI 모델 ID를 지정할 수 있습니다. 다른 제공자를 쓴다면 `RagPipeline`에 `embeddings=`, `chat_model=`, `reranker=` 객체를 주입할 수 있습니다.

## 파일 구조

- `rfp_rag/config.py`: 기본 설정과 검증
- `rfp_rag/data.py`: 벤치마크 검증, 문서·표 추출, CSV 메타데이터 연결, 청킹
- `rfp_rag/hwp_parser_fix.py`: 손상된 HWP 하이퍼링크 필드에서 읽기 위치를 전진시키는 파서 보정
- `rfp_rag/pipeline.py`: 사업 식별, 검색기, RRF, 리랭커, 답변 생성
- `rfp_rag/evaluation.py`: 청크 Hit@5·MRR, Gemini 판정, 원문 길이별 청크 MRR
- `rfp_rag/ragas_eval.py`: RAGAS Factual Correctness (F1)
- `run.py`: 실행 명령

## 실행

별도 Python 환경에서 설치합니다. Colab에서도 패키지를 설치한 뒤 Python 스크립트로 실행할 수 있습니다.

Colab에서는 [`RFP_RAG_Colab_실행.ipynb`](RFP_RAG_Colab_실행.ipynb)을 열고 위에서부터 실행하세요. 문서 파싱부터 네 평가 지표까지 모든 실행 코드와 설치 패키지 목록이 노트북 안에 있어 GitHub의 Python 파일을 가져오지 않습니다. 문서 번호와 Drive 경로는 노트북 앞부분의 설정 셀에서 바꾸고, API 키는 Colab 보안 비밀에 등록합니다. 결과는 Drive의 `RFP_RAG_results/`에 저장됩니다.

```bash
python -m pip install -r requirements.txt
python run.py show-config
```

Colab에서는 먼저 Drive를 마운트합니다. 기본 경로는 원본 노트북과 같은 `/content/drive/MyDrive/data`, `/content/drive/MyDrive/data_list.csv`, `/content/drive/MyDrive/dev_benchmark_v1.1.json`입니다. 로컬에서는 아래처럼 경로를 지정합니다.

```bash
python run.py evaluate --data-dir ./data --metadata-csv ./data_list.csv --benchmark ./dev_benchmark_v1.1.json --out-dir ./results
```

위 기본 실행은 선택한 **10개 문서만** 파싱·청킹·임베딩해 검색에 사용하고, 그 문서의 **30개 질문만** 답변 생성·평가에 사용합니다. 문서 번호를 코드 수정 없이 바꾸려면 다음처럼 입력합니다.

```bash
python run.py evaluate --document-numbers 1,10,20,30,40,50,60,70,80,90 --data-dir ./data --metadata-csv ./data_list.csv --benchmark ./dev_benchmark_v1.1.json --out-dir ./results
```

전체 100개 문서와 300문항을 다시 사용하려면 `--document-numbers all`을 지정하세요. 선택한 번호는 `run_config.json`에 저장되며, 나중에 `ragas` 명령을 실행해도 같은 문항을 사용합니다. 각 실행의 결과는 서로 다른 `--out-dir`에 저장하세요.

검색 평가는 리랭커가 정렬한 **청크 순위**에서 `chunk_hit_at_5`와 `chunk_mrr`을 계산합니다. MRR은 첫 정답 청크의 전체 검색 순위의 역수입니다. 생성 평가는 Gemini의 `llm_as_judge`와 RAGAS의 `factual_correctness` (F1)를 사용합니다. 평가 JSON에는 현재 700/100 청킹의 정답 청크 ID가 없으므로, 각 문항의 `evidence` 문장을 같은 원문 파일의 청크 본문에 대조해 근거 청크 후보를 만듭니다. 이 점수는 **근거 문장 일치 기반의 임시 정답 청크**를 사용한 결과이며, 사람이 검증한 정답 청크 라벨과 같다고 간주하면 안 됩니다. 공백과 유니코드 표기만 정규화해 일치 여부를 확인합니다.

근거 문장 중 하나라도 원문 청크에 연결되지 않으면 해당 문항의 청크 Hit@5와 MRR은 빈칸으로 저장합니다. `chunk_evidence_aligned`와 `evidence_quote_aligned_count`로 해당 문항을 확인하고, `chunk_evidence_alignment.jsonl`에서 연결되지 않은 근거 문장을 볼 수 있습니다. 문서 길이별 `mrr_by_length.csv`의 평균도 청크 MRR이며, 함께 표시되는 `aligned_question_count`로 평균에 포함된 문항 수를 확인할 수 있습니다. `all_scores.csv`에는 네 지표가 문항별로 저장됩니다.

`evaluate`는 선택한 문항의 답변 생성, Gemini 판정, RAGAS 평가를 실행합니다. `--skip-ragas`로 기본 평가까지만 저장했다면 다음 명령으로 저장된 답변·문맥을 다시 생성하지 않고 RAGAS만 실행합니다. 이전에 GPT 모델로 저장한 실행 설정도 이 명령에서는 Gemini 기본값으로 전환됩니다.

```bash
python run.py ragas --out-dir ./results
```

한 질문에 답하려면 사업명이 들어 있는 질문을 입력합니다.

```bash
python run.py answer --question "사업명에 관한 입찰 참가 자격은?" --data-dir ./data --metadata-csv ./data_list.csv --benchmark ./dev_benchmark_v1.1.json
```

예를 들어 임베딩만 바꾸려면 `--embedding-model BAAI/bge-m3`, 일반 BM25를 비교하려면 `--bm25-mode current`를 사용합니다. 그 외 `--chunk-size`, `--chunk-overlap`, `--dense-k`, `--bm25-k`, `--rrf-k`, `--reranker-model`, `--final-context-count`, `--generation-model`, `--generation-max-tokens`, `--judge-model`, `--ragas-model`을 독립적으로 바꿀 수 있습니다. `ragas` 명령은 저장된 `run_config.json`을 읽어 원래 실행의 설정을 이어받습니다.

## API 키 넣는 곳

VS Code에서 저장소를 연 뒤, 터미널을 `final_pipeline` 폴더에서 열어 아래 명령으로 `.env`를 만드세요. `run.py`가 같은 폴더의 `.env`를 실행할 때 자동으로 읽습니다. 팀원도 각자 자신의 컴퓨터에서 같은 작업을 하고 자신의 키를 입력합니다. `.env`는 `.gitignore`에 등록되어 GitHub에 올라가지 않습니다.

**GitHub 웹사이트에서 `.env.example`을 편집해 실제 키를 넣지 마세요.** Secret scanning 경고가 나오면 우회하지 말고 편집 내용에서 키를 지우세요. `.env.example`은 빈 양식으로 유지합니다.

```powershell
Copy-Item .env.example .env
```

`.env` 파일의 형식은 아래와 같습니다. `=` 뒤에 본인의 실제 키를 입력하세요. 키를 코드나 `.env.example`에 적어 GitHub에 올리지 마세요.

```dotenv
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GOOGLE_API_KEY=
```

| 키 이름 | 쓰는 곳 | 필요한 명령 |
|---|---|---|
| `ANTHROPIC_API_KEY` | Claude Haiku 4.5 답변 생성 | `answer`, `evaluate` |
| `OPENAI_API_KEY` | OpenAI 모델을 답변 생성 모델로 바꿨을 때만 사용 | OpenAI 생성 모델을 지정한 `answer`, `evaluate` |
| `GOOGLE_API_KEY` | Gemini 3.1 Flash-Lite 답변 판정과 RAGAS 평가 | `evaluate`, `ragas` |

기존 환경 변수 값이 있으면 그 값을 우선 사용합니다. Colab에서는 `final_pipeline/.env`에 키를 넣거나 보안 비밀(Secrets)을 사용할 수 있습니다. GitHub Actions에서 실행하는 경우에는 저장소의 **Settings → Secrets and variables → Actions**에 등록하고 워크플로에서 환경 변수로 전달해야 합니다. GitHub에 등록한 비밀은 Colab으로 자동 전달되지 않습니다.

`evaluate --skip-ragas`도 Gemini 답변 판정을 하므로 `GOOGLE_API_KEY`가 필요합니다. 저장된 결과에 `ragas`만 실행할 때도 `GOOGLE_API_KEY`만 필요합니다.

## 제거한 오류 대응 코드

- 특정 파일을 건너뛰고 나중에 수동으로 합치던 셀을 없앴습니다. 지원 확장자의 파일만 순회하므로 `parsed_results` 폴더가 파일 목록에 섞이지 않습니다. HWP/HWPX를 읽기 전에는 손상된 하이퍼링크 필드에서도 위치를 전진시키도록 파서 메서드를 현재 프로세스에 적용합니다. 원본 문서 파일은 수정하지 않습니다.
- 존재하지 않는 VertexAI 모듈을 `sys.modules`에 가짜로 등록하던 코드를 없앴습니다. RAGAS 0.3.9와 호환되는 LangChain Community 0.3 계열을 설치합니다.
- 중복된 RAGAS 재실행·임시 점수 재구성 셀을 없앴습니다. 기본 평가 결과를 `results/base_scores.csv`에 저장하고, RAGAS는 그 결과와 `ragas_samples.json`을 읽어 한 번만 실행합니다.
- 원본의 LangSmith 업로드는 429 저장 제한으로 일부 실행·채점이 기록되지 않았습니다. 평가는 **로컬 CSV에 먼저 저장**합니다. 따라서 LangSmith 웹 실험 기록은 이 버전에서 만들지 않습니다.

생성되는 `base_answers.jsonl`과 `ragas_samples.json`에는 문서 내용과 답변이 들어갑니다. `results/`는 Git에서 제외했습니다. 원본 문서, 평가 JSON, CSV, API 키도 GitHub에 올리지 마세요.
