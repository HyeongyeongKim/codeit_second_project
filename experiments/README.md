# RFP RAG 주요 실험

실험이 이어진 **변경 이유 → 방법 → 결과 → 다음 변경 이유**는 [실험의 의사결정 흐름](../EXPERIMENT_JOURNEY.md)에 정리했습니다. 이 표는 각 노트북의 설정을 찾기 위한 색인입니다.

**기존 주요 실험 21개와 생성 모델 비교 22번, 청킹 비교 23번을 모았습니다.** 번호별로 사용한 모델·설정과 시도한 이유를 짧게 적었습니다. 한 파일씩 선택해 실행하는 기록이며, 1번부터 모두 이어서 실행하는 과정은 아닙니다. 02~21번의 재순위화 모델은 `Alibaba-NLP/gte-multilingual-reranker-base`입니다.

모델 이름: **GTE**=`Alibaba-NLP/gte-multilingual-base`, **KoE5**=`nlpai-lab/KoE5`, **EXAONE AWQ**=`LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-AWQ`, **BGE-m3-ko**=`dragonkue/BGE-m3-ko`. **GPT-5 mini**는 답변 생성 또는 채점에 사용한 OpenAI 모델입니다.

| 번호 | 모델·핵심 설정 | 왜 시도했나 · 확인한 점 |
| --- | --- | --- |
| [01](01_early_search_methods.ipynb) | 연말정산 PDF, KoE5 임베딩, Qwen2.5-3B, 청크 500/100, 후보 8개 → `BAAI/bge-reranker-v2-m3`로 4개 선택 | **일반 질문과 HyDE를 비교**했습니다. 당시 한글 문서와 표를 읽는 방법을 몰라 PDF로 먼저 시험했습니다. |
| [02](02_rfp_dense_exaone_baseline.ipynb) | RFP HWP/PDF, GTE 임베딩, EXAONE AWQ 답변, 청크 700/100, 재순위화 4개 | **한글 로더와 표의 마크다운 추출을 적용**하고 RFP 파일 100개를 임베딩했습니다. 일반 질문과 HyDE 비교를 이어 갔습니다. |
| [03](03_openai_generation_trial.ipynb) | GPT-5 mini 답변, 청크 700/100, 재순위화 4개. OpenAI 임베딩과 GTE 임베딩 셀은 각각 별도 실행 | **답변 품질을 높이기 위해 OpenAI 생성 모델을 시험**했고 결과가 매우 좋았습니다. GTE 경로도 비교용으로 구현했습니다. |
| [04](04_hybrid_rrf_qwen_judge.ipynb) | GTE+EXAONE, Dense 8개·BM25 8개 → RRF → 재순위화 4개 | **하이브리드 검색**으로 의미 검색과 단어 검색을 합쳐 근거를 더 잘 찾으려 했습니다. |
| [05](05_hybrid_with_evaluation.ipynb) | 04의 하이브리드 검색 + Qwen2.5-1.5B 채점(0/0.5/1) + 정답 청크 ID 대조 | **검색 근거와 답변을 평가**했습니다. Qwen은 검색 청크로 질문에 답할 수 있는지와 생성 답변이 빠짐없이 근거에 맞는지 채점합니다. 정답 청크 포함 여부도 따로 봅니다. HyDE 설정은 있지만 실제 검색은 원래 질문을 사용합니다. |
| [06](06_ragas_basic_preprocessing.ipynb) | GTE+EXAONE, 청크 700/100, 하이브리드 8+8 → 4개, RAGAS | **RAGAS의 문맥 재현율·충실도·사실 정확도**를 계산하려 했습니다. 각각 정답 근거를 찾았는지, 답변이 근거에 기반하는지, 정답과 맞는지를 봅니다. 당시 연결 오류가 있었습니다. |
| [07](07_ragas_cleanup_preservation.ipynb) | 06 설정에 쪽 번호·목차·빈 표 정제와 원문 인용 보존검사 추가 | 정제 효과를 보려 **기본본과 정제본을 별도 파일로 진행**했습니다. 이 단계 결과만으로는 두 방식의 우열을 비교할 수 없었습니다. |
| [08](08_openai_top6_all_projects.ipynb) | GTE+GPT-5 mini, Dense 20개·BM25 20개 → 재순위화 6개, 첫 10문항 평가 | 검색 후보와 답변 근거를 늘렸지만 다른 사업 문서가 섞이는 문제를 확인했습니다. |
| [09](09_project_filter_gte_700_100.ipynb) | 08 설정에 질문의 사업명 식별·사업별 검색 추가, 청크 700/100 | 다른 사업 근거가 섞이지 않도록 검색 범위를 제한했습니다. 이 단계에서 답변 평가가 크게 개선됐습니다. |
| [10](10_project_filter_with_cleanup.ipynb) | 09 설정에 문서 정제·보존검사 추가 | 사업별 검색을 유지한 채 정제 효과를 다시 살폈지만 **뚜렷한 효과가 없어 이후 구성에서 제외**했습니다. |
| [11](11_chunk_1000_200.ipynb) | 09와 같고 청크만 1000/200으로 변경 | 큰 청크가 근거 검색과 답변에 도움이 되는지 시험했습니다. 채점 오류 사례가 있어 우열은 미확정입니다. |
| [12](12_embedding_bge_m3.ipynb) | 09 설정에서 임베딩만 `BAAI/bge-m3`로 변경 | GTE와 검색 모델을 비교하기 위한 **재현용 재구성본**입니다. |
| [13](13_embedding_kure_v1.ipynb) | 09 설정에서 임베딩만 `nlpai-lab/KURE-v1`로 변경 | 같은 질문으로 임베딩 영향을 비교하기 위한 **재현용 재구성본**입니다. |
| [14](14_embedding_bge_m3_ko.ipynb) | 09 설정에서 임베딩만 `dragonkue/BGE-m3-ko`로 변경 | 한국어 임베딩 후보를 시험한 **재현용 재구성본**입니다. 당시 한 사업 10문항에서 답변 점수가 가장 높았습니다. |
| [15](15_basic_vs_kiwi_bm25.ipynb) | BGE-m3-ko+GPT-5 mini, 청크 700/100, 기본·Kiwi BM25, 2개 사업 × 2문항 | **기본 BM25와 Kiwi BM25를 4문항으로 비교**했습니다. 표본이 너무 적어 질문 수를 늘리기로 했습니다. |
| [16](16_langsmith_36_basic.ipynb) | BGE-m3-ko+GPT-5 mini 답변, 기본 BM25, 근거 6개, Claude 채점, 3개 사업 × 12문항 | 기본 BM25 평가를 **36문항**으로 늘리고 Claude 채점을 시도했으나 실패했습니다. |
| [17](17_langsmith_36_kiwi.ipynb) | 16과 같은 설정에서 Kiwi BM25 사용 | Kiwi BM25도 같은 **36문항**과 Claude 채점으로 비교하려 했으나 채점에 실패했습니다. |
| [18](18_langsmith_30_claude_trial.ipynb) | BGE-m3-ko+GPT-5 mini 답변, 기본 BM25, 근거 6개, Claude 채점, 3×10문항 | 질문 구성을 **3개 사업 × 10문항**으로 바꿔 Claude 채점을 다시 시도했지만 API 오류로 실패했습니다. 15~18번은 평가 방식과 문항 구성이 바뀌는 과도기입니다. |
| [19](19_langsmith_30_basic_gpt.ipynb) | BGE-m3-ko+GPT-5 mini 답변·채점, 기본 BM25, 재순위화 20개, 3×10문항 | **기본 BM25 → 하이브리드 검색 → 재순위화**를 거쳐 30문항을 평가했습니다. 통합 파일에서 분리했고 당시 실행 기록이 있습니다. |
| [20](20_langsmith_30_kiwi_gpt.ipynb) | 19와 같은 설정에서 Kiwi BM25 사용 | **Kiwi BM25 → 하이브리드 검색 → 재순위화**로 같은 30문항을 평가했습니다. 당시 실행 기록이 있습니다. |
| [21](21_bm25_search_stage_comparison.ipynb) | BGE-m3-ko, 기본·Kiwi BM25 단독/RRF/재순위화 각 상위 20개, 3×10문항 | 기본·Kiwi 각각에서 **단독 검색, 하이브리드 검색, 재순위화**를 비교해 어느 단계가 근거 검색을 개선하는지 봅니다. 결과 JSON에는 문서 본문이 있어 공개하면 안 됩니다. |
| [22](22_kiwi_bm25_gpt5mini_vs_qwen38_30.ipynb) | Kiwi BM25 단독 검색 상위 20개 중 앞의 4개를 GPT-5 mini와 Qwen3.8-27B Q3_K_S에 동일하게 전달, 3×10문항 | **생성 모델만 바꿔 답변을 비교**합니다. 20번의 GPT 답변은 하이브리드 검색 결과이므로 여기서 다시 생성합니다. [10월 12일 Notion 기록](https://app.notion.com/p/3f767f3920118062bf69f1db0e198c33)은 서버 중단 뒤 저장된 답변부터 이어 실행했고 두 모델의 품질 차이가 크지 않았다고 보고합니다. 이 저장소에는 결과 CSV가 없어 수치와 완료 범위를 독립 검증할 수 없습니다. |
| [23](23_chunking_three_way.ipynb) | 같은 30문항·BGE-m3-ko·기본 BM25·RRF·재정렬·GPT 평가에서 700/100, parent–child, 구조 기반 청킹을 비교 | **세 방식 × 30문항을 2회 실행.** 1회차 GPT 평균은 700/100 **0.7550**, parent–child **0.6957**, 구조 기반 **0.7133**이었고, 2회차에는 각각 **0.6673·0.7013·0.6733**으로 순위가 바뀌었습니다. parent–child는 인용 포함률이 가장 높지만 문맥도 가장 깁니다. [두 회차 결과와 문항별 점수](../CHUNKING_3WAY_RESULTS_2026-10-13.md) |

**실행 준비:** Colab에서 원하는 파일 하나를 열고 위에서 아래로 실행합니다. 대부분 `/content/drive/MyDrive/data`의 RFP 문서, `/content/drive/MyDrive/data_list.csv`, 평가용 JSON이 필요합니다. API를 쓰는 파일은 첫 셀의 안내대로 Colab 비밀에 키를 등록하세요. 실행 출력·원본 문서·API 키는 이 폴더에 넣지 않았습니다.

점수와 한계의 자세한 근거는 [실험 기록](../RAG_실험_기록.md)과 [10월 8일 보고서](../2026-10-08_RAG_상세_보고서.md)에 있습니다.

