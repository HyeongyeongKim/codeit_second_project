# RFP RAG 주요 실험

**큰 변경을 한 실험 21개를 시간순으로 모았습니다.** 번호별로 사용한 모델·설정과 시도한 이유를 짧게 적었습니다. 한 파일씩 선택해 실행하는 기록이며, 1번부터 모두 이어서 실행하는 과정은 아닙니다. 02~21번의 재순위화 모델은 `Alibaba-NLP/gte-multilingual-reranker-base`입니다.

모델 이름: **GTE**=`Alibaba-NLP/gte-multilingual-base`, **KoE5**=`nlpai-lab/KoE5`, **EXAONE AWQ**=`LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-AWQ`, **BGE-m3-ko**=`dragonkue/BGE-m3-ko`. **GPT-5 mini**는 답변 생성 또는 채점에 사용한 OpenAI 모델입니다.

| 번호 | 모델·핵심 설정 | 왜 시도했나 · 확인한 점 |
| --- | --- | --- |
| [01](01_early_search_methods.ipynb) | 연말정산 PDF, KoE5 임베딩, Qwen2.5-3B, 청크 500/100, 후보 8개 → `BAAI/bge-reranker-v2-m3`로 4개 선택 | RFP 작업 전에 기본 검색·질문 재작성·HyDE가 검색 결과를 어떻게 바꾸는지 시험했습니다. |
| [02](02_rfp_dense_exaone_baseline.ipynb) | RFP HWP/PDF, GTE 임베딩, EXAONE AWQ 답변, 청크 700/100, 재순위화 4개 | 실제 RFP 문서와 CSV 정보를 연결해 기본 검색과 HyDE의 답변을 비교했습니다. |
| [03](03_openai_generation_trial.ipynb) | GPT-5 mini 답변, 청크 700/100, 재순위화 4개. OpenAI 임베딩 셀 뒤에 GTE 임베딩 셀이 다시 실행됨 | 답변 모델 변경을 시험한 탐색본입니다. 두 임베딩 결과를 공정하게 비교한 파일은 아닙니다. |
| [04](04_hybrid_rrf_qwen_judge.ipynb) | GTE+EXAONE, Dense 8개·BM25 8개 → RRF → 재순위화 4개, Qwen2.5-1.5B 채점 | 단어 검색을 더해 정답 청크를 찾고, 검색과 답변을 따로 평가하려 했습니다. |
| [05](05_hybrid_with_evaluation.ipynb) | 04와 같은 하이브리드·Qwen 평가. HyDE 설정이 있어도 실제 검색은 원래 질문 사용 | 기존 검색 코드 뒤에 평가를 붙인 중간본입니다. HyDE 성능 결과로 해석하면 안 됩니다. |
| [06](06_ragas_basic_preprocessing.ipynb) | GTE+EXAONE, 청크 700/100, 하이브리드 8+8 → 4개, RAGAS | 실제 참조답이 있는 평가셋으로 검색 근거·답변 충실도·정답성을 평가하려 했습니다. 당시 EXAONE과 RAGAS 연결 오류가 있었습니다. |
| [07](07_ragas_cleanup_preservation.ipynb) | 06 설정에 쪽 번호·목차·빈 표 정제와 원문 인용 보존검사 추가 | 문서 정제가 검색에 도움이 되는지, 필요한 근거를 지우지 않는지 비교하려 했습니다. |
| [08](08_openai_top6_all_projects.ipynb) | GTE+GPT-5 mini, Dense 20개·BM25 20개 → 재순위화 6개, 첫 10문항 평가 | 검색 후보와 답변 근거를 늘렸지만 다른 사업 문서가 섞이는 문제를 확인했습니다. |
| [09](09_project_filter_gte_700_100.ipynb) | 08 설정에 질문의 사업명 식별·사업별 검색 추가, 청크 700/100 | 다른 사업 근거가 섞이지 않도록 검색 범위를 제한했습니다. 이 단계에서 답변 평가가 크게 개선됐습니다. |
| [10](10_project_filter_with_cleanup.ipynb) | 09 설정에 문서 정제·보존검사 추가 | 사업별 검색을 유지한 채 정제 효과를 다시 살폈습니다. 점수 상승만으로 정제의 효과를 확정하지 않았습니다. |
| [11](11_chunk_1000_200.ipynb) | 09와 같고 청크만 1000/200으로 변경 | 큰 청크가 근거 검색과 답변에 도움이 되는지 시험했습니다. 채점 오류 사례가 있어 우열은 미확정입니다. |
| [12](12_embedding_bge_m3.ipynb) | 09 설정에서 임베딩만 `BAAI/bge-m3`로 변경 | GTE와 검색 모델을 비교하기 위한 **재현용 재구성본**입니다. |
| [13](13_embedding_kure_v1.ipynb) | 09 설정에서 임베딩만 `nlpai-lab/KURE-v1`로 변경 | 같은 질문으로 임베딩 영향을 비교하기 위한 **재현용 재구성본**입니다. |
| [14](14_embedding_bge_m3_ko.ipynb) | 09 설정에서 임베딩만 `dragonkue/BGE-m3-ko`로 변경 | 한국어 임베딩 후보를 시험한 **재현용 재구성본**입니다. 당시 한 사업 10문항에서 답변 점수가 가장 높았습니다. |
| [15](15_basic_vs_kiwi_bm25.ipynb) | BGE-m3-ko+GPT-5 mini, 청크 700/100, 기본·Kiwi BM25, 2개 사업 × 2문항 | 한국어 형태소 토큰화가 기존 BM25 검색에 도움이 되는지 작은 범위에서 먼저 비교했습니다. |
| [16](16_langsmith_36_basic.ipynb) | BGE-m3-ko+GPT-5 mini 답변, 기본 BM25, 근거 6개, Claude 채점, 3개 사업 × 12문항 | 기본 BM25 평가를 여러 사업으로 넓히기 위해 만들었습니다. 전체 실행 완료 여부는 확인되지 않았습니다. |
| [17](17_langsmith_36_kiwi.ipynb) | 16과 같은 설정에서 Kiwi BM25 사용 | 같은 36문항으로 형태소 토큰화의 차이를 보려 했습니다. 전체 실행 완료 여부는 확인되지 않았습니다. |
| [18](18_langsmith_30_claude_trial.ipynb) | BGE-m3-ko+GPT-5 mini 답변, 기본 BM25, 근거 6개, Claude 채점, 3×10문항 | 30문항 채점을 시도했지만 당시 Claude API 오류가 있었습니다. 실패 과정도 남겼습니다. |
| [19](19_langsmith_30_basic_gpt.ipynb) | BGE-m3-ko+GPT-5 mini 답변·채점, 기본 BM25, 재순위화 20개, 3×10문항 | 기본 BM25의 30문항 결과를 LangSmith에 기록하기 위해 통합 파일에서 분리했습니다. 당시 실행 기록이 있습니다. |
| [20](20_langsmith_30_kiwi_gpt.ipynb) | 19와 같은 설정에서 Kiwi BM25 사용 | 동일한 30문항에서 BM25 토큰화 방법만 바꿔 비교하도록 분리했습니다. 당시 실행 기록이 있습니다. |
| [21](21_bm25_search_stage_comparison.ipynb) | BGE-m3-ko, 기본·Kiwi BM25 단독/RRF/재순위화 각 상위 20개, 3×10문항 | 답변 모델 없이 어느 검색 단계에서 정답 근거가 달라지는지 확인합니다. 결과 JSON에는 문서 본문이 있어 공개하면 안 됩니다. |

**실행 준비:** Colab에서 원하는 파일 하나를 열고 위에서 아래로 실행합니다. 대부분 `/content/drive/MyDrive/data`의 RFP 문서, `/content/drive/MyDrive/data_list.csv`, 평가용 JSON이 필요합니다. API를 쓰는 파일은 첫 셀의 안내대로 Colab 비밀에 키를 등록하세요. 실행 출력·원본 문서·API 키는 이 폴더에 넣지 않았습니다.

점수와 한계의 자세한 근거는 [실험 기록](../RAG_실험_기록.md)과 [10월 8일 보고서](../2026-10-08_RAG_상세_보고서.md)에 있습니다.
