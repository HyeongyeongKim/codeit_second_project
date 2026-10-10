# RAG 실험별 노트북

사용자가 지정한 [`base_+_top_k_수정_+_generation_openai_+_공고_확인.ipynb`](00_supplied_notebook.ipynb)을 기준으로 만든 **설정별 코드 버전**이다. `00`은 전달받은 파일의 소스 코드를 그대로 두고 실행 출력만 비운 기준 파일이다. 나머지 파일은 그 노트북의 관련 셀만 바꾸고 실행 출력을 지웠다. **새 파일을 다시 실행한 결과는 아직 없다.** 이전 점수는 저장소 루트의 [실험 기록](../../RAG_실험_기록.md)에 있는 당시 실행 결과다.

| 파일 | 문서 처리 | 검색 범위 | 후보 / 최종 문맥 | 청킹 | 임베딩 |
| --- | --- | --- | --- | --- | --- |
| [01 OpenAI 기본](01_openai_topk4_global_base.ipynb) | 기본 | 전체 | dense 8 + BM25 8 / 4 | 700/100 | Alibaba |
| [02 OpenAI 정제](02_openai_topk4_global_cleanup.ipynb) | 정제·보존검사 | 전체 | dense 8 + BM25 8 / 4 | 700/100 | Alibaba |
| [03 후보·문맥 확대](03_openai_topk6_global_base.ipynb) | 기본 | 전체 | dense 20 + BM25 20 / 6 | 700/100 | Alibaba |
| [04 사업 식별 기본](04_openai_topk6_project_base.ipynb) | 기본 | 식별 사업 | dense 20 + BM25 20 / 6 | 700/100 | Alibaba |
| [05 사업 식별 정제](05_openai_topk6_project_cleanup.ipynb) | 정제·보존검사 | 식별 사업 | dense 20 + BM25 20 / 6 | 700/100 | Alibaba |
| [06 청킹 확대](06_openai_topk6_project_chunk1000_200.ipynb) | 기본 | 식별 사업 | dense 20 + BM25 20 / 6 | 1000/200 | Alibaba |
| [07 BAAI 임베딩](07_embedding_baai_bge_m3.ipynb) | 기본 | 식별 사업 | dense 20 + BM25 20 / 6 | 700/100 | BAAI/bge-m3 |
| [08 dragonkue 임베딩](08_embedding_dragonkue_bge_m3_ko.ipynb) | 기본 | 식별 사업 | dense 20 + BM25 20 / 6 | 700/100 | dragonkue/BGE-m3-ko |
| [09 KURE 임베딩](09_embedding_nlpai_kure_v1.ipynb) | 기본 | 식별 사업 | dense 20 + BM25 20 / 6 | 700/100 | nlpai-lab/KURE-v1 |

모든 새 버전의 답변 생성과 RAGAS 평가는 GPT-5-mini를 사용한다. 원본 노트북은 dragonkue 임베딩을 선택한 상태였으며, `04`가 Alibaba 기본 임베딩 비교 기준이다. 정제 버전의 문서 처리 셀은 당시 정제·보존검사 노트북에서 가져왔다. 전체 문서 검색 버전은 사업 식별만 우회하고 같은 검색 구조를 사용한다.

초기 로컬 생성 모델을 사용한 실험은 이 묶음에 포함하지 않았다. 이 파일들은 당시 실행 파일의 바이트 단위 복사본이 아니라, 지정 노트북에서 설정 차이를 재구성한 버전이다. 각 파일은 실행 전에 Colab의 문서 폴더, `data_list.csv`, `rag_eval_dataset.json`, API 키가 필요하다. 실행하면 결과 CSV에 파일별 고유 이름이 붙는다.
