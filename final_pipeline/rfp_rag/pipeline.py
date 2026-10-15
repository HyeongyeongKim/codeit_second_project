"""Project routing, Dense+Kiwi BM25+RRF retrieval, reranking, and generation."""

from __future__ import annotations

import hashlib
import time
import unicodedata
from collections import defaultdict

from .config import RagSettings


ANSWER_RULES = """
너는 RFP 문서의 사실 확인 담당자다. 아래 <문서>에 적힌 내용만으로 <질문>에 답한다.

규칙:
1. 질문이 여러 항목을 묻는다면 항목별로 근거를 따로 찾는다.
2. 질문의 사업명과 다른 사업의 정보는 사용하지 않는다.
3. 각 항목에 대해 문서에서 직접 인용할 수 있는 문구를 먼저 제시한다.
4. 직접 인용할 근거가 없는 항목은 추측하지 말고 "제공된 문서에서 확인할 수 없습니다"라고 쓴다.
5. 금액, 기간, 날짜, 비율, 인력, 제출 조건은 문서의 표현을 그대로 옮긴다. 계산하거나 일반적인 관행으로 빈칸을 채우지 않는다.
6. 답변의 각 사실 바로 뒤에 근거 문서 번호 [1]처럼 붙인다.
7. 모델의 사전 지식은 근거로 사용하지 않는다.

출력 형식:
근거:
- 항목명: [문서 번호] "문서에서 그대로 가져온 짧은 문구"
- 근거가 없는 항목명: 확인 불가

답변:
근거가 확인된 내용만 간결하게 답한다. 근거가 없는 항목은 확인할 수 없다고 밝힌다.

<문서>
{context}
</문서>

<질문>
{query}
</질문>
"""

PUNCTUATION_TAGS = {"SF", "SP", "SS", "SE", "SO", "SW"}


def normalize_project_text(value: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFKC", str(value)).casefold()
        if char.isalnum()
    )


def project_title_matches_query(project_key: tuple[str, str], normalized_query: str) -> bool:
    agency, title = project_key
    normalized_title = normalize_project_text(title)
    if normalized_title in normalized_query:
        return True
    normalized_agency = normalize_project_text(agency)
    if normalized_agency and normalized_title.startswith(normalized_agency):
        short_title = normalized_title[len(normalized_agency):]
        if len(short_title) >= 8 and normalized_agency in normalized_query:
            return short_title in normalized_query
    return False


def _content(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        ).strip()
    return ""


class RagPipeline:
    """The original search/generation sequence, with replaceable model objects."""

    def __init__(self, chunks, settings: RagSettings, *, aliases=None,
                 embeddings=None, chat_model=None, reranker=None, kiwi=None):
        settings.validate()
        self.settings = settings
        self.chunks = chunks
        self.aliases = aliases or {}
        self.device = self._device()

        if embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={"device": self.device,
                              "trust_remote_code": settings.embedding_trust_remote_code},
                encode_kwargs={"batch_size": settings.embedding_batch_size,
                               "normalize_embeddings": settings.normalize_embeddings},
            )
        if chat_model is None:
            if settings.generation_model.startswith("claude-"):
                from langchain_anthropic import ChatAnthropic

                chat_model = ChatAnthropic(
                    model=settings.generation_model,
                    max_tokens=settings.generation_max_tokens,
                    timeout=settings.generation_timeout,
                    max_retries=settings.generation_retries,
                )
            else:
                from langchain_openai import ChatOpenAI

                chat_model = ChatOpenAI(
                    model=settings.generation_model,
                    timeout=settings.generation_timeout,
                    max_retries=settings.generation_retries,
                )
        if reranker is None:
            from sentence_transformers import CrossEncoder

            reranker = CrossEncoder(
                settings.reranker_model, trust_remote_code=True, device=self.device,
            )
        if kiwi is None and settings.bm25_mode == "kiwi":
            from kiwipiepy import Kiwi

            kiwi = Kiwi()
        self.embeddings = embeddings
        self.chat_model = chat_model
        self.reranker = reranker
        self.kiwi = kiwi

        self.project_chunks_by_key = defaultdict(list)
        for doc in chunks:
            agency = str(doc.metadata.get("발주 기관", "") or "").strip()
            title = str(doc.metadata.get("사업명", "") or "").strip()
            if title:
                self.project_chunks_by_key[(agency, title)].append(doc)
        self.project_catalog = sorted(self.project_chunks_by_key)
        if not self.project_catalog:
            raise ValueError("사업명 메타데이터가 있는 청크가 없습니다.")
        self.dense_retrievers = {}
        self.bm25_retrievers = {}
        self.retrieval_cache = {}
        self.answer_cache = {}

    @staticmethod
    def _device() -> str:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _choose_project(self, matches, query):
        matches = sorted(set(matches))
        if len(matches) == 1:
            return matches[0]
        normalized_query = normalize_project_text(query)
        agency_matches = [
            key for key in matches
            if key[0] and normalize_project_text(key[0]) in normalized_query
        ]
        if len(agency_matches) == 1:
            return agency_matches[0]
        labels = ", ".join(f'[{agency}] "{title}"' for agency, title in matches)
        raise ValueError(f"사업을 하나로 특정할 수 없습니다: {labels}")

    def identify_project(self, query):
        normalized_query = normalize_project_text(query)
        if not normalized_query:
            raise ValueError("질문이 비어 있습니다.")
        direct_matches = [
            key for key in self.project_catalog
            if project_title_matches_query(key, normalized_query)
        ]
        if direct_matches:
            specific = [
                key for key in direct_matches
                if not any(
                    normalize_project_text(key[1]) != normalize_project_text(other[1])
                    and normalize_project_text(key[1]) in normalize_project_text(other[1])
                    for other in direct_matches
                )
            ]
            return self._choose_project(specific, query)

        alias_matches = [
            (alias, target) for alias, target in self.aliases.items()
            if normalize_project_text(alias) in normalized_query
        ]
        if alias_matches:
            longest = max(len(normalize_project_text(alias)) for alias, _ in alias_matches)
            targets = [target for alias, target in alias_matches
                       if len(normalize_project_text(alias)) == longest]
            unknown = [target for target in targets if target not in self.project_chunks_by_key]
            if unknown:
                raise ValueError(f"PROJECT_ALIASES의 사업이 문서 목록에 없습니다: {unknown}")
            return self._choose_project(targets, query)
        raise ValueError("질문에서 사업명을 찾지 못했습니다.")

    def _kiwi_morphs(self, text):
        return [token.form for token in self.kiwi.tokenize(text)
                if token.form.strip() and token.tag not in PUNCTUATION_TAGS]

    def _get_retrievers(self, project_key, bm25_mode):
        if bm25_mode not in {"kiwi", "current"}:
            raise ValueError(f"지원하지 않는 BM25 방식: {bm25_mode}")
        from langchain_community.retrievers import BM25Retriever
        from langchain_community.vectorstores import FAISS

        project_chunks = self.chunks if project_key is None else self.project_chunks_by_key[project_key]
        if project_key not in self.dense_retrievers:
            store = None
            for start in range(0, len(project_chunks), 1000):
                batch = project_chunks[start:start + 1000]
                if store is None:
                    store = FAISS.from_documents(batch, self.embeddings)
                else:
                    store.add_documents(batch)
            # Cache only the complete index, never a partially built batch.
            self.dense_retrievers[project_key] = store.as_retriever(
                search_kwargs={"k": self.settings.dense_k}
            )
        bm25_key = (project_key, bm25_mode)
        if bm25_key not in self.bm25_retrievers:
            if bm25_mode == "kiwi":
                if self.kiwi is None:
                    from kiwipiepy import Kiwi

                    self.kiwi = Kiwi()
                bm25 = BM25Retriever.from_documents(
                    project_chunks, preprocess_func=self._kiwi_morphs
                )
            else:
                bm25 = BM25Retriever.from_documents(project_chunks)
            bm25.k = self.settings.bm25_k
            self.bm25_retrievers[bm25_key] = bm25
        return self.dense_retrievers[project_key], self.bm25_retrievers[bm25_key]

    @staticmethod
    def rrf(dense_results, sparse_results, k):
        scores = {}
        for results in (dense_results, sparse_results):
            for rank, doc in enumerate(results, start=1):
                doc_id = doc.metadata["chunk_id"]
                if doc_id not in scores:
                    scores[doc_id] = {"doc": doc, "score": 0.0}
                scores[doc_id]["score"] += 1 / (k + rank)
        ranked = sorted(scores.values(), key=lambda item: item["score"], reverse=True)
        return [item["doc"] for item in ranked]

    def retrieve(self, query, *, bm25_mode=None):
        bm25_mode = bm25_mode or self.settings.bm25_mode
        route_started = time.perf_counter()
        try:
            project_key = self.identify_project(query)
        except ValueError:
            project_key = None  # Original behavior: ambiguous/unknown -> all 100 documents.
        route_seconds = time.perf_counter() - route_started
        cache_key = (project_key, query, bm25_mode)
        if cache_key in self.retrieval_cache:
            return self.retrieval_cache[cache_key]
        dense, bm25 = self._get_retrievers(project_key, bm25_mode)
        started = time.perf_counter()  # The first index build is excluded, as in the notebook.
        found = self.rrf(dense.invoke(query), bm25.invoke(query), self.settings.rrf_k)
        if not found:
            raise RuntimeError(f"검색 결과가 없습니다: {query}")
        ranking = self.reranker.rank(
            query, [doc.page_content for doc in found],
            top_k=len(found), batch_size=self.settings.reranker_batch_size,
        )
        ranked_chunks = [found[row["corpus_id"]] for row in ranking]
        top_contexts = ranked_chunks[:self.settings.final_context_count]
        ranked_chunk_ids = [doc.metadata["chunk_id"] for doc in ranked_chunks]
        ranked_sources = list(dict.fromkeys(doc.metadata.get("파일명", "")
                                            for doc in ranked_chunks))
        elapsed_ms = 1000 * (route_seconds + time.perf_counter() - started)
        result = (project_key, query, found, top_contexts, ranked_sources,
                  ranked_chunk_ids, elapsed_ms)
        self.retrieval_cache[cache_key] = result
        return result

    def answer(self, query, *, bm25_mode=None):
        bm25_mode = bm25_mode or self.settings.bm25_mode
        project_key, search_text, found, contexts, ranked_sources, ranked_chunk_ids, elapsed_ms = self.retrieve(
            query, bm25_mode=bm25_mode
        )
        context = "\n\n".join(
            f"[{i}] 메타데이터: {doc.metadata}\n본문: {doc.page_content}"
            for i, doc in enumerate(contexts, start=1)
        )
        prompt = ANSWER_RULES.format(context=context, query=query)
        cache_key = hashlib.sha256(
            f"{self.settings.generation_model}|{bm25_mode}|{prompt}".encode("utf-8")
        ).hexdigest()
        if cache_key in self.answer_cache:
            return dict(self.answer_cache[cache_key], answer_cached=True)
        answer = _content(self.chat_model.invoke(prompt))
        if not answer:
            raise ValueError(f"답변이 비어 있습니다: {query}")
        record = {
            "query": query, "answer_cached": False, "bm25_mode": bm25_mode,
            "identified_agency": project_key[0] if project_key else "",
            "identified_project": project_key[1] if project_key else "",
            "retrieval_scope": "identified_project" if project_key else "all_documents",
            "retrieved_sources": [doc.metadata.get("파일명", "") for doc in contexts],
            "ranked_document_sources": ranked_sources,
            "ranked_chunk_ids": ranked_chunk_ids,
            "retrieval_time_ms": elapsed_ms,
            "search_text": search_text, "answer": answer,
            "contexts": [doc.page_content for doc in contexts],
            "chunk_ids": [doc.metadata["chunk_id"] for doc in contexts],
            "rrf_chunk_ids": [doc.metadata["chunk_id"] for doc in found],
        }
        self.answer_cache[cache_key] = record
        return record
