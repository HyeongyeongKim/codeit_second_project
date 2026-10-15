"""Single source of truth for the notebook's experiment settings."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RagSettings:
    # These defaults are copied from the supplied notebook. CLI flags may override them.
    chunk_size: int = 700
    chunk_overlap: int = 100
    embedding_model: str = "dragonkue/BGE-m3-ko"
    embedding_batch_size: int = 16
    normalize_embeddings: bool = True
    embedding_trust_remote_code: bool = True
    bm25_mode: str = "kiwi"
    dense_k: int = 20
    bm25_k: int = 20
    rrf_k: int = 60
    reranker_model: str = "Alibaba-NLP/gte-multilingual-reranker-base"
    reranker_batch_size: int = 8
    final_context_count: int = 6
    search_method: str = "basic"
    generation_model: str = "gpt-5-mini-2025-08-07"
    generation_timeout: int = 180
    generation_retries: int = 4
    judge_model: str = "gpt-5-mini-2025-08-07"
    judge_reasoning_effort: str = "low"
    ragas_model: str = "gpt-5-mini-2025-08-07"
    ragas_max_workers: int = 1
    ragas_max_retries: int = 2
    ragas_timeout: int = 300

    def validate(self) -> None:
        if self.chunk_size <= 0 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("청킹 크기와 겹침 크기를 확인하세요.")
        if self.bm25_mode not in {"kiwi", "current"}:
            raise ValueError("bm25_mode는 kiwi 또는 current여야 합니다.")
        if self.search_method not in {"basic", "hyde"}:
            raise ValueError("search_method는 basic 또는 hyde여야 합니다.")
        for name in ("dense_k", "bm25_k", "rrf_k", "reranker_batch_size", "final_context_count"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name}은 양수여야 합니다.")

    def to_dict(self) -> dict:
        return asdict(self)
