"""Single source of truth for the notebook's experiment settings."""

from dataclasses import asdict, dataclass


# Edit this one list to change which documents are parsed, embedded, and evaluated.
DEFAULT_DOCUMENT_NUMBERS = (1, 10, 20, 30, 40, 50, 60, 70, 80, 90)


@dataclass(frozen=True)
class RagSettings:
    # Retrieval defaults preserve the supplied notebook; generation uses Claude Haiku 4.5.
    # The same selection controls parsing, embedding, search corpus, answers, and evaluation.
    document_numbers: tuple[int, ...] = DEFAULT_DOCUMENT_NUMBERS
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
    generation_model: str = "claude-haiku-4-5-20251001"
    generation_max_tokens: int = 4096
    generation_timeout: int = 180
    generation_retries: int = 4
    judge_model: str = "gemini-3.1-flash-lite"
    ragas_model: str = "gemini-3.1-flash-lite"
    ragas_max_workers: int = 1
    ragas_max_retries: int = 2
    ragas_timeout: int = 300

    def validate(self) -> None:
        numbers = self.document_numbers
        if (not isinstance(numbers, (tuple, list)) or not numbers
                or any(isinstance(number, bool) or not isinstance(number, int)
                       or not 1 <= number <= 100 for number in numbers)
                or len(numbers) != len(set(numbers))):
            raise ValueError("document_numbers는 중복 없는 1~100 범위의 문서 번호 목록이어야 합니다.")
        if self.chunk_size <= 0 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("청킹 크기와 겹침 크기를 확인하세요.")
        if self.bm25_mode not in {"kiwi", "current"}:
            raise ValueError("bm25_mode는 kiwi 또는 current여야 합니다.")
        if self.generation_max_tokens <= 0:
            raise ValueError("generation_max_tokens는 양수여야 합니다.")
        if not self.ragas_model.startswith("gemini-"):
            raise ValueError("ragas_model에는 Gemini 모델 ID를 지정하세요.")
        for name in ("dense_k", "bm25_k", "rrf_k", "reranker_batch_size", "final_context_count"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name}은 양수여야 합니다.")

    def to_dict(self) -> dict:
        return asdict(self)
