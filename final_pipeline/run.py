"""Run the modular final RFP RAG pipeline from a terminal or Colab."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rfp_rag.config import RagSettings


def parse_args():
    parser = argparse.ArgumentParser(description="RFP RAG: Dense + Kiwi BM25 + RRF + reranker")
    parser.add_argument("command", choices=("show-config", "answer", "evaluate", "ragas"))
    parser.add_argument("--data-dir", type=Path, default=Path("/content/drive/MyDrive/data"))
    parser.add_argument("--metadata-csv", type=Path, default=Path("/content/drive/MyDrive/data_list.csv"))
    parser.add_argument("--benchmark", type=Path, default=Path("/content/drive/MyDrive/dev_benchmark_v1.1.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--aliases", type=Path, help="선택 사항: {줄임말: [발주 기관, 정식 사업명]} JSON")
    parser.add_argument("--question", help="answer 명령에 사용할 질문")
    parser.add_argument("--refresh-cache", action="store_true", help="기존 문서 추출 캐시를 다시 생성")
    parser.add_argument("--limit", type=int, help="평가할 앞쪽 문항 수. 기본값은 전체 300문항")
    parser.add_argument("--skip-ragas", action="store_true", help="기본 평가만 실행하고 RAGAS는 나중에 실행")
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--embedding-model", default="dragonkue/BGE-m3-ko")
    parser.add_argument("--bm25-mode", choices=("kiwi", "current"), default="kiwi")
    parser.add_argument("--dense-k", type=int, default=20)
    parser.add_argument("--bm25-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--reranker-model", default="Alibaba-NLP/gte-multilingual-reranker-base")
    parser.add_argument("--final-context-count", type=int, default=6)
    parser.add_argument("--search-method", choices=("basic", "hyde"), default="basic")
    parser.add_argument("--generation-model", default="gpt-5-mini-2025-08-07")
    parser.add_argument("--judge-model", default="gpt-5-mini-2025-08-07")
    parser.add_argument("--ragas-model", help="RAGAS 평가 모델. 기본값: gpt-5-mini-2025-08-07")
    return parser.parse_args()


def settings_from_args(args) -> RagSettings:
    settings = RagSettings(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
        bm25_mode=args.bm25_mode,
        dense_k=args.dense_k,
        bm25_k=args.bm25_k,
        rrf_k=args.rrf_k,
        reranker_model=args.reranker_model,
        final_context_count=args.final_context_count,
        search_method=args.search_method,
        generation_model=args.generation_model,
        judge_model=args.judge_model,
        ragas_model=args.ragas_model or "gpt-5-mini-2025-08-07",
    )
    settings.validate()
    return settings


def require_openai_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        try:
            from google.colab import userdata

            value = userdata.get("OPENAI_API_KEY")
        except Exception:
            value = None
        if value:
            os.environ["OPENAI_API_KEY"] = value
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY를 환경 변수 또는 Colab 보안 비밀에 설정하세요.")


def load_aliases(path: Path | None) -> dict[str, tuple[str, str]]:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict) or any(
        not isinstance(alias, str) or not isinstance(target, list)
        or len(target) != 2 or not all(isinstance(part, str) for part in target)
        for alias, target in raw.items()
    ):
        raise ValueError("별칭 JSON은 {줄임말: [발주 기관, 정식 사업명]} 형식이어야 합니다.")
    return {alias: tuple(target) for alias, target in raw.items()}


def run_ragas(samples, ids, base_scores, settings, out_dir):
    from langchain_openai import ChatOpenAI

    from rfp_rag.ragas_eval import evaluate_ragas

    chat_model = ChatOpenAI(
        model=settings.ragas_model,
        timeout=settings.generation_timeout,
        max_retries=settings.generation_retries,
    )
    result = evaluate_ragas(samples, ids, base_scores, chat_model, settings, out_dir)
    print("평균 점수:\n", result.select_dtypes("number").mean().to_string())


def main():
    args = parse_args()
    settings = settings_from_args(args)
    if args.command == "show-config":
        print(json.dumps(settings.to_dict(), ensure_ascii=False, indent=2))
        return
    require_openai_key()
    # The notebook hit LangSmith's 429 ingest limit. Scoring is local; no tracing is needed.
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_TRACING"] = "false"

    if args.command == "ragas":
        import pandas as pd

        saved_settings = json.loads((args.out_dir / "run_config.json").read_text(encoding="utf-8"))
        settings = RagSettings(**saved_settings)
        if args.ragas_model:
            from dataclasses import replace

            settings = replace(settings, ragas_model=args.ragas_model)
        samples = json.loads((args.out_dir / "ragas_samples.json").read_text(encoding="utf-8"))
        base_scores = pd.read_csv(args.out_dir / "base_scores.csv", encoding="utf-8-sig")
        run_ragas(samples, base_scores["id"].tolist(), base_scores, settings, args.out_dir)
        return

    from rfp_rag.data import prepare_corpus
    from rfp_rag.pipeline import RagPipeline

    items, documents, chunks = prepare_corpus(
        args.data_dir, args.metadata_csv, args.benchmark, settings,
        refresh_cache=args.refresh_cache,
    )
    pipeline = RagPipeline(chunks, settings, aliases=load_aliases(args.aliases))
    print(f"준비 완료: 문서 {len(documents)}개, 청크 {len(chunks)}개")

    if args.command == "answer":
        if not args.question:
            raise ValueError("answer 명령에는 --question이 필요합니다.")
        record = pipeline.answer(args.question)
        print(record["answer"])
        return

    if args.limit is not None:
        if not 1 <= args.limit <= len(items):
            raise ValueError("--limit는 1~300이어야 합니다.")
        items = items[:args.limit]
    from rfp_rag.evaluation import evaluate_questions

    scores, samples = evaluate_questions(pipeline, items, args.out_dir)
    (args.out_dir / "run_config.json").write_text(
        json.dumps(settings.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("기본 평가 평균:\n", scores.select_dtypes("number").mean().to_string())
    if not args.skip_ragas:
        run_ragas(samples, [item["id"] for item in items], scores, settings, args.out_dir)


if __name__ == "__main__":
    main()
