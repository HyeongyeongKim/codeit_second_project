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
    parser.add_argument("--generation-model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--generation-max-tokens", type=int, default=4096)
    parser.add_argument("--judge-model", default="gpt-5-mini-2025-08-07")
    parser.add_argument("--ragas-model", help="RAGAS 평가용 Gemini 모델 ID. 기본값: gemini-3.1-flash-lite")
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
        generation_model=args.generation_model,
        generation_max_tokens=args.generation_max_tokens,
        judge_model=args.judge_model,
        ragas_model=args.ragas_model or "gemini-3.1-flash-lite",
    )
    settings.validate()
    return settings


def require_api_key(name: str) -> None:
    if not os.getenv(name):
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
    if not os.getenv(name):
        try:
            from google.colab import userdata

            value = userdata.get(name)
        except Exception:
            value = None
        if value:
            os.environ[name] = value
    if not os.getenv(name):
        raise RuntimeError(f"{name}를 final_pipeline/.env, 환경 변수 또는 Colab 보안 비밀에 설정하세요.")


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
    from rfp_rag.data import EVALUATION_QUESTION_IDS

    if ids != list(EVALUATION_QUESTION_IDS) or len(samples) != len(ids):
        raise ValueError("RAGAS 입력은 지정한 10개 문서의 질문 30개여야 합니다.")
    if base_scores["id"].tolist() != ids:
        raise ValueError("RAGAS 입력과 기본 평가의 질문 ID 또는 순서가 다릅니다.")
    require_api_key("GOOGLE_API_KEY")
    from langchain_google_genai import ChatGoogleGenerativeAI

    from rfp_rag.ragas_eval import evaluate_ragas

    chat_model = ChatGoogleGenerativeAI(
        model=settings.ragas_model,
        timeout=settings.ragas_timeout,
        max_retries=settings.ragas_max_retries,
    )
    result = evaluate_ragas(samples, ids, base_scores, chat_model, settings, out_dir)
    print("평균 점수:\n", result.select_dtypes("number").mean().to_string())


def main():
    args = parse_args()
    settings = settings_from_args(args)
    if args.command == "show-config":
        print(json.dumps(settings.to_dict(), ensure_ascii=False, indent=2))
        return
    # The notebook hit LangSmith's 429 ingest limit. Scoring is local; no tracing is needed.
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_TRACING"] = "false"

    if args.command == "ragas":
        import pandas as pd

        saved_settings = json.loads((args.out_dir / "run_config.json").read_text(encoding="utf-8"))
        settings = RagSettings(**saved_settings)
        from dataclasses import replace

        ragas_model = args.ragas_model or (
            settings.ragas_model if settings.ragas_model.startswith("gemini-")
            else RagSettings().ragas_model
        )
        settings = replace(settings, ragas_model=ragas_model)
        settings.validate()
        samples = json.loads((args.out_dir / "ragas_samples.json").read_text(encoding="utf-8"))
        base_scores = pd.read_csv(args.out_dir / "base_scores.csv", encoding="utf-8-sig")
        with (args.out_dir / "base_answers.jsonl").open(encoding="utf-8") as stream:
            answers = [json.loads(line) for line in stream if line.strip()]
        if ([answer.get("id") for answer in answers] != base_scores["id"].tolist()
                or [answer.get("query") for answer in answers]
                != [sample.get("user_input") for sample in samples]):
            raise ValueError("저장된 검색·생성 질문과 RAGAS 질문의 ID 또는 내용이 다릅니다.")
        run_ragas(samples, base_scores["id"].tolist(), base_scores, settings, args.out_dir)
        (args.out_dir / "run_config.json").write_text(
            json.dumps(settings.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return

    require_api_key("ANTHROPIC_API_KEY" if settings.generation_model.startswith("claude-")
                    else "OPENAI_API_KEY")
    if args.command == "evaluate":
        require_api_key("OPENAI_API_KEY")  # GPT-5 mini judges the generated answers.
        if not args.skip_ragas:
            require_api_key("GOOGLE_API_KEY")

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

    from rfp_rag.data import select_evaluation_questions

    items = select_evaluation_questions(items)
    print(f"평가 질문: {len(items)}문항 / 문서 ID: "
          f"{', '.join(dict.fromkeys(item['document_id'] for item in items))}")
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
