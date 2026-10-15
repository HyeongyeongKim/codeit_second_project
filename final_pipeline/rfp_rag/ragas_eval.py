"""The notebook's three RAGAS metrics, without fake VertexAI modules."""

from __future__ import annotations

from pathlib import Path


def evaluate_ragas(samples: list[dict], ids: list[str], base_scores,
                   chat_model, settings, out_dir: Path):
    import pandas as pd
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import FactualCorrectness, Faithfulness, LLMContextRecall
    from ragas.run_config import RunConfig

    if len(samples) != len(ids) or len(base_scores) != len(ids):
        raise ValueError("기본 평가와 RAGAS 입력 문항 수가 다릅니다.")
    result = evaluate(
        dataset=EvaluationDataset.from_list(samples),
        metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness(mode="f1")],
        llm=LangchainLLMWrapper(chat_model, bypass_temperature=True, bypass_n=True),
        run_config=RunConfig(
            max_workers=settings.ragas_max_workers,
            max_retries=settings.ragas_max_retries,
            timeout=settings.ragas_timeout,
        ),
        raise_exceptions=True,
        show_progress=True,
    )
    frame = result.to_pandas()
    if len(frame) != len(ids):
        raise RuntimeError("RAGAS 결과 문항 수가 다릅니다.")
    frame.insert(0, "id", ids)
    columns = {
        "context_recall": next((col for col in frame if col in ("context_recall", "llm_context_recall")), None),
        "faithfulness": next((col for col in frame if col == "faithfulness"), None),
        "factual_correctness": next((col for col in frame if col.startswith("factual_correctness")), None),
    }
    if any(column is None for column in columns.values()):
        raise RuntimeError(f"RAGAS 지표 열이 없습니다: 실제 열={list(frame)}")
    scores = frame[["id", *columns.values()]].rename(
        columns={source: target for target, source in columns.items()}
    )
    for key in columns:
        scores[key] = pd.to_numeric(scores[key], errors="coerce")
        if scores[key].isna().any() or not scores[key].between(0, 1).all():
            raise RuntimeError(f"RAGAS 지표에 누락 또는 범위 오류가 있습니다: {key}")
    if scores["id"].duplicated().any() or set(scores["id"]) != set(base_scores["id"]):
        raise RuntimeError("RAGAS 결과 ID가 기본 평가와 다릅니다.")
    frame.to_csv(out_dir / "ragas_results.csv", index=False, encoding="utf-8-sig")
    merged = base_scores.merge(scores, on="id", validate="one_to_one", how="inner")
    merged.to_csv(out_dir / "all_scores.csv", index=False, encoding="utf-8-sig")
    return merged
