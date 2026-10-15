"""Benchmark validation, HWP/PDF extraction, metadata matching, and chunking."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from .config import RagSettings

SUPPORTED_SUFFIXES = {".pdf", ".hwp", ".hwpx"}
METADATA_COLUMNS = (
    "공고 번호", "공고 차수", "사업명", "사업 금액", "발주 기관",
    "공개 일자", "입찰 참여 시작일", "입찰 참여 마감일", "파일명",
)
EVALUATION_DOCUMENT_NUMBERS = (1, 10, 20, 30, 40, 50, 60, 70, 80, 90)


def normalize_filename(value: str) -> str:
    """Match file, CSV, and benchmark names despite Hangul normalization/spacing."""
    name = str(value)
    if Path(name).suffix.lower() in SUPPORTED_SUFFIXES:
        name = name.rsplit(".", 1)[0]
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", name)).strip()


def clean_text(value: str) -> str:
    text = value.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError("DEV 벤치마크는 metadata와 items가 있는 JSON 객체여야 합니다.")
    if payload.get("metadata", {}).get("version") != "1.1":
        raise ValueError("DEV 벤치마크 버전이 1.1인지 확인하세요.")
    records = payload["items"]
    if len(records) != 300:
        raise ValueError(f"평가 문항 수: {len(records)}/300")

    required = ("id", "document_id", "source_filename", "project_name", "question", "reference_answer")
    result = []
    for record in records:
        if not isinstance(record, dict) or not all(
            isinstance(record.get(field), str) and record[field].strip() for field in required
        ):
            raise ValueError(f"필수 평가 필드가 없거나 비어 있습니다: {record!r}")
        evidence = record.get("evidence", [])
        if not isinstance(evidence, list) or any(not isinstance(item, dict) for item in evidence):
            raise ValueError(f"evidence 형식 오류: {record['id']}")
        quotes = [item["text"].strip() for item in evidence
                  if isinstance(item.get("text"), str) and item["text"].strip()]
        result.append({
            "id": record["id"], "document_id": record["document_id"],
            "source_filename": record["source_filename"],
            "project": record["project_name"], "query": record["question"],
            "reference": record["reference_answer"],
            "question_type": record.get("question_type", ""),
            "answerable": record.get("answerable"),
            "evidence_quotes": quotes,
            "atomic_facts": record.get("atomic_facts", []),
        })
    counts = Counter(item["document_id"] for item in result)
    if len(counts) != 100 or set(counts.values()) != {3}:
        raise ValueError("100개 문서에 각 3문항이 있는지 확인하세요.")
    if len({item["id"] for item in result}) != 300:
        raise ValueError("평가 문항 ID가 중복됩니다.")
    return result


def select_evaluation_questions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use three questions from each of the ten requested documents, in ID order."""
    by_id = {item["id"]: item for item in items}
    selected = []
    for number in EVALUATION_DOCUMENT_NUMBERS:
        document_id = f"DOC{number:03d}"
        for question_number in range(1, 4):
            question_id = f"{document_id}-Q{question_number:02d}"
            item = by_id.get(question_id)
            if item is None or item["document_id"] != document_id:
                raise ValueError(f"선택한 평가 문항이 없거나 문서 ID가 다릅니다: {question_id}")
            selected.append(item)
    return selected


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"파싱 캐시는 JSON 객체여야 합니다: {path}")
    return value


def _extract_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        import pymupdf4llm

        return pymupdf4llm.to_markdown(str(path))
    from .hwp_parser_fix import apply_hwp_parser_fix

    apply_hwp_parser_fix()
    from hwp_hwpx_parser import ExtractOptions, TableStyle
    from langchain_hwp_hwpx import HwpHwpxLoader

    loader = HwpHwpxLoader(
        str(path), mode="single",
        extract_options=ExtractOptions(table_style=TableStyle.MARKDOWN),
    )
    return "\n\n".join(document.page_content for document in loader.load())


def parse_files(data_dir: Path, required_names: set[str], refresh_cache: bool = False) -> dict[str, str]:
    """Parse each required supported file once; no filename-specific skips/patches."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"문서 폴더가 없습니다: {data_dir}")
    files = sorted(
        (path for path in data_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES),
        key=lambda path: (unicodedata.normalize("NFKC", path.name).casefold(), path.name),
    )
    selected = [path for path in files if normalize_filename(path.name) in required_names]
    found = Counter(normalize_filename(path.name) for path in selected)
    duplicate = [name for name, count in found.items() if count > 1]
    if duplicate:
        raise ValueError(f"정규화 후 같은 이름의 문서가 중복됩니다: {duplicate[:10]}")
    missing = required_names - set(found)
    if missing:
        raise FileNotFoundError(f"벤치마크 문서가 data 폴더에 없습니다: {sorted(missing)[:10]}")

    cache_dir = data_dir / "parsed_results"
    success_path = cache_dir / "success_files.json"
    failed_path = cache_dir / "failed_files.json"
    success = _read_json(success_path)
    failed = _read_json(failed_path)
    parsed = {}
    for index, path in enumerate(selected, 1):
        cached = success.get(path.name)
        if not refresh_cache and isinstance(cached, str) and cached.strip():
            parsed[path.name] = cached
            if path.name in failed:
                failed.pop(path.name)
                _write_json(failed_path, failed)
            continue
        try:
            text = _extract_file(path)
            if not isinstance(text, str) or not text.strip():
                raise ValueError("본문이 비어 있습니다.")
        except Exception as exc:
            failed[path.name] = {"error": str(exc)}
            _write_json(failed_path, failed)
            raise RuntimeError(f"문서 파싱 실패 ({index}/{len(selected)}): {path.name}") from exc
        success[path.name] = text
        parsed[path.name] = text
        failed.pop(path.name, None)
        _write_json(success_path, success)
        _write_json(failed_path, failed)
    return parsed


def load_documents(parsed: dict[str, str], metadata_csv: Path):
    import pandas as pd
    from langchain_core.documents import Document

    frame = pd.read_csv(metadata_csv, encoding="utf-8-sig")
    missing_columns = set(METADATA_COLUMNS) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"data_list.csv에 없는 열: {sorted(missing_columns)}")
    by_name: dict[str, list] = {}
    for _, row in frame.iterrows():
        by_name.setdefault(normalize_filename(row["파일명"]), []).append(row)

    documents = []
    unmatched = []
    for name, text in parsed.items():
        rows = by_name.get(normalize_filename(name), [])
        if not rows:
            unmatched.append(name)
            continue
        identities = {
            (str(row["발주 기관"]), str(row["사업명"])) for row in rows
        }
        if len(identities) != 1:
            raise ValueError(f"파일의 사업 메타데이터가 여러 개입니다: {name}")
        row = rows[0]
        metadata = {column: "" if pd.isna(row[column]) else str(row[column])
                    for column in METADATA_COLUMNS}
        documents.append(Document(page_content=clean_text(text), metadata=metadata))
    if unmatched:
        raise ValueError(f"data_list.csv와 연결되지 않은 문서: {unmatched[:10]}")
    if not documents:
        raise ValueError("파싱·메타데이터 연결에 성공한 문서가 없습니다.")
    return documents


def prepare_corpus(data_dir: Path, metadata_csv: Path, benchmark_path: Path,
                   settings: RagSettings, refresh_cache: bool = False):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    items = load_benchmark(benchmark_path)
    required = {normalize_filename(item["source_filename"]) for item in items}
    if len(required) != 100:
        raise ValueError(f"벤치마크의 고유 정답 문서 수가 100개가 아닙니다: {len(required)}")
    parsed = parse_files(data_dir, required, refresh_cache=refresh_cache)
    documents = load_documents(parsed, metadata_csv)
    sources = {normalize_filename(doc.metadata["파일명"]) for doc in documents}
    if sources != required or len(documents) != 100:
        raise ValueError(f"벤치마크 문서 100개가 필요합니다: {len(documents)}개 로드")
    lengths = {normalize_filename(doc.metadata["파일명"]): len(doc.page_content)
               for doc in documents}
    for item in items:
        item["source_length_chars"] = lengths[normalize_filename(item["source_filename"])]
        if item["source_length_chars"] <= 0:
            raise ValueError(f"문서 길이가 0입니다: {item['source_filename']}")
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    for chunk_id, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = chunk_id
    return items, documents, chunks
