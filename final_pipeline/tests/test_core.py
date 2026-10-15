import json
import math
import os
import struct
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from rfp_rag.config import RagSettings
from rfp_rag.data import (EVALUATION_QUESTION_IDS, load_benchmark, normalize_filename, parse_files,
                          select_evaluation_questions)
from rfp_rag.evaluation import (build_chunk_evidence_index, match_evidence_chunks,
                                retrieval_scores, evaluate_questions)
from rfp_rag.hwp_parser_fix import apply_hwp_parser_fix
from rfp_rag.pipeline import RagPipeline
from run import run_ragas


class CoreTests(unittest.TestCase):
    def test_hwp_parser_fix_moves_past_control_fields_and_keeps_hyperlinks(self):
        class FakeHWP5Reader:
            pass

        package = ModuleType("hwp_hwpx_parser")
        package.__path__ = []
        module = ModuleType("hwp_hwpx_parser.hwp5")
        module.HWP5Reader = FakeHWP5Reader
        module.CTRL_ID_HYPERLINK = 0x12345678

        def field(control_id, text):
            return (struct.pack("<HI", 0x03, control_id) + b"\x00" * 8
                    + text.encode("utf-16-le") + struct.pack("<H", 0x04)
                    + b"\x00" * 8)

        data = field(module.CTRL_ID_HYPERLINK, "첫 링크")
        data += field(0x12345679, "건너뜀")
        data += field(module.CTRL_ID_HYPERLINK, "둘째 링크")
        with patch.dict("sys.modules", {
            "hwp_hwpx_parser": package, "hwp_hwpx_parser.hwp5": module,
        }):
            apply_hwp_parser_fix()
        self.assertEqual(
            FakeHWP5Reader()._extract_hyperlink_texts_from_para(data),
            ["첫 링크", "둘째 링크"],
        )

    def test_unicode_names_and_supported_files_include_previous_skipped_hwpx(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ["케빈랩 주식회사_평택시 시스템.hwpx", "다른 사업.pdf"]
            for name in names:
                (root / name).write_bytes(b"fixture")
            (root / "parsed_results").mkdir()  # This directory must not be parsed.
            with patch("rfp_rag.data._extract_file", side_effect=lambda path: f"본문 {path.name}") as extractor:
                parsed = parse_files(root, {normalize_filename(name) for name in names})
            self.assertEqual(set(parsed), set(names))
            self.assertEqual(extractor.call_count, 2)
            self.assertEqual(normalize_filename("케빈랩.hwpx"), normalize_filename("케빈랩.hwpx"))

    def test_benchmark_has_100_documents_and_three_questions_each(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dev_benchmark_v1.1.json"
            rows = [
                {"id": f"DOC{doc:03d}-Q{question:02d}", "document_id": f"DOC{doc:03d}",
                 "source_filename": f"문서_{doc}.pdf", "project_name": f"사업 {doc}",
                 "question": "질문", "reference_answer": "답변", "evidence": [{"text": "근거"}]}
                for doc in range(1, 101) for question in range(1, 4)
            ]
            path.write_text(json.dumps({"metadata": {"version": "1.1"}, "items": rows}), encoding="utf-8")
            result = load_benchmark(path)
            self.assertEqual(len(result), 300)
            self.assertEqual(result[0]["evidence_quotes"], ["근거"])
            selected = select_evaluation_questions(list(reversed(result)))
            expected = [f"DOC{document:03d}-Q{question:02d}"
                        for document in range(1, 101)
                        for question in range(1, 4)]
            self.assertEqual(len(selected), 300)
            self.assertEqual([item["id"] for item in selected], expected)
            with self.assertRaises(ValueError):
                select_evaluation_questions(result[1:])

    def test_rrf_preserves_dense_first_tie_order_and_deduplicates_chunks(self):
        docs = [SimpleNamespace(metadata={"chunk_id": i}) for i in (1, 2, 3)]
        ranked = RagPipeline.rrf(docs[:2], docs[1:], 60)
        self.assertEqual([doc.metadata["chunk_id"] for doc in ranked], [2, 1, 3])

    def test_chunk_ranking_scores_relevant_chunk_ids(self):
        scores = retrieval_scores([3, 2, 4, 1], {1, 2})
        self.assertEqual(scores["chunk_hit_at_1"], 0)
        self.assertEqual(scores["chunk_hit_at_5"], 1)
        self.assertEqual(scores["chunk_recall_at_5"], 1)
        self.assertEqual(scores["chunk_mrr"], 0.5)
        expected_dcg = 1 / math.log2(3) + 1 / math.log2(5)
        self.assertAlmostEqual(scores["chunk_ndcg_at_5"], expected_dcg / (1 + 1 / math.log2(3)))

    def test_evidence_quotes_match_only_the_correct_source(self):
        chunks = [
            SimpleNamespace(page_content="복수의결권 주식 발행 보고", metadata={"파일명": "원문.hwp", "chunk_id": 1}),
            SimpleNamespace(page_content="다른 내용", metadata={"파일명": "원문.hwp", "chunk_id": 2}),
            SimpleNamespace(page_content="복수의결권 주식 발행 보고", metadata={"파일명": "다른문서.pdf", "chunk_id": 3}),
        ]
        index = build_chunk_evidence_index(chunks)
        item = {"id": "Q1", "source_filename": "원문.hwp",
                "evidence_quotes": ["복수의결권주식 발행 보고", "원문에 없는 문장"]}
        matches = match_evidence_chunks(item, index)
        self.assertEqual(matches, [{1}, set()])
        self.assertFalse(all(matches))

    def test_evaluation_writes_chunk_scores_and_leaves_unaligned_scores_blank(self):
        class FakeEmbeddings:
            def embed_documents(self, texts):
                return [[1.0, 0.0] for _ in texts]

        chunk = SimpleNamespace(
            page_content="확인된 근거", metadata={"파일명": "원문.hwp", "chunk_id": 7},
        )
        record = {
            "answer": "답변: 확인된 근거", "contexts": [chunk.page_content],
            "chunk_ids": [7], "ranked_chunk_ids": [7],
            "retrieval_time_ms": 12.0,
        }
        pipeline = SimpleNamespace(
            chunks=[chunk], settings=RagSettings(), embeddings=FakeEmbeddings(),
            answer=lambda _: record,
        )
        judge = SimpleNamespace(score=lambda *_: (1.0, "일치"))
        item = {
            "id": "Q1", "document_id": "DOC001", "project": "사업", "source_filename": "원문.hwp",
            "source_length_chars": 100, "query": "질문", "reference": "확인된 근거",
            "evidence_quotes": ["확인된 근거", "누락된 근거"],
        }
        with tempfile.TemporaryDirectory() as directory:
            scores, _ = evaluate_questions(pipeline, [item], Path(directory), judge=judge)
            self.assertFalse(scores.loc[0, "chunk_evidence_aligned"])
            self.assertTrue(math.isnan(scores.loc[0, "chunk_mrr"]))
            self.assertEqual(scores.loc[0, "evidence_quote_recall_in_context"], 0.5)
            self.assertNotIn("document_mrr", scores)
            self.assertTrue((Path(directory) / "chunk_evidence_alignment.jsonl").is_file())

    def test_project_routing_matches_full_title(self):
        document = SimpleNamespace(
            metadata={"발주 기관": "평택시", "사업명": "평택시 스마트시티 영상감시 시스템", "chunk_id": 0},
            page_content="본문",
        )
        with patch.object(RagPipeline, "_device", return_value="cpu"):
            pipeline = RagPipeline(
                [document], RagSettings(), embeddings=object(), chat_model=object(),
                reranker=object(), kiwi=object(),
            )
        self.assertEqual(
            pipeline.identify_project("평택시 스마트시티 영상감시 시스템의 기간은?"),
            ("평택시", "평택시 스마트시티 영상감시 시스템"),
        )

    def test_default_generation_uses_claude_haiku(self):
        calls = []
        module = ModuleType("langchain_anthropic")

        class FakeChatAnthropic:
            def __init__(self, **kwargs):
                calls.append(kwargs)

        module.ChatAnthropic = FakeChatAnthropic
        document = SimpleNamespace(
            metadata={"발주 기관": "기관", "사업명": "사업", "chunk_id": 0},
            page_content="본문",
        )
        with patch.dict("sys.modules", {"langchain_anthropic": module}), \
                patch.object(RagPipeline, "_device", return_value="cpu"):
            pipeline = RagPipeline(
                [document], RagSettings(), embeddings=object(), reranker=object(), kiwi=object(),
            )
        self.assertIsInstance(pipeline.chat_model, FakeChatAnthropic)
        self.assertEqual(calls[0]["model"], "claude-haiku-4-5-20251001")
        self.assertEqual(calls[0]["max_tokens"], 4096)

    def test_ragas_uses_gemini_for_all_300_questions(self):
        calls = []
        module = ModuleType("langchain_google_genai")

        class FakeChatGoogleGenerativeAI:
            def __init__(self, **kwargs):
                calls.append(kwargs)

        class FakeScores:
            def __getitem__(self, key):
                self_id = list(EVALUATION_QUESTION_IDS)
                return SimpleNamespace(tolist=lambda: self_id)

        class FakeResult:
            def select_dtypes(self, kind):
                return SimpleNamespace(mean=lambda: SimpleNamespace(to_string=lambda: "ok"))

        module.ChatGoogleGenerativeAI = FakeChatGoogleGenerativeAI
        ids = list(EVALUATION_QUESTION_IDS)
        samples = [{"user_input": question_id} for question_id in ids]
        with patch.dict("sys.modules", {"langchain_google_genai": module}), \
                patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}), \
                patch("rfp_rag.ragas_eval.evaluate_ragas", return_value=FakeResult()) as evaluate:
            run_ragas(samples, ids, FakeScores(), RagSettings(), Path("results"))
            self.assertEqual(evaluate.call_args.args[0], samples)
            self.assertEqual(evaluate.call_args.args[1], ids)
            with self.assertRaisesRegex(ValueError, "300개"):
                run_ragas(samples[:-1], ids[:-1], FakeScores(), RagSettings(), Path("results"))
        self.assertEqual(calls[0]["model"], "gemini-3.1-flash-lite")
        self.assertEqual(calls[0]["timeout"], 300)


if __name__ == "__main__":
    unittest.main()
