import json
import math
import struct
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from rfp_rag.config import RagSettings
from rfp_rag.data import load_benchmark, normalize_filename, parse_files
from rfp_rag.evaluation import retrieval_scores
from rfp_rag.hwp_parser_fix import apply_hwp_parser_fix
from rfp_rag.pipeline import RagPipeline


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

    def test_rrf_preserves_dense_first_tie_order_and_deduplicates_chunks(self):
        docs = [SimpleNamespace(metadata={"chunk_id": i}) for i in (1, 2, 3)]
        ranked = RagPipeline.rrf(docs[:2], docs[1:], 60)
        self.assertEqual([doc.metadata["chunk_id"] for doc in ranked], [2, 1, 3])

    def test_document_ranking_uses_unique_filenames(self):
        scores = retrieval_scores(["A.pdf", "A.pdf", "B.pdf"], "B.pdf")
        self.assertEqual(scores["document_hit_at_1"], 0)
        self.assertEqual(scores["document_hit_at_5"], 1)
        self.assertEqual(scores["document_mrr"], 0.5)
        self.assertAlmostEqual(scores["document_ndcg_at_5"], 1 / math.log2(3))

    def test_project_routing_matches_full_title(self):
        document = SimpleNamespace(
            metadata={"발주 기관": "평택시", "사업명": "평택시 스마트시티 영상감시 시스템", "chunk_id": 0},
            page_content="본문",
        )
        pipeline = RagPipeline(
            [document], RagSettings(), embeddings=object(), chat_model=object(),
            reranker=object(), kiwi=object(),
        )
        self.assertEqual(
            pipeline.identify_project("평택시 스마트시티 영상감시 시스템의 기간은?"),
            ("평택시", "평택시 스마트시티 영상감시 시스템"),
        )


if __name__ == "__main__":
    unittest.main()
