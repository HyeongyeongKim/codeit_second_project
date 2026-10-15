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
                          prepare_corpus, select_evaluation_questions)
from rfp_rag.evaluation import (build_chunk_evidence_index, match_evidence_chunks,
                                retrieval_scores, evaluate_questions, GeminiJudge)
from rfp_rag.hwp_parser_fix import apply_hwp_parser_fix
from rfp_rag.pipeline import RagPipeline
from run import parse_document_numbers, run_ragas


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
                        for document in (1, 10, 20, 30, 40, 50, 60, 70, 80, 90)
                        for question in range(1, 4)]
            self.assertEqual(len(selected), 30)
            self.assertEqual([item["id"] for item in selected], expected)
            self.assertEqual(len(select_evaluation_questions(result, range(1, 101))), 300)
            with self.assertRaises(ValueError):
                select_evaluation_questions(result[1:])

    def test_selected_documents_alone_are_loaded_and_chunked(self):
        items = [
            {"id": f"DOC{doc:03d}-Q{question:02d}", "document_id": f"DOC{doc:03d}",
             "source_filename": f"문서_{doc}.pdf"}
            for doc in range(1, 101) for question in range(1, 4)
        ]
        requested = []

        def fake_parse(_, required, refresh_cache=False):
            requested.append(required)
            return {f"문서_{doc}.pdf": "본문" for doc in (1, 10)}

        def fake_load(parsed, _):
            return [SimpleNamespace(page_content=text, metadata={"파일명": name})
                    for name, text in parsed.items()]

        class FakeSplitter:
            def __init__(self, **kwargs):
                pass

            def split_documents(self, documents):
                return documents

        splitter_module = ModuleType("langchain_text_splitters")
        splitter_module.RecursiveCharacterTextSplitter = FakeSplitter
        with patch.dict("sys.modules", {"langchain_text_splitters": splitter_module}), \
                patch("rfp_rag.data.load_benchmark", return_value=items), \
                patch("rfp_rag.data.parse_files", side_effect=fake_parse), \
                patch("rfp_rag.data.load_documents", side_effect=fake_load):
            selected, documents, chunks = prepare_corpus(
                Path("data"), Path("data_list.csv"), Path("benchmark.json"),
                RagSettings(document_numbers=(1, 10)),
            )
        self.assertEqual(len(selected), 6)
        self.assertEqual(len(documents), 2)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(requested, [{"문서_1", "문서_10"}])
        self.assertEqual([chunk.metadata["chunk_id"] for chunk in chunks], [0, 1])

    def test_document_number_option_accepts_selection_or_all(self):
        self.assertEqual(parse_document_numbers("1,10,20"), (1, 10, 20))
        self.assertEqual(len(parse_document_numbers("all")), 100)
        with self.assertRaises(ValueError):
            RagSettings(document_numbers=(1, 1)).validate()
        with self.assertRaises(ValueError):
            RagSettings(document_numbers=(0,)).validate()

    def test_rrf_preserves_dense_first_tie_order_and_deduplicates_chunks(self):
        docs = [SimpleNamespace(metadata={"chunk_id": i}) for i in (1, 2, 3)]
        ranked = RagPipeline.rrf(docs[:2], docs[1:], 60)
        self.assertEqual([doc.metadata["chunk_id"] for doc in ranked], [2, 1, 3])

    def test_chunk_ranking_scores_relevant_chunk_ids(self):
        scores = retrieval_scores([3, 2, 4, 1], {1, 2})
        self.assertEqual(scores, {"chunk_hit_at_5": 1.0, "chunk_mrr": 0.5})
        self.assertEqual(retrieval_scores([3, 4, 5, 6, 7, 2], {2}),
                         {"chunk_hit_at_5": 0.0, "chunk_mrr": 1 / 6})

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
        chunk = SimpleNamespace(
            page_content="확인된 근거", metadata={"파일명": "원문.hwp", "chunk_id": 7},
        )
        record = {
            "answer": "답변: 확인된 근거", "contexts": [chunk.page_content],
            "chunk_ids": [7], "ranked_chunk_ids": [7],
            "retrieval_time_ms": 12.0,
        }
        pipeline = SimpleNamespace(
            chunks=[chunk], settings=RagSettings(),
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
            self.assertTrue(math.isnan(scores.loc[0, "chunk_hit_at_5"]))
            self.assertEqual(scores.loc[0, "llm_as_judge"], 1)
            self.assertNotIn("semantic_similarity", scores)
            self.assertNotIn("document_mrr", scores)
            self.assertTrue((Path(directory) / "chunk_evidence_alignment.jsonl").is_file())

    def test_gemini_judge_uses_flash_lite_and_caches_the_score(self):
        constructor_calls = []
        prompts = []
        module = ModuleType("langchain_google_genai")

        class FakeChatGoogleGenerativeAI:
            def __init__(self, **kwargs):
                constructor_calls.append(kwargs)

            def invoke(self, prompt):
                prompts.append(prompt)
                return SimpleNamespace(content='{"score": 0.75, "reason": "핵심 사실 일치"}')

        module.ChatGoogleGenerativeAI = FakeChatGoogleGenerativeAI
        with patch.dict("sys.modules", {"langchain_google_genai": module}):
            judge = GeminiJudge(RagSettings())
            self.assertEqual(judge.score("질문", "기준", "답변"), (0.75, "핵심 사실 일치"))
            self.assertEqual(judge.score("질문", "기준", "답변"), (0.75, "핵심 사실 일치"))
        self.assertEqual(constructor_calls[0]["model"], "gemini-3.1-flash-lite")
        self.assertEqual(len(prompts), 1)

    def test_gemini_judge_rejects_non_gemini_model(self):
        with self.assertRaisesRegex(ValueError, "Gemini"):
            GeminiJudge(RagSettings(judge_model="gpt-5-mini"), client=object())

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

    def test_ragas_uses_gemini_for_the_selected_30_questions(self):
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
            def __getitem__(self, keys):
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
            with self.assertRaisesRegex(ValueError, "30개"):
                run_ragas(samples[:-1], ids[:-1], FakeScores(), RagSettings(), Path("results"))
        self.assertEqual(calls[0]["model"], "gemini-3.1-flash-lite")
        self.assertEqual(calls[0]["timeout"], 300)


if __name__ == "__main__":
    unittest.main()
