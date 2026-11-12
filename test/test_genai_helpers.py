# mypy: disable-error-code="attr-defined,name-defined,call-arg"
import asyncio
import base64
import concurrent.futures
import json
import queue
import sys
import tempfile
import threading
import unittest
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, cast
from unittest.mock import MagicMock, mock_open, patch

import pyopl.genai._strategy_base as strategy_base
import pyopl.genai.exemplar_ranking_worker as exemplar_ranking_worker
import pyopl.genai.feedback_validation as feedback_validation
from pyopl.genai import (
    genai_pricing,
    model_discovery,
    pyopl_generative,
    pyopl_generative_graphchain,
    pyopl_tree_of_thoughts,
    rag_helper,
)
from pyopl.genai._strategy_base import GenAIStrategyBase, GoogleClient, Grammar, LLMProvider, Usage

STRATEGY_MODULE_NAMES = [
    "pyopl.genai.pyopl_standard",
    "pyopl.genai.pyopl_generative",
    "pyopl.genai.pyopl_chain_of_thought",
    "pyopl.genai.pyopl_reflexion",
    "pyopl.genai.pyopl_tree_of_thoughts",
    "pyopl.genai.pyopl_chain_of_experts",
    "pyopl.genai.pyopl_cafa",
]


class TestFeedbackValidation(unittest.TestCase):
    def _files(self, directory: str) -> tuple[Path, Path]:
        model_file = Path(directory) / "model.mod"
        data_file = Path(directory) / "data.dat"
        model_file.write_text("original model", encoding="utf-8")
        data_file.write_text("original data", encoding="utf-8")
        return model_file, data_file

    def test_feedback_without_revisions_skips_compilation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_file, data_file = self._files(directory)
            with (
                patch.object(
                    feedback_validation._BASE,
                    "llm_generate_text",
                    return_value=json.dumps({"feedback": "looks good"}),
                ),
                patch.object(feedback_validation.OPLCompiler, "compile_model") as compile_model,
            ):
                result = feedback_validation.generative_feedback(
                    "review",
                    model_file,
                    data_file,
                    mode=Grammar.NONE,
                )

        self.assertEqual(result, {"feedback": "looks good"})
        compile_model.assert_not_called()

    def test_model_only_revision_compiles_with_original_data(self) -> None:
        responses = iter(
            [
                json.dumps({"feedback": "updated", "revised_model": "revised model"}),
                json.dumps({"aligned": True, "assessment": "request satisfied"}),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            model_file, data_file = self._files(directory)
            with (
                patch.object(
                    feedback_validation._BASE,
                    "llm_generate_text",
                    side_effect=lambda **_kwargs: next(responses),
                ),
                patch.object(feedback_validation.OPLCompiler, "compile_model") as compile_model,
            ):
                result = feedback_validation.generative_feedback(
                    "change model",
                    model_file,
                    data_file,
                    mode=Grammar.NONE,
                )

        compile_model.assert_called_once_with("revised model", "original data")
        self.assertEqual(result, {"feedback": "updated", "revised_model": "revised model"})

    def test_failed_compile_repairs_and_returns_changed_companion(self) -> None:
        responses = iter(
            [
                json.dumps({"feedback": "updated", "revised_model": "invalid model"}),
                json.dumps({"model": "fixed model", "data": "fixed data"}),
                json.dumps({"aligned": True, "assessment": "request satisfied"}),
            ]
        )
        compile_results = iter([RuntimeError("syntax bad"), None])

        def compile_model(_model: str, _data: str) -> None:
            result = next(compile_results)
            if result:
                raise result

        with tempfile.TemporaryDirectory() as directory:
            model_file, data_file = self._files(directory)
            with (
                patch.object(
                    feedback_validation._BASE,
                    "llm_generate_text",
                    side_effect=lambda **_kwargs: next(responses),
                ),
                patch.object(feedback_validation.OPLCompiler, "compile_model", side_effect=compile_model),
            ):
                result = feedback_validation.generative_feedback(
                    "change model",
                    model_file,
                    data_file,
                    mode=Grammar.NONE,
                    validation_iterations=2,
                )

        self.assertEqual(
            result,
            {
                "feedback": "updated",
                "revised_model": "fixed model",
                "revised_data": "fixed data",
            },
        )

    def test_exhausted_validation_withholds_revisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_file, data_file = self._files(directory)
            with (
                patch.object(
                    feedback_validation._BASE,
                    "llm_generate_text",
                    return_value=json.dumps({"feedback": "updated", "revised_data": "invalid data"}),
                ),
                patch.object(
                    feedback_validation.OPLCompiler,
                    "compile_model",
                    side_effect=RuntimeError("syntax bad"),
                ),
            ):
                result = feedback_validation.generative_feedback(
                    "change data",
                    model_file,
                    data_file,
                    mode=Grammar.NONE,
                    validation_iterations=1,
                )

        self.assertNotIn("revised_model", result)
        self.assertNotIn("revised_data", result)
        self.assertIn("withheld because validation failed", result["feedback"])
        self.assertIn("syntax bad", result["feedback"])

    def test_failed_alignment_repairs_and_revalidates(self) -> None:
        responses = iter(
            [
                json.dumps({"feedback": "updated", "revised_data": "candidate data"}),
                json.dumps({"aligned": False, "assessment": "request not satisfied"}),
                json.dumps({"model": "original model", "data": "aligned data"}),
                json.dumps({"aligned": True, "assessment": "request satisfied"}),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            model_file, data_file = self._files(directory)
            with (
                patch.object(
                    feedback_validation._BASE,
                    "llm_generate_text",
                    side_effect=lambda **_kwargs: next(responses),
                ),
                patch.object(feedback_validation.OPLCompiler, "compile_model") as compile_model,
            ):
                result = feedback_validation.generative_feedback(
                    "change data",
                    model_file,
                    data_file,
                    mode=Grammar.NONE,
                    validation_iterations=2,
                )

        self.assertEqual(compile_model.call_count, 2)
        self.assertEqual(result, {"feedback": "updated", "revised_data": "aligned data"})

    def test_malformed_alignment_response_withholds_revision(self) -> None:
        responses = iter(
            [
                json.dumps({"feedback": "updated", "revised_model": "candidate model"}),
                json.dumps({"aligned": "yes", "assessment": "invalid schema"}),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            model_file, data_file = self._files(directory)
            with (
                patch.object(
                    feedback_validation._BASE,
                    "llm_generate_text",
                    side_effect=lambda **_kwargs: next(responses),
                ),
                patch.object(feedback_validation.OPLCompiler, "compile_model"),
            ):
                result = feedback_validation.generative_feedback(
                    "change model",
                    model_file,
                    data_file,
                    mode=Grammar.NONE,
                    validation_iterations=1,
                )

        self.assertNotIn("revised_model", result)
        self.assertIn("Alignment response requires", result["feedback"])

    def test_malformed_feedback_payload_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_file, data_file = self._files(directory)
            with patch.object(
                feedback_validation._BASE,
                "llm_generate_text",
                return_value=json.dumps({"feedback": 42}),
            ):
                with self.assertRaisesRegex(RuntimeError, "field 'feedback'.*string"):
                    feedback_validation.generative_feedback(
                        "review",
                        model_file,
                        data_file,
                        mode=Grammar.NONE,
                    )

    def test_malformed_repair_payload_withholds_revision(self) -> None:
        responses = iter(
            [
                json.dumps({"feedback": "updated", "revised_model": "invalid model"}),
                json.dumps({"model": "fixed model"}),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            model_file, data_file = self._files(directory)
            with (
                patch.object(
                    feedback_validation._BASE,
                    "llm_generate_text",
                    side_effect=lambda **_kwargs: next(responses),
                ),
                patch.object(
                    feedback_validation.OPLCompiler,
                    "compile_model",
                    side_effect=RuntimeError("syntax bad"),
                ),
            ):
                result = feedback_validation.generative_feedback(
                    "change model",
                    model_file,
                    data_file,
                    mode=Grammar.NONE,
                    validation_iterations=2,
                )

        self.assertNotIn("revised_model", result)
        self.assertIn("Revision repair failed", result["feedback"])
        self.assertIn("field 'data'", result["feedback"])


class TestGenAIStrategyBaseHelpers(unittest.TestCase):
    def test_exemplar_ranking_worker_preloads_once_and_serves_multiple_queries(self) -> None:
        commands: queue.Queue[tuple[str, int | None, str | None]] = queue.Queue()
        results: queue.Queue[tuple[str, int | None, Any]] = queue.Queue()
        commands.put(("rank", 1, "first"))
        commands.put(("rank", 2, "second"))
        commands.put(("stop", None, None))

        with (
            patch.object(rag_helper, "_load_model", return_value=object()) as load_model,
            patch.object(
                rag_helper,
                "rank_problem_descriptions",
                side_effect=lambda query, **_kwargs: [{"path": f"{query}.md"}],
            ) as rank,
        ):
            exemplar_ranking_worker._run_ranking_worker(["models"], commands, results)

        self.assertEqual(results.get_nowait(), ("ready", None, None))
        self.assertEqual(results.get_nowait(), ("success", 1, [{"path": "first.md"}]))
        self.assertEqual(results.get_nowait(), ("success", 2, [{"path": "second.md"}]))
        load_model.assert_called_once_with(rag_helper.DEFAULT_EMBEDDING_MODEL)
        self.assertEqual(rank.call_count, 2)

    def test_embedding_model_is_loaded_once_for_concurrent_callers(self) -> None:
        load_count = 0
        callers_ready = threading.Barrier(4)

        class FakeSentenceTransformer:
            def __init__(self, _model_name: str) -> None:
                nonlocal load_count
                load_count += 1

        def load() -> object:
            callers_ready.wait()
            return rag_helper._load_model("test-model")

        rag_helper._load_model_cached.cache_clear()
        fake_module = SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
        with (
            patch.dict(sys.modules, {"sentence_transformers": fake_module}),
            concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor,
        ):
            models = list(executor.map(lambda _index: load(), range(4)))
        rag_helper._load_model_cached.cache_clear()

        self.assertEqual(load_count, 1)
        self.assertTrue(all(model is models[0] for model in models))

    def test_extract_json_from_fence_and_prose(self) -> None:
        fenced = 'prefix```json\n{"model": {"name": "x"}, "ok": true}\n```suffix'
        prose = 'Here is the answer: {"outer": {"inner": 3}, "items": [1, 2]}. Done.'

        self.assertEqual(GenAIStrategyBase.extract_json_from_markdown(fenced), '{"model": {"name": "x"}, "ok": true}')
        self.assertEqual(GenAIStrategyBase.json_loads_relaxed(prose), {"outer": {"inner": 3}, "items": [1, 2]})
        self.assertEqual(GenAIStrategyBase.extract_json_from_markdown("plain answer"), "plain answer")
        self.assertEqual(GenAIStrategyBase.extract_json_from_markdown("partial {json"), "partial {json")

    def test_normalize_prompt_input_accepts_text_and_images(self) -> None:
        text, images = GenAIStrategyBase.normalize_prompt_input(
            cast(
                strategy_base.PromptWithImages,
                {
                    "text": "build a model",
                    "images": [
                        Path("diagram.png"),
                        {"url": "https://example.test/img.png", "mime_type": "image/png", "extra": "ignored"},
                    ],
                },
            )
        )

        self.assertEqual(text, "build a model")
        self.assertEqual(images, [{"path": "diagram.png"}, {"url": "https://example.test/img.png", "mime_type": "image/png"}])

    def test_prompt_normalization_coerces_text_and_discards_invalid_images(self) -> None:
        self.assertEqual(GenAIStrategyBase.normalize_prompt_input("plain text"), ("plain text", []))
        self.assertEqual(GenAIStrategyBase._normalize_images(None), [])
        self.assertEqual(GenAIStrategyBase.normalize_prompt_input({}), ("", []))
        self.assertEqual(
            GenAIStrategyBase.normalize_prompt_input(cast(Any, {"text": 12, "image": Path("diagram.png")})),
            ("12", [{"path": "diagram.png"}]),
        )
        self.assertEqual(
            GenAIStrategyBase._normalize_images("diagram.png"),
            [{"path": "diagram.png"}],
        )
        self.assertEqual(
            GenAIStrategyBase._normalize_images({"data_base64": "YWJj"}),
            [{"data_base64": "YWJj"}],
        )
        self.assertEqual(
            GenAIStrategyBase._normalize_images({"path": "diagram.png"}),
            [{"path": "diagram.png"}],
        )
        self.assertEqual(
            GenAIStrategyBase._normalize_images([{"url": 5, "extra": "ignored"}, object()]),
            [],
        )
        prompt = object()
        self.assertEqual(GenAIStrategyBase.normalize_prompt_input(cast(Any, prompt)), (str(prompt), []))

    def test_notify_handles_logger_progress_and_callback_errors(self) -> None:
        base = GenAIStrategyBase(logger=genai_pricing.logger)
        messages: list[str] = []
        with patch.object(base._logger, "debug") as debug:
            base.notify(None, "logged")
        debug.assert_called_once_with("logged")

        base.notify(messages.append, "shown")
        self.assertEqual(messages, ["shown"])
        base.notify(lambda _message: (_ for _ in ()).throw(RuntimeError("UI closed")), "ignored")

    def test_grammar_and_file_resources_are_readable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "input.txt"
            path.write_text("contents", encoding="utf-8")
            self.assertEqual(GenAIStrategyBase.read_file(str(path)), "contents")

        self.assertTrue(GenAIStrategyBase.read_pyopl_GBNF())
        self.assertTrue(GenAIStrategyBase.read_pyopl_grammar())
        self.assertEqual(
            GenAIStrategyBase.get_grammar_implementation(Grammar.CODE),
            GenAIStrategyBase.read_pyopl_code(),
        )

    def test_safe_read_text_limits_content_and_handles_read_errors(self) -> None:
        base = GenAIStrategyBase(logger=genai_pricing.logger, few_shot_max_chars=4)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "description.md"
            path.write_text("abcdef", encoding="utf-8")
            self.assertEqual(base.safe_read_text(path), "abcd")
            self.assertEqual(base.safe_read_text(path, max_chars=7), "abcdef")
            self.assertEqual(base.safe_read_text(Path(td) / "missing.md"), "")

    def test_default_model_directory_falls_back_when_resources_fail(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with (
                patch.object(strategy_base, "files", side_effect=RuntimeError("resources unavailable")),
                patch.object(strategy_base.Path, "cwd", return_value=Path(td)),
            ):
                model_dirs = GenAIStrategyBase.default_models_dirs()

        self.assertEqual(model_dirs, [Path(strategy_base.__file__).parent / "opl_models"])

    def test_default_model_directory_falls_back_when_packaged_root_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with (
                patch.object(strategy_base, "files", return_value=root),
                patch.object(strategy_base.Path, "cwd", return_value=root / "workspace"),
            ):
                model_dirs = GenAIStrategyBase.default_models_dirs()

        self.assertEqual(model_dirs, [Path(strategy_base.__file__).parent / "opl_models"])

    def test_few_shot_retrieval_failure_is_reported_and_skipped(self) -> None:
        base = GenAIStrategyBase(logger=genai_pricing.logger)
        messages: list[str] = []
        with patch.object(strategy_base, "rag_rank", side_effect=RuntimeError("index unavailable")):
            examples = base.gather_few_shots("query", models_dir="models", progress=messages.append)

        self.assertEqual(examples, [])
        self.assertTrue(any("retrieval failed" in message for message in messages))

    def test_few_shot_retrieval_skips_incomplete_hits_and_stops_at_limit(self) -> None:
        base = GenAIStrategyBase(logger=genai_pricing.logger)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            missing_folder = root / "missing"
            missing_folder.mkdir()
            missing_pair = missing_folder / "missing.md"
            missing_pair.write_text("description without files", encoding="utf-8")
            empty_pair = root / "empty.md"
            empty_pair.write_text("empty example", encoding="utf-8")
            (root / "empty.mod").write_text("", encoding="utf-8")
            (root / "empty.dat").write_text("data", encoding="utf-8")
            first = root / "first.md"
            first.write_text("first description", encoding="utf-8")
            (root / "first.mod").write_text("first model", encoding="utf-8")
            (root / "first.dat").write_text("first data", encoding="utf-8")
            second = root / "second.md"
            second.write_text("second description", encoding="utf-8")
            (root / "second.mod").write_text("second model", encoding="utf-8")
            (root / "second.dat").write_text("second data", encoding="utf-8")

            hits = [{"path": str(path)} for path in (missing_pair, empty_pair, first, second)]
            with patch.object(strategy_base, "rag_rank", return_value=hits):
                examples = base.gather_few_shots("query", k=1, models_dir=root)

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["description"], "first description")

    def test_few_shot_retrieval_skips_hits_that_raise_while_reading(self) -> None:
        base = GenAIStrategyBase(logger=genai_pricing.logger)
        with patch.object(strategy_base, "rag_rank", return_value=[{"path": "broken.md"}]):
            with patch.object(base, "safe_read_text", side_effect=OSError("read failed")):
                self.assertEqual(base.gather_few_shots("query", models_dir="models"), [])

    def test_get_grammar_implementation_rejects_invalid_mode(self) -> None:
        self.assertEqual(GenAIStrategyBase.get_grammar_implementation(Grammar.NONE), "")
        with self.assertRaisesRegex(ValueError, "Invalid mode"):
            GenAIStrategyBase.get_grammar_implementation(object())  # type: ignore[arg-type]

    def test_find_pair_in_folder_prefers_matching_stems_then_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            desc = folder / "problem.md"
            desc.write_text("description", encoding="utf-8")
            (folder / "fallback.mod").write_text("model", encoding="utf-8")
            (folder / "fallback.dat").write_text("data", encoding="utf-8")

            mod, dat = GenAIStrategyBase.find_pair_in_folder(desc)

            self.assertEqual(mod, folder / "fallback.mod")
            self.assertEqual(dat, folder / "fallback.dat")

    def test_find_pair_in_folder_prefers_exact_stem_and_handles_missing_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            desc = folder / "problem.md"
            desc.write_text("description", encoding="utf-8")
            (folder / "problem.mod").write_text("matching model", encoding="utf-8")
            (folder / "problem.dat").write_text("matching data", encoding="utf-8")
            (folder / "other.mod").write_text("other model", encoding="utf-8")

            self.assertEqual(
                GenAIStrategyBase.find_pair_in_folder(desc),
                (folder / "problem.mod", folder / "problem.dat"),
            )
            (folder / "problem.mod").unlink()
            (folder / "problem.dat").unlink()
            self.assertEqual(GenAIStrategyBase.find_pair_in_folder(desc), (folder / "other.mod", None))

    def test_usage_and_response_helpers_cover_fallback_shapes(self) -> None:
        usage = Usage()
        usage.add({"prompt_tokens": 4, "completion_tokens": 9})
        response = SimpleNamespace(
            output=[
                SimpleNamespace(text="loose"),
                SimpleNamespace(content=[{"text": " dict"}, SimpleNamespace(text=" object"), object()]),
                SimpleNamespace(content=[object()]),
                SimpleNamespace(content=None),
            ]
        )

        self.assertEqual(usage.as_dict(), {"prompt_tokens": 4, "completion_tokens": 9})
        self.assertEqual(GenAIStrategyBase._coalesce_response_text(response), "loose dict object")

    def test_response_text_coalescing_uses_legacy_fallback(self) -> None:
        class FallbackOutput:
            def __iter__(self):
                raise RuntimeError("not iterable")

            def __getitem__(self, _index):
                return SimpleNamespace(content=[SimpleNamespace(text="fallback")])

        self.assertEqual(GenAIStrategyBase._coalesce_response_text(SimpleNamespace(output_text="direct")), "direct")
        self.assertEqual(GenAIStrategyBase._coalesce_response_text(SimpleNamespace(output=FallbackOutput())), "fallback")
        self.assertEqual(GenAIStrategyBase._coalesce_response_text(SimpleNamespace(output=[])), "")

        class NoTextFallbackOutput(FallbackOutput):
            def __getitem__(self, _index):
                return SimpleNamespace(content=[object()])

        self.assertEqual(GenAIStrategyBase._coalesce_response_text(SimpleNamespace(output=NoTextFallbackOutput())), "")

    def test_base_gather_and_render_few_shots(self) -> None:
        base = GenAIStrategyBase(logger=genai_pricing.logger, few_shot_max_chars=100)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            desc = root / "sample.md"
            mod = root / "sample.mod"
            dat = root / "sample.dat"
            desc.write_text("description", encoding="utf-8")
            mod.write_text("model", encoding="utf-8")
            dat.write_text("data", encoding="utf-8")

            with patch.object(strategy_base, "rag_rank", return_value=[{"path": str(desc), "score": 0.9}]):
                examples = base.gather_few_shots("query", k=2, models_dir=root)

        rendered = base.render_few_shots_section(examples)

        self.assertEqual(examples[0]["description"], "description")
        self.assertIn("few_shot_examples", rendered)
        self.assertIn("model", rendered)

    def test_default_models_dirs_adds_optional_working_directory_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            packaged = root / "package" / "opl_models"
            working = root / "workspace"
            local_models = working / "opl_models"
            packaged.mkdir(parents=True)
            local_models.mkdir(parents=True)

            with (
                patch.object(strategy_base, "files", return_value=root / "package"),
                patch.object(strategy_base.Path, "cwd", return_value=working),
            ):
                models_dirs = GenAIStrategyBase.default_models_dirs()

        self.assertEqual(models_dirs, [packaged, local_models])

    def test_base_default_gather_passes_all_model_roots_to_ranker(self) -> None:
        base = GenAIStrategyBase(logger=genai_pricing.logger)
        roots = [Path("packaged"), Path("workspace") / "opl_models"]

        with (
            patch.object(base, "default_models_dirs", return_value=roots),
            patch.object(strategy_base, "rag_rank", return_value=[]) as rank,
        ):
            self.assertEqual(base.gather_few_shots("query"), [])

        rank.assert_called_once_with(query="query", models_dir=roots, top_k=base.FEW_SHOT_TOP_K)

    def test_base_image_payload_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            image_path = Path(td) / "tiny.png"
            image_path.write_bytes(b"png-bytes")

            data_url = GenAIStrategyBase._image_to_openai_image_url({"path": str(image_path), "mime_type": "image/png"})
            openai_input = GenAIStrategyBase._build_openai_input(
                input_text="describe", images=[{"data_base64": "YWJj", "mime_type": "text/plain"}]
            )

        self.assertTrue(data_url.startswith("data:image/png;base64,"))
        self.assertEqual(openai_input[0]["role"], "user")
        self.assertEqual(openai_input[0]["content"][0], {"type": "input_text", "text": "describe"})
        self.assertEqual(
            GenAIStrategyBase._image_to_openai_image_url({"url": "https://example.test/i.png"}), "https://example.test/i.png"
        )
        self.assertEqual(
            GenAIStrategyBase._image_to_openai_image_url({"data_base64": "data:image/png;base64,abc"}),
            "data:image/png;base64,abc",
        )
        with self.assertRaisesRegex(ValueError, "missing 'path', 'url', or 'data_base64'"):
            GenAIStrategyBase._image_to_openai_image_url({})

    def test_base_gemini_part_helpers_with_fake_types(self) -> None:
        class FakePart:
            @staticmethod
            def from_uri(file_uri, mime_type):
                return ("uri", file_uri, mime_type)

            @staticmethod
            def from_bytes(data, mime_type):
                return ("bytes", data, mime_type)

        fake_types = SimpleNamespace(Part=FakePart)

        self.assertEqual(
            GenAIStrategyBase._image_to_gemini_part(img={"url": "https://example.test/i.png"}, genai_types=fake_types),
            ("uri", "https://example.test/i.png", "image/png"),
        )
        self.assertEqual(
            GenAIStrategyBase._image_to_gemini_part(
                img={"data_base64": "data:text/plain;base64,YWJj"}, genai_types=fake_types
            ),
            ("bytes", b"abc", "text/plain"),
        )
        with patch.object(strategy_base.base64, "b64decode", return_value=b"raw"):
            self.assertEqual(
                GenAIStrategyBase._image_to_gemini_part(img={"data_base64": "data:broken"}, genai_types=fake_types),
                ("bytes", b"raw", "image/png"),
            )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "image.png"
            path.write_bytes(b"image-bytes")
            self.assertEqual(
                GenAIStrategyBase._image_to_gemini_part(img={"path": str(path)}, genai_types=fake_types),
                ("bytes", b"image-bytes", "image/png"),
            )
        with self.assertRaisesRegex(RuntimeError, "Part.from_uri support"):
            GenAIStrategyBase._image_to_gemini_part(
                img={"url": "https://example.test/image.png"},
                genai_types=SimpleNamespace(Part=SimpleNamespace()),
            )
        with self.assertRaisesRegex(ValueError, "missing 'path', 'url', or 'data_base64'"):
            GenAIStrategyBase._image_to_gemini_part(img={}, genai_types=fake_types)

    def test_base_openai_params_retry_and_generation(self) -> None:
        base = GenAIStrategyBase(logger=genai_pricing.logger)
        params = base._build_openai_create_params(
            model_name="gpt-test",
            input_content="prompt",
            max_tokens=5,
            temperature=0.2,
            stop=["END"],
            expected_json=True,
        )

        calls: list[dict[str, Any]] = []

        def create(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError("unsupported parameter: 'response_format'")
            return SimpleNamespace(output_text="ok", usage=SimpleNamespace(input_tokens=3, output_tokens=4))

        client = SimpleNamespace(responses=SimpleNamespace(create=create))
        response = base._call_openai_with_retry(client, params, retries=2, backoff_sec=0)

        self.assertEqual(response.output_text, "ok")
        self.assertIn("response_format", calls[0])
        self.assertNotIn("response_format", calls[1])

        with (
            patch.object(base, "_openai_client", return_value=client),
            patch.object(base, "_call_openai_with_retry", return_value=response),
        ):
            text, usage = base._generate_openai(
                model_name="gpt-test",
                input_text="prompt",
                images=None,
                mt=5,
                temperature=None,
                stop=None,
                progress=None,
                capture_usage=True,
                expected_json=False,
            )

        self.assertEqual(text, "ok")
        self.assertEqual(usage, {"prompt_tokens": 3, "completion_tokens": 4})

    def test_base_gemini_and_ollama_generation_are_mockable(self) -> None:
        base = GenAIStrategyBase(logger=genai_pricing.logger)
        gemini_client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=lambda **kwargs: SimpleNamespace(
                    text="gemini", usage_metadata={"prompt_token_count": 6, "candidates_token_count": 7}
                )
            )
        )

        gemini_text, gemini_usage = base._generate_gemini_newsdk(
            gemini_client,
            model_name="gemini-test",
            input_text="prompt",
            images=None,
            mt=4,
            temperature=0.0,
            progress=None,
            capture_usage=True,
            expected_json=True,
        )
        with patch.object(
            base, "_ollama_generate_text", return_value=("ollama", {"prompt_tokens": 1, "completion_tokens": 2})
        ):
            ollama_text, ollama_usage = base._generate_ollama(
                model_name="llama",
                input_text="prompt",
                images=None,
                mt=3,
                progress=None,
                capture_usage=True,
                expected_json=True,
            )

        self.assertEqual(gemini_text, "gemini")
        self.assertEqual(gemini_usage, {"prompt_tokens": 6, "completion_tokens": 7})
        self.assertEqual(ollama_text, "ollama")
        self.assertEqual(ollama_usage, {"prompt_tokens": 1, "completion_tokens": 2})

    def test_google_client_prefers_new_sdk_and_falls_back_to_legacy(self) -> None:
        google_package = ModuleType("google")
        google_package.__path__ = []
        new_client = object()
        new_sdk = ModuleType("google.genai")
        new_sdk.Client = lambda **kwargs: (kwargs["api_key"], new_client)
        google_package.genai = new_sdk

        with (
            patch.dict(sys.modules, {"google": google_package, "google.genai": new_sdk}),
            patch.dict("os.environ", {"GEMINI_API_KEY": "new-key"}),
        ):
            result = GenAIStrategyBase._google_client()

        self.assertEqual(result.kind, "new")
        self.assertEqual(result.client, ("new-key", new_client))

        legacy_configuration = []
        legacy_sdk = ModuleType("google.generativeai")
        legacy_sdk.configure = lambda **kwargs: legacy_configuration.append(kwargs)
        google_package = ModuleType("google")
        google_package.__path__ = []
        google_package.generativeai = legacy_sdk
        with (
            patch.dict(
                sys.modules,
                {"google": google_package, "google.genai": None, "google.generativeai": legacy_sdk},
            ),
            patch.dict("os.environ", {"GEMINI_API_KEY": "legacy-key"}),
        ):
            result = GenAIStrategyBase._google_client()

        self.assertEqual(result.kind, "legacy")
        self.assertIs(result.client, legacy_sdk)
        self.assertEqual(legacy_configuration, [{"api_key": "legacy-key"}])

        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                GenAIStrategyBase._google_client()

        google_package = ModuleType("google")
        google_package.__path__ = []
        with (
            patch.dict(
                sys.modules,
                {"google": google_package, "google.genai": None, "google.generativeai": None},
            ),
            patch.dict("os.environ", {"GEMINI_API_KEY": "missing-sdk-key"}),
        ):
            with self.assertRaisesRegex(RuntimeError, "google-genai is not installed"):
                GenAIStrategyBase._google_client()

    def test_ollama_generation_shapes_and_response_error(self) -> None:
        calls: list[dict[str, Any]] = []
        fake_ollama = ModuleType("ollama")

        def fake_generate(**kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            return {"response": "answer", "prompt_eval_count": 3, "eval_count": 5}

        fake_ollama.generate = fake_generate

        with patch.dict(sys.modules, {"ollama": fake_ollama}):
            result = GenAIStrategyBase._ollama_generate_text(
                model_name="llama",
                prompt="prompt",
                num_predict=20,
                return_usage=True,
                enforce_json=True,
            )

        self.assertEqual(result, ("answer", {"prompt_tokens": 3, "completion_tokens": 5}))
        self.assertEqual(calls, [{"model": "llama", "prompt": "prompt", "options": {"num_predict": 20, "format": "json"}}])

        fake_ollama.generate = lambda **_kwargs: {"response": "plain answer"}
        with patch.dict(sys.modules, {"ollama": fake_ollama}):
            self.assertEqual(
                GenAIStrategyBase._ollama_generate_text(
                    model_name="llama",
                    prompt="prompt",
                    num_predict=None,
                    return_usage=False,
                    enforce_json=False,
                ),
                "plain answer",
            )

        class BadResponse:
            def get(self, _key, _default=None):
                raise KeyError("response")

        fake_ollama.generate = lambda **_kwargs: BadResponse()
        with patch.dict(sys.modules, {"ollama": fake_ollama}):
            with self.assertRaisesRegex(RuntimeError, "Failed to retrieve response text"):
                GenAIStrategyBase._ollama_generate_text(
                    model_name="llama",
                    prompt="prompt",
                    num_predict=None,
                    return_usage=False,
                    enforce_json=False,
                )

        with patch.dict(sys.modules, {"ollama": None}):
            with self.assertRaisesRegex(RuntimeError, "ollama package is not installed"):
                GenAIStrategyBase._ollama_generate_text(
                    model_name="llama",
                    prompt="prompt",
                    num_predict=None,
                    return_usage=False,
                    enforce_json=False,
                )

    def test_ollama_generation_without_usage_rejects_images(self) -> None:
        base = GenAIStrategyBase(logger=genai_pricing.logger)
        with patch.object(base, "_ollama_generate_text", return_value="answer") as generate:
            self.assertEqual(
                base._generate_ollama(
                    model_name="llama",
                    input_text="prompt",
                    images=None,
                    mt=7,
                    progress=None,
                    capture_usage=False,
                    expected_json=False,
                ),
                ("answer", None),
            )
        self.assertFalse(generate.call_args.kwargs["return_usage"])

        with self.assertRaisesRegex(RuntimeError, "image prompts are not supported"):
            base._generate_ollama(
                model_name="llama",
                input_text="prompt",
                images=[{"path": "image.png"}],
                mt=7,
                progress=None,
                capture_usage=False,
                expected_json=False,
            )

    def test_gemini_legacy_generation_handles_images_and_candidate_text(self) -> None:
        from PIL import Image

        image_bytes = b"image-bytes"
        image_data = base64.b64encode(image_bytes).decode("ascii")
        with tempfile.TemporaryDirectory() as td:
            image_path = Path(td) / "image.png"
            image_path.write_bytes(image_bytes)
            captured = []
            response = SimpleNamespace(
                text=None,
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[SimpleNamespace(text="legacy "), SimpleNamespace(text="answer"), object()]
                        )
                    ),
                    SimpleNamespace(content=None),
                ],
                usage_metadata={"prompt_token_count": 2, "candidates_token_count": 4},
            )

            def generate_content(parts, generation_config):
                captured.append((parts, generation_config))
                return response

            legacy_client = SimpleNamespace(GenerativeModel=lambda _name: SimpleNamespace(generate_content=generate_content))
            with patch.object(Image, "open", side_effect=lambda source: ("opened", source)):
                result = GenAIStrategyBase(logger=genai_pricing.logger)._generate_gemini_legacy(
                    legacy_client,
                    model_name="gemini-test",
                    input_text="prompt",
                    images=[
                        {"data_base64": f"data:image/png;base64,{image_data}"},
                        {"data_base64": image_data},
                        {"path": str(image_path)},
                    ],
                    mt=12,
                    temperature=0.1,
                    progress=None,
                    capture_usage=True,
                    expected_json=True,
                )

        self.assertEqual(result, ("legacy answer", {"prompt_tokens": 2, "completion_tokens": 4}))
        self.assertEqual(len(captured[0][0]), 4)
        self.assertEqual(
            captured[0][1],
            {
                "max_output_tokens": 12,
                "temperature": 0.1,
                "response_mime_type": "application/json",
            },
        )

        with self.assertRaisesRegex(RuntimeError, "does not support URL images"):
            GenAIStrategyBase(logger=genai_pricing.logger)._generate_gemini_legacy(
                legacy_client,
                model_name="gemini-test",
                input_text="prompt",
                images=[{"url": "https://example.test/image.png"}],
                mt=None,
                temperature=None,
                progress=None,
                capture_usage=False,
                expected_json=False,
            )

        with self.assertRaisesRegex(ValueError, "missing 'path' or 'data_base64'"):
            GenAIStrategyBase(logger=genai_pricing.logger)._generate_gemini_legacy(
                legacy_client,
                model_name="gemini-test",
                input_text="prompt",
                images=[{}],
                mt=None,
                temperature=None,
                progress=None,
                capture_usage=False,
                expected_json=False,
            )

        with (
            patch.object(strategy_base.base64, "b64decode", return_value=b"malformed"),
            patch.object(Image, "open", return_value="decoded image"),
        ):
            self.assertEqual(
                GenAIStrategyBase(logger=genai_pricing.logger)._generate_gemini_legacy(
                    legacy_client,
                    model_name="gemini-test",
                    input_text="prompt",
                    images=[{"data_base64": "data:broken"}],
                    mt=None,
                    temperature=None,
                    progress=None,
                    capture_usage=False,
                    expected_json=False,
                ),
                ("legacy answer", None),
            )

        with (
            patch.dict(sys.modules, {"PIL": None}),
            self.assertRaisesRegex(RuntimeError, "image prompts require Pillow"),
        ):
            GenAIStrategyBase(logger=genai_pricing.logger)._generate_gemini_legacy(
                legacy_client,
                model_name="gemini-test",
                input_text="prompt",
                images=[{"path": "image.png"}],
                mt=None,
                temperature=None,
                progress=None,
                capture_usage=False,
                expected_json=False,
            )

        direct_response = SimpleNamespace(text="direct response")
        direct_client = SimpleNamespace(
            GenerativeModel=lambda _name: SimpleNamespace(generate_content=lambda *_args, **_kwargs: direct_response)
        )
        self.assertEqual(
            GenAIStrategyBase(logger=genai_pricing.logger)._generate_gemini_legacy(
                direct_client,
                model_name="gemini-test",
                input_text="prompt",
                images=None,
                mt=None,
                temperature=None,
                progress=None,
                capture_usage=False,
                expected_json=False,
            ),
            ("direct response", None),
        )

    def test_gemini_new_sdk_falls_back_to_inline_image_parts(self) -> None:
        fake_types = SimpleNamespace(
            Part=SimpleNamespace(
                from_text=lambda **kwargs: ("text", kwargs),
                from_bytes=lambda **kwargs: ("bytes", kwargs),
            ),
            Content=lambda **kwargs: kwargs,
            GenerateContentConfig=lambda **kwargs: kwargs,
        )
        google_package = ModuleType("google")
        google_package.__path__ = []
        genai_module = ModuleType("google.genai")
        genai_module.types = fake_types
        google_package.genai = genai_module
        calls = []
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(parts=[SimpleNamespace(text="candidate text"), SimpleNamespace(text=1), object()])
                ),
                SimpleNamespace(content=None),
            ],
            usage_metadata={"prompt_token_count": 5, "candidates_token_count": 2},
        )

        def generate_content(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError("force dictionary-config fallback")
            return response

        client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
        with patch.dict(sys.modules, {"google": google_package, "google.genai": genai_module}):
            result = GenAIStrategyBase(logger=genai_pricing.logger)._generate_gemini_newsdk(
                client,
                model_name="gemini-test",
                input_text="prompt",
                images=[{"data_base64": "YWJj", "mime_type": "image/png"}],
                mt=8,
                temperature=0.2,
                progress=None,
                capture_usage=True,
                expected_json=True,
            )
            base64_contents = calls[1]["contents"]

            with self.assertRaisesRegex(RuntimeError, "image URLs require google.genai typed parts"):
                GenAIStrategyBase(logger=genai_pricing.logger)._generate_gemini_newsdk(
                    client,
                    model_name="gemini-test",
                    input_text="prompt",
                    images=[{"url": "https://example.test/image.png"}],
                    mt=None,
                    temperature=None,
                    progress=None,
                    capture_usage=False,
                    expected_json=False,
                )

            with self.assertRaisesRegex(ValueError, "missing 'path' or 'data_base64'"):
                GenAIStrategyBase(logger=genai_pricing.logger)._generate_gemini_newsdk(
                    client,
                    model_name="gemini-test",
                    input_text="prompt",
                    images=[{}],
                    mt=None,
                    temperature=None,
                    progress=None,
                    capture_usage=False,
                    expected_json=False,
                )

            calls.clear()
            with patch.object(Path, "read_bytes", return_value=b"path-bytes"):
                path_result = GenAIStrategyBase(logger=genai_pricing.logger)._generate_gemini_newsdk(
                    client,
                    model_name="gemini-test",
                    input_text="prompt",
                    images=[{"path": "image.png"}],
                    mt=None,
                    temperature=None,
                    progress=None,
                    capture_usage=False,
                    expected_json=False,
                )
            path_contents = calls[1]["contents"]

            calls.clear()
            no_image_result = GenAIStrategyBase(logger=genai_pricing.logger)._generate_gemini_newsdk(
                client,
                model_name="gemini-test",
                input_text="plain prompt",
                images=None,
                mt=None,
                temperature=None,
                progress=None,
                capture_usage=False,
                expected_json=False,
            )

        self.assertEqual(result, ("candidate text", {"prompt_tokens": 5, "completion_tokens": 2}))
        self.assertEqual(
            base64_contents,
            [{"role": "user", "parts": [{"text": "prompt"}, {"inline_data": {"mime_type": "image/png", "data": "YWJj"}}]}],
        )
        self.assertEqual(path_result, ("candidate text", None))
        self.assertEqual(
            path_contents,
            [
                {
                    "role": "user",
                    "parts": [
                        {"text": "prompt"},
                        {"inline_data": {"mime_type": "image/png", "data": "cGF0aC1ieXRlcw=="}},
                    ],
                }
            ],
        )
        self.assertEqual(no_image_result, ("candidate text", None))
        self.assertEqual(calls[1]["contents"], "plain prompt")

        class BrokenCandidates:
            text = None

            @property
            def candidates(self):
                raise RuntimeError("unreadable candidates")

        broken_client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **_kwargs: BrokenCandidates()))
        with patch.dict(sys.modules, {"google": google_package, "google.genai": genai_module}):
            self.assertEqual(
                GenAIStrategyBase(logger=genai_pricing.logger)._generate_gemini_newsdk(
                    broken_client,
                    model_name="gemini-test",
                    input_text="prompt",
                    images=None,
                    mt=None,
                    temperature=None,
                    progress=None,
                    capture_usage=False,
                    expected_json=False,
                ),
                ("", None),
            )

    def test_llm_dispatch_routes_openai_and_google_providers(self) -> None:
        base = GenAIStrategyBase(logger=genai_pricing.logger, max_output_tokens=19)
        usage = {"prompt_tokens": 2, "completion_tokens": 4}

        with patch.object(base, "_generate_openai", return_value=("openai", usage)) as openai:
            self.assertEqual(
                base.llm_generate_text(provider=LLMProvider.OPENAI, model_name="gpt-test", input_text="prompt"),
                "openai",
            )
        self.assertEqual(openai.call_args.kwargs["mt"], 19)
        with patch.object(base, "_generate_openai", return_value=("openai", usage)):
            self.assertEqual(
                base.llm_generate_text(
                    provider=LLMProvider.OPENAI,
                    model_name="gpt-test",
                    input_text="prompt",
                    capture_usage=True,
                ),
                ("openai", usage),
            )

        with (
            patch.object(base, "_google_client", return_value=GoogleClient(kind="new", client="new-client")),
            patch.object(base, "_generate_gemini_newsdk", return_value=("new gemini", usage)) as generate_new,
        ):
            self.assertEqual(
                base.llm_generate_text(
                    provider=LLMProvider.GOOGLE,
                    model_name="gemini-test",
                    input_text="prompt",
                    capture_usage=True,
                ),
                ("new gemini", usage),
            )
        self.assertEqual(generate_new.call_args.args[0], "new-client")

        with (
            patch.object(base, "_google_client", return_value=GoogleClient(kind="legacy", client="legacy-client")),
            patch.object(base, "_generate_gemini_legacy", return_value=("legacy gemini", None)) as generate_legacy,
        ):
            self.assertEqual(
                base.llm_generate_text(provider=LLMProvider.GOOGLE, model_name="gemini-test", input_text="prompt"),
                "legacy gemini",
            )
        self.assertEqual(generate_legacy.call_args.args[0], "legacy-client")

        with patch.object(base, "_generate_ollama", return_value=("ollama", None)):
            self.assertEqual(
                base.llm_generate_text(
                    provider=LLMProvider.OLLAMA,
                    model_name="llama",
                    input_text="prompt",
                ),
                "ollama",
            )

        with self.assertRaisesRegex(ValueError, "Unsupported LLM provider"):
            base.llm_generate_text(provider=cast(LLMProvider, "invalid"), model_name="test", input_text="prompt")

    def test_openai_retry_reports_exhaustion_without_sleeping(self) -> None:
        base = GenAIStrategyBase(logger=genai_pricing.logger)
        create = MagicMock(side_effect=RuntimeError("temporary failure"))
        client = SimpleNamespace(responses=SimpleNamespace(create=create))
        progress_messages: list[str] = []

        with patch.object(strategy_base, "sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "OpenAI request failed after 2 attempts: temporary failure"):
                base._call_openai_with_retry(
                    client,
                    {"model": "gpt-test", "input": "prompt"},
                    retries=2,
                    backoff_sec=0.5,
                    progress=progress_messages.append,
                )

        self.assertEqual(create.call_count, 2)
        sleep.assert_any_call(0.5)
        sleep.assert_any_call(1.0)
        self.assertTrue(any("failed after 2 attempts" in message for message in progress_messages))

    def test_openai_retry_prunes_signature_and_unknown_parameters(self) -> None:
        base = GenAIStrategyBase(logger=genai_pricing.logger)
        accepted = []

        def restricted_create(model, input):
            accepted.append((model, input))
            return "accepted"

        client = SimpleNamespace(responses=SimpleNamespace(create=restricted_create))
        self.assertEqual(
            base._call_openai_with_retry(
                client,
                {"model": "gpt-test", "input": "prompt", "temperature": 0.2},
            ),
            "accepted",
        )
        self.assertEqual(accepted, [("gpt-test", "prompt")])

        calls = []

        def unsupported_create(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError("unsupported parameter: 'max_output_tokens'")
            return "retried"

        client = SimpleNamespace(responses=SimpleNamespace(create=unsupported_create))
        self.assertEqual(
            base._call_openai_with_retry(
                client,
                {"model": "gpt-test", "input": "prompt", "max_output_tokens": 10},
                retries=2,
                backoff_sec=0,
            ),
            "retried",
        )
        self.assertIn("max_output_tokens", calls[0])
        self.assertNotIn("max_output_tokens", calls[1])

        def retry_with_unrecognized_message(**kwargs):
            calls.append(kwargs)
            if len(calls) == 3:
                raise RuntimeError("parameter 'max_output_tokens' is not supported")
            return "supported after retry"

        client = SimpleNamespace(responses=SimpleNamespace(create=retry_with_unrecognized_message))
        self.assertEqual(
            base._call_openai_with_retry(
                client,
                {"model": "gpt-test", "input": "prompt", "max_output_tokens": 10},
                retries=2,
                backoff_sec=0,
            ),
            "supported after retry",
        )
        self.assertNotIn("max_output_tokens", calls[3])

        opaque_client = SimpleNamespace(responses=SimpleNamespace(create=lambda **_kwargs: "opaque"))
        with patch.object(strategy_base.inspect, "signature", side_effect=ValueError("signature unavailable")):
            self.assertEqual(
                base._call_openai_with_retry(opaque_client, {"model": "gpt-test", "input": "prompt"}),
                "opaque",
            )

    def test_openai_generation_rejects_empty_response(self) -> None:
        base = GenAIStrategyBase(logger=genai_pricing.logger)
        with (
            patch.object(base, "_call_openai_with_retry", return_value=SimpleNamespace(output_text="")),
            self.assertRaisesRegex(RuntimeError, "Empty OpenAI response"),
        ):
            base._generate_openai(
                client=object(),
                model_name="gpt-test",
                input_text="prompt",
                images=None,
                mt=None,
                temperature=None,
                stop=None,
                progress=None,
                capture_usage=False,
                expected_json=False,
            )

    def test_base_dispatch_compile_write_and_cost_helpers(self) -> None:
        base = GenAIStrategyBase(logger=genai_pricing.logger)

        with patch.object(base, "_generate_ollama", return_value=("text", {"prompt_tokens": 8, "completion_tokens": 9})):
            generated = base.llm_generate_text(
                provider=LLMProvider.OLLAMA,
                model_name="llama",
                input_text="prompt",
                capture_usage=True,
            )
        with tempfile.TemporaryDirectory() as td:
            model_file = str(Path(td) / "nested" / "model.mod")
            data_file = str(Path(td) / "nested" / "data.dat")
            base.write_model_data_files(model_file, data_file, "model", "data")

            self.assertEqual(Path(model_file).read_text(encoding="utf-8"), "model")
            self.assertEqual(Path(data_file).read_text(encoding="utf-8"), "data")

        with (
            patch.object(strategy_base.OPLCompiler, "compile_model", return_value=None),
            patch.object(strategy_base, "_estimate_costs", return_value={"total_cost": 1.2}),
        ):
            errors = base.compile_model_data("model", "data")
            usage = Usage(prompt_tokens=2, completion_tokens=3)
            cost = base.estimate_cost("gpt-test", usage)

        self.assertEqual(generated, ("text", {"prompt_tokens": 8, "completion_tokens": 9}))
        self.assertEqual(errors, [])
        self.assertEqual(cost["estimated_costs"], {"total_cost": 1.2})
        self.assertEqual(GenAIStrategyBase.infer_provider(None, "gemini-2"), LLMProvider.GOOGLE)
        self.assertEqual(GenAIStrategyBase.infer_provider(None, "llama3"), LLMProvider.OLLAMA)
        self.assertEqual(GenAIStrategyBase.infer_provider("elm", "gpt-test"), LLMProvider.ELM)
        self.assertEqual(GenAIStrategyBase.infer_provider(" OAI ", "model"), LLMProvider.OPENAI)
        self.assertEqual(GenAIStrategyBase.infer_provider("ollama", "model"), LLMProvider.OLLAMA)
        self.assertEqual(GenAIStrategyBase.infer_provider(None, "gpt-oss-20b"), LLMProvider.OLLAMA)
        self.assertEqual(GenAIStrategyBase.infer_provider("unsupported", "model"), LLMProvider.OPENAI)

    def test_compile_and_cost_helpers_return_errors_without_raising(self) -> None:
        with patch.object(
            strategy_base.OPLCompiler,
            "compile_model",
            side_effect=strategy_base.SemanticError("semantic problem"),
        ):
            self.assertEqual(
                GenAIStrategyBase.compile_model_data("model", "data"),
                [str(strategy_base.SemanticError("semantic problem"))],
            )
        with patch.object(strategy_base.OPLCompiler, "compile_model", side_effect=ValueError("invalid input")):
            self.assertEqual(
                GenAIStrategyBase.compile_model_data("model", "data"),
                ["ValueError: invalid input"],
            )

        usage = Usage(prompt_tokens=1)
        with patch.object(strategy_base, "_estimate_costs", side_effect=RuntimeError("pricing unavailable")):
            result = GenAIStrategyBase.estimate_cost("model", usage)
        self.assertEqual(result["estimated_costs"], {})

        with (
            patch.object(strategy_base.os.path, "dirname", return_value=""),
            patch("builtins.open", mock_open()) as open_file,
        ):
            GenAIStrategyBase.write_model_data_files("model.mod", "data.dat", "model", "data")
        self.assertEqual(open_file.call_count, 2)

    def test_base_elm_dispatch_uses_elm_client(self) -> None:
        base = GenAIStrategyBase(logger=genai_pricing.logger)
        elm_client = object()

        with (
            patch.object(base, "_ELM_client", return_value=elm_client),
            patch.object(base, "_generate_openai", return_value=("elm", None)) as generate,
        ):
            result = base.llm_generate_text(
                provider=LLMProvider.ELM,
                model_name="gpt-test",
                input_text="prompt",
            )

        self.assertEqual(result, "elm")
        self.assertIs(generate.call_args.kwargs["client"], elm_client)
        self.assertEqual(generate.call_args.kwargs["provider_name"], "ELM")

    def test_base_elm_client_uses_configured_endpoint_and_api_key(self) -> None:
        created_with: dict[str, str] = {}

        def openai_client(**kwargs):
            created_with.update(kwargs)
            return "elm-client"

        with (
            patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=openai_client)}),
            patch.dict("os.environ", {"ELM_API_KEY": "elm-key"}, clear=True),
        ):
            client = GenAIStrategyBase._ELM_client()

        self.assertEqual(client, "elm-client")
        self.assertEqual(created_with, {"base_url": "https://elm.edina.ac.uk/api/v1", "api_key": "elm-key"})

        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "ELM_API_KEY environment variable not set"):
                GenAIStrategyBase._ELM_client()
        with (
            patch.dict(sys.modules, {"openai": None}),
            patch.dict("os.environ", {"ELM_API_KEY": "elm-key"}),
        ):
            with self.assertRaisesRegex(RuntimeError, "openai is not installed"):
                GenAIStrategyBase._ELM_client()

    def test_base_elm_dispatch_returns_captured_usage(self) -> None:
        base = GenAIStrategyBase(logger=genai_pricing.logger)
        expected = ("elm", {"prompt_tokens": 2, "completion_tokens": 3})

        with (
            patch.object(base, "_ELM_client", return_value=object()),
            patch.object(base, "_generate_openai", return_value=expected),
        ):
            result = base.llm_generate_text(
                provider=LLMProvider.ELM,
                model_name="gpt-test",
                input_text="prompt",
                capture_usage=True,
            )

        self.assertEqual(result, expected)

    def test_base_elm_generation_uses_elm_progress_labels(self) -> None:
        base = GenAIStrategyBase(logger=genai_pricing.logger)
        progress_messages: list[str] = []
        response = SimpleNamespace(output_text="ok")
        client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: response))

        text, usage = base._generate_openai(
            client=client,
            provider_name="ELM",
            model_name="gpt-test",
            input_text="prompt",
            images=None,
            mt=None,
            temperature=None,
            stop=None,
            progress=progress_messages.append,
            capture_usage=False,
            expected_json=False,
        )

        self.assertEqual((text, usage), ("ok", None))
        self.assertTrue(progress_messages)
        self.assertTrue(all("OpenAI" not in message for message in progress_messages))
        self.assertTrue(all("ELM" in message for message in progress_messages))


class TestGenAIPricing(unittest.TestCase):
    def setUp(self) -> None:
        genai_pricing.clear_pricing_cache()

    def tearDown(self) -> None:
        genai_pricing.clear_pricing_cache()

    def test_parse_pricing_reads_markdown_table_and_inline_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pricing_path = Path(td) / "pricing.md"
            pricing_path.write_text(
                "\n".join(
                    [
                        "| Model | Prompt / 1K | Completion / 1M |",
                        "| --- | --- | --- |",
                        "| gpt-test | $0.01 | $2.50 |",
                        "inline-model: prompt $0.20 / 1M, completion $0.60 / 1M",
                    ]
                ),
                encoding="utf-8",
            )

            rates = genai_pricing._parse_pricing(str(pricing_path))

        self.assertEqual(rates["gpt-test"], {"prompt_per_1M": 10.0, "completion_per_1M": 2.5})
        self.assertEqual(rates["inline-model"], {"prompt_per_1M": 0.2, "completion_per_1M": 0.6})

    def test_parse_pricing_reads_litellm_json_rates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pricing_path = Path(td) / "model_prices_and_context_window_backup.json"
            pricing_path.write_text(
                '{"gpt-json": {"input_cost_per_token": 0.000002, "output_cost_per_token": 0.000008}}',
                encoding="utf-8",
            )

            rates = genai_pricing._parse_pricing(str(pricing_path))

        self.assertEqual(rates["gpt-json"], {"prompt_per_1M": 2.0, "completion_per_1M": 8.0})

    def test_parse_pricing_uses_extension_instead_of_json_markdown_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pricing_path = Path(td) / "model_prices_and_context_window_backup.json"
            pricing_path.write_text("| gpt-test | $0.01 | $2.50 |", encoding="utf-8")

            rates = genai_pricing._parse_pricing(str(pricing_path))

        self.assertEqual(rates, {})

    def test_estimate_costs_reads_current_litellm_backup_models(self) -> None:
        usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}

        codex = genai_pricing.estimate_costs(SimpleNamespace(model="gpt-5.3-codex"), usage)
        gpt55 = genai_pricing.estimate_costs(SimpleNamespace(model="gpt-5.5"), usage)

        self.assertEqual(codex, {"prompt_cost": 1.75, "completion_cost": 14.0, "total_cost": 15.75})
        self.assertEqual(gpt55, {"prompt_cost": 5.0, "completion_cost": 30.0, "total_cost": 35.0})

    def test_parse_pricing_rejects_non_http_remote_scheme(self) -> None:
        with patch.object(genai_pricing.urllib.request, "urlopen") as urlopen_mock:
            rates = genai_pricing._parse_pricing("file://tmp/pricing.md")

        self.assertEqual(rates, {})
        urlopen_mock.assert_not_called()

    def test_extract_usage_from_openai_and_gemini_shapes(self) -> None:
        openai_resp = SimpleNamespace(usage=SimpleNamespace(input_tokens=12, output_tokens=5))
        gemini_resp = {"usage_metadata": {"prompt_token_count": 20, "candidates_token_count": 7}}

        self.assertEqual(
            genai_pricing._extract_openai_usage(openai_resp, "input", "output", "gpt-test"),
            {"prompt_tokens": 12, "completion_tokens": 5},
        )
        self.assertEqual(
            genai_pricing._extract_gemini_usage(gemini_resp, "input", "output"), {"prompt_tokens": 20, "completion_tokens": 7}
        )

    def test_estimate_costs_uses_exact_and_substring_model_matches(self) -> None:
        rates = {"gpt-test": {"prompt_per_1M": 2.0, "completion_per_1M": 8.0}}
        usage = {"prompt_tokens": 500_000, "completion_tokens": 250_000}

        with patch.object(genai_pricing, "_parse_pricing", return_value=rates):
            exact = genai_pricing.estimate_costs(SimpleNamespace(model="gpt-test"), usage)
            dated = genai_pricing.estimate_costs(SimpleNamespace(model="gpt-test-2026"), usage)

        expected = {"prompt_cost": 1.0, "completion_cost": 2.0, "total_cost": 3.0}
        self.assertEqual(exact, expected)
        self.assertEqual(dated, expected)

    def test_pricing_fallbacks_and_missing_model_costs(self) -> None:
        self.assertEqual(genai_pricing._approx_token_count("12345"), 2)
        self.assertEqual(genai_pricing._usage_dict(None, 7), {"prompt_tokens": 0, "completion_tokens": 7})

        with patch.object(genai_pricing, "_count_openai_tokens", side_effect=[3, 4]):
            usage = genai_pricing._extract_openai_usage({"usage": {}}, "abc", "def", "missing-model")
        with patch.object(genai_pricing, "_parse_pricing", return_value={}):
            costs = genai_pricing.estimate_costs(SimpleNamespace(model="missing-model"), {"prompt_tokens": 10})

        self.assertEqual(usage, {"prompt_tokens": 3, "completion_tokens": 4})
        self.assertEqual(costs, {"total_cost": 0.0})


class TestModelDiscovery(unittest.TestCase):
    def test_openai_model_listing_filters_prefix_and_deduplicates(self) -> None:
        client = SimpleNamespace(
            models=SimpleNamespace(
                list=lambda: SimpleNamespace(data=[{"id": "gpt-b"}, {"id": "other"}, {"id": "gpt-a"}, {"id": "gpt-a"}])
            )
        )

        with patch.object(GenAIStrategyBase, "_openai_client", return_value=client):
            self.assertEqual(model_discovery.list_openai_models(prefix="gpt"), ["gpt-a", "gpt-b"])

    def test_elm_model_listing_uses_elm_client(self) -> None:
        client = SimpleNamespace(models=SimpleNamespace(list=lambda: SimpleNamespace(data=[{"id": "local"}, {"id": "gpt"}])))

        with patch.object(GenAIStrategyBase, "_ELM_client", return_value=client):
            self.assertEqual(model_discovery.list_elm_models(), ["gpt", "local"])

    def test_gemini_new_sdk_listing_strips_models_prefix(self) -> None:
        google_client = SimpleNamespace(
            models=SimpleNamespace(
                list=lambda: [SimpleNamespace(name="models/gemini-2"), {"name": "models/other"}, {"name": "gemini-1"}]
            )
        )

        with patch.object(GenAIStrategyBase, "_google_client", return_value=GoogleClient(kind="new", client=google_client)):
            self.assertEqual(model_discovery.list_gemini_models(prefix="gemini"), ["gemini-1", "gemini-2"])

    def test_ollama_model_listing_accepts_dict_and_object_shapes(self) -> None:
        fake_ollama = SimpleNamespace(
            list=lambda: {"models": [{"model": "llama3.1"}, SimpleNamespace(name="mistral"), {"name": "llama3.2"}]}
        )

        with patch.dict(sys.modules, {"ollama": fake_ollama}):
            self.assertEqual(model_discovery.list_ollama_models(prefix="llama"), ["llama3.1", "llama3.2"])

    def test_list_models_dispatches_from_inferred_provider(self) -> None:
        with (
            patch.object(GenAIStrategyBase, "infer_provider", return_value=LLMProvider.OLLAMA),
            patch.object(strategy_base, "list_ollama_models", return_value=["local"]),
        ):
            self.assertEqual(model_discovery.list_models(model_name="anything"), ["local"])

    def test_list_models_dispatches_to_elm_and_labels_errors(self) -> None:
        with patch.object(strategy_base, "list_elm_models", return_value=["elm-model"]) as list_elm:
            self.assertEqual(strategy_base.list_models(llm_provider="elm", model_name="gpt-test"), ["elm-model"])
        list_elm.assert_called_once_with()

        failing_client = SimpleNamespace(models=SimpleNamespace(list=lambda: (_ for _ in ()).throw(RuntimeError("down"))))
        with patch.object(GenAIStrategyBase, "_ELM_client", return_value=failing_client):
            with self.assertRaisesRegex(RuntimeError, "Failed to list ELM models: down"):
                strategy_base.list_elm_models()

    def test_model_listing_error_paths_are_wrapped(self) -> None:
        with patch.object(
            GenAIStrategyBase,
            "_openai_client",
            return_value=SimpleNamespace(models=SimpleNamespace(list=lambda: (_ for _ in ()).throw(RuntimeError("down")))),
        ):
            with self.assertRaisesRegex(RuntimeError, "Failed to list OpenAI models"):
                model_discovery.list_openai_models()

        with patch.dict(sys.modules, {"ollama": None}):
            with self.assertRaisesRegex(RuntimeError, "ollama package is not installed"):
                model_discovery.list_ollama_models()

    def test_shared_openai_listing_accepts_iterable_sdk_responses(self) -> None:
        client = SimpleNamespace(
            models=SimpleNamespace(
                list=lambda: [
                    SimpleNamespace(id="gpt-z"),
                    {"id": "gpt-a"},
                    {"id": "other"},
                    {"name": "ignored"},
                    SimpleNamespace(id=5),
                ]
            )
        )

        self.assertEqual(
            strategy_base._list_openai_compatible_models(client, prefix=None, provider_name="OpenAI"),
            ["gpt-a", "gpt-z", "other"],
        )
        with patch.object(GenAIStrategyBase, "_openai_client", return_value=client):
            self.assertEqual(strategy_base.list_openai_models(), ["gpt-a", "gpt-z"])
        data_client = SimpleNamespace(
            models=SimpleNamespace(list=lambda: SimpleNamespace(data=[{"id": "gpt-c"}, SimpleNamespace(id="gpt-b")]))
        )
        self.assertEqual(
            strategy_base._list_openai_compatible_models(data_client, prefix="gpt", provider_name="OpenAI"),
            ["gpt-b", "gpt-c"],
        )

    def test_openai_client_uses_configured_api_key(self) -> None:
        created_with: list[dict[str, Any]] = []
        fake_openai = ModuleType("openai")

        def fake_openai_constructor(**kwargs: Any) -> str:
            created_with.append(kwargs)
            return "openai-client"

        fake_openai.OpenAI = fake_openai_constructor

        with (
            patch.dict(sys.modules, {"openai": fake_openai}),
            patch.dict("os.environ", {"OPENAI_API_KEY": "openai-key"}),
        ):
            self.assertEqual(GenAIStrategyBase._openai_client(), "openai-client")
        self.assertEqual(created_with, [{"api_key": "openai-key"}])

        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                GenAIStrategyBase._openai_client()

        with (
            patch.dict(sys.modules, {"openai": None}),
            patch.dict("os.environ", {"OPENAI_API_KEY": "openai-key"}),
        ):
            with self.assertRaisesRegex(RuntimeError, "openai is not installed"):
                GenAIStrategyBase._openai_client()

    def test_gemini_listing_handles_new_and_legacy_sdk_results(self) -> None:
        new_client = SimpleNamespace(
            models=SimpleNamespace(
                list=lambda: [
                    SimpleNamespace(name="models/gemini-z"),
                    {"name": "models/gemini-a"},
                    {"name": "other"},
                    {"name": None},
                    {"name": "models/gemini-a"},
                ]
            )
        )
        self.assertEqual(
            strategy_base._list_gemini_models_newsdk(new_client, prefix="gemini"),
            ["gemini-a", "gemini-z"],
        )

        legacy_client = SimpleNamespace(
            list_models=lambda: [
                SimpleNamespace(name="models/gemini-z", supported_generation_methods=["generateContent"]),
                SimpleNamespace(name="models/text-only", supported_generation_methods=["embedContent"]),
                SimpleNamespace(name="models/gemini-a", supported_generation_methods=["generateContent"]),
                SimpleNamespace(name="gemini-unprefixed", supported_generation_methods=["generateContent"]),
            ]
        )
        self.assertEqual(
            strategy_base._list_gemini_models_legacy(legacy_client, prefix="gemini"),
            ["gemini-a", "gemini-z"],
        )
        self.assertEqual(strategy_base._list_gemini_models_legacy(legacy_client, prefix=None), [])

        with patch.object(
            GenAIStrategyBase,
            "_google_client",
            return_value=GoogleClient(kind="legacy", client=legacy_client),
        ):
            self.assertEqual(strategy_base.list_gemini_models(), ["gemini-a", "gemini-z"])

        with patch.object(
            GenAIStrategyBase,
            "_google_client",
            return_value=GoogleClient(kind="new", client=new_client),
        ):
            self.assertEqual(strategy_base.list_gemini_models(prefix=None), ["gemini-a", "gemini-z", "other"])

        with self.assertRaisesRegex(RuntimeError, "Failed to list Gemini models"):
            strategy_base._list_gemini_models_legacy(
                SimpleNamespace(list_models=lambda: (_ for _ in ()).throw(RuntimeError("offline"))),
                prefix="gemini",
            )
        with self.assertRaisesRegex(RuntimeError, "Failed to list Gemini models \\(google.genai\\)"):
            strategy_base._list_gemini_models_newsdk(
                SimpleNamespace(models=SimpleNamespace(list=lambda: (_ for _ in ()).throw(RuntimeError("offline")))),
                prefix=None,
            )

    def test_ollama_listing_shapes_errors_and_provider_dispatch(self) -> None:
        fake_ollama = ModuleType("ollama")
        fake_ollama.list = lambda: {
            "models": [
                {"model": "llama-z"},
                {"name": "llama-a"},
                SimpleNamespace(model="mistral"),
                {"model": 4},
            ]
        }
        with patch.dict(sys.modules, {"ollama": fake_ollama}):
            self.assertEqual(strategy_base.list_ollama_models(prefix="llama"), ["llama-a", "llama-z"])
            self.assertEqual(strategy_base.list_ollama_models(), ["llama-a", "llama-z", "mistral"])

        fake_ollama.list = lambda: (_ for _ in ()).throw(RuntimeError("offline"))
        with patch.dict(sys.modules, {"ollama": fake_ollama}):
            with self.assertRaisesRegex(RuntimeError, "Failed to list Ollama models: offline"):
                strategy_base.list_ollama_models()

        with patch.dict(sys.modules, {"ollama": None}):
            with self.assertRaisesRegex(RuntimeError, "ollama package is not installed"):
                strategy_base.list_ollama_models()

        with (
            patch.object(GenAIStrategyBase, "infer_provider", return_value=LLMProvider.OPENAI),
            patch.object(strategy_base, "list_openai_models", return_value=["openai"]) as list_openai,
        ):
            self.assertEqual(strategy_base.list_models(model_name="test"), ["openai"])
        list_openai.assert_called_once_with()

        with (
            patch.object(GenAIStrategyBase, "infer_provider", return_value=LLMProvider.GOOGLE),
            patch.object(strategy_base, "list_gemini_models", return_value=["gemini"]) as list_gemini,
        ):
            self.assertEqual(strategy_base.list_models(model_name="test"), ["gemini"])
        list_gemini.assert_called_once_with()

        with (
            patch.object(GenAIStrategyBase, "infer_provider", return_value=cast(LLMProvider, "invalid")),
            self.assertRaisesRegex(ValueError, "Unsupported LLM provider"),
        ):
            strategy_base.list_models(model_name="test")

        with (
            patch.object(GenAIStrategyBase, "infer_provider", return_value=LLMProvider.OLLAMA),
            patch.object(strategy_base, "list_ollama_models", return_value=["local"]) as list_ollama,
        ):
            self.assertEqual(strategy_base.list_models(model_name="test"), ["local"])
        list_ollama.assert_called_once_with()


class TestRagHelper(unittest.TestCase):
    def test_iter_description_files_and_read_text(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            nested = root / "nested"
            nested.mkdir()
            first = root / "a.md"
            second = nested / "b.md"
            ignored = root / "c.mod"
            first.write_text(" abc ", encoding="utf-8")
            second.write_text("123456", encoding="utf-8")
            ignored.write_text("not a description", encoding="utf-8")

            self.assertEqual(rag_helper._iter_description_files(root), [first, second])
            self.assertEqual(rag_helper._read_text(first), "abc")
            self.assertEqual(rag_helper._read_text(second, max_chars=3), "123")

    def test_rank_problem_descriptions_handles_missing_and_empty_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            with self.assertRaises(FileNotFoundError):
                rag_helper.rank_problem_descriptions("query", root / "missing")

            self.assertEqual(rag_helper.rank_problem_descriptions("query", root), [])
            (root / "empty.md").write_text("   ", encoding="utf-8")
            self.assertEqual(rag_helper.rank_problem_descriptions("query", root), [])

    def test_description_files_from_roots_are_recursive_and_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            nested = root / "opl_models" / "category"
            nested.mkdir(parents=True)
            description = nested / "sample.md"
            description.write_text("sample", encoding="utf-8")

            descriptions = rag_helper._iter_description_files_from_roots([root / "opl_models", nested])

        self.assertEqual(descriptions, [description])

    def test_rank_problem_descriptions_combines_roots_and_ignores_missing_optional_root(self) -> None:
        class FakeModel:
            def encode(self, texts, **kwargs):
                if texts == ["query"]:
                    return [[1.0, 0.0]]
                return [[0.2, 0.0], [0.9, 0.0]]

        class FakeScores:
            def __init__(self, values):
                self.values = values

            def cpu(self):
                return self

            def tolist(self):
                return self.values

        fake_torch = SimpleNamespace(matmul=lambda doc_embs, query_emb: FakeScores([row[0] for row in doc_embs]))

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            packaged = root / "packaged"
            local = root / "workspace" / "opl_models" / "nested"
            packaged.mkdir()
            local.mkdir(parents=True)
            packaged_description = packaged / "packaged.md"
            local_description = local / "local.md"
            packaged_description.write_text("packaged", encoding="utf-8")
            local_description.write_text("local", encoding="utf-8")

            with (
                patch.object(rag_helper, "_load_model", return_value=FakeModel()),
                patch.dict(sys.modules, {"torch": fake_torch}),
            ):
                ranked = rag_helper.rank_problem_descriptions(
                    "query", [packaged, root / "missing", root / "workspace" / "opl_models"], top_k=1
                )

        self.assertEqual(ranked[0]["path"], str(local_description))

    def test_rank_problem_descriptions_returns_ranked_previews(self) -> None:
        class FakeModel:
            def encode(self, texts, **kwargs):
                if texts == ["query"]:
                    return [[1.0, 0.0]]
                return [[0.9, 0.0], [0.2, 0.0]]

        class FakeScores:
            def __init__(self, values):
                self.values = values

            def cpu(self):
                return self

            def tolist(self):
                return self.values

        fake_torch = SimpleNamespace(matmul=lambda doc_embs, query_emb: FakeScores([row[0] for row in doc_embs]))

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            low = root / "low.md"
            high = root / "high.md"
            low.write_text("low\nmatch", encoding="utf-8")
            high.write_text("high match", encoding="utf-8")

            with (
                patch.object(rag_helper, "_load_model", return_value=FakeModel()),
                patch.dict(sys.modules, {"torch": fake_torch}),
            ):
                ranked = rag_helper.rank_problem_descriptions("query", root, top_k=1)

        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["path"], str(high))
        self.assertEqual(ranked[0]["score"], 0.9)
        self.assertEqual(ranked[0]["preview"], "high match")


class TestStrategyModuleHelpers(unittest.TestCase):
    def test_generate_observable_mapping_supplies_tuple_column_names(self) -> None:
        model = """
tuple Arc { int from; int to; }
{Arc} Arcs = ...;
dvar float+ flow[Arcs];
maximize totalFlow: sum(a in Arcs) flow[a];
subject to { forall(a in Arcs) flow[a] <= 10; }
"""
        data = "Arcs = { <0, 1>, <1, 2> };"
        response = json.dumps(
            {
                "mapping": [
                    {
                        "name": "flow",
                        "left": {"coefficients": {"flow[(0, 1)]": 1}},
                        "right": {"coefficients": {"flow[(0, 1)]": 1}},
                    }
                ]
            }
        )
        with patch.object(pyopl_generative, "_llm_generate_text", return_value=response) as llm:
            pyopl_generative.generative_observable_mapping(model, model, data, data)

        prompt = llm.call_args.args[2]
        self.assertIn('"flow[(0, 1)]"', prompt)
        self.assertIn('"flow[(1, 2)]"', prompt)
        self.assertIn("Use only names from the supplied column lists", prompt)

    def test_generate_observable_mapping_parses_fenced_json_and_grounds_prompt(self) -> None:
        response = """Here is the mapping:
```json
[{"name": "flow", "left": {"coefficients": {"x[1]": 1}},
  "right": {"coefficients": {"y[1]": 1}}}]
```"""
        with (
            patch.object(pyopl_generative, "linear_problem_from_opl", return_value=SimpleNamespace(var_names=["x[1]"])),
            patch.object(pyopl_generative, "_llm_generate_text", return_value=response) as llm,
        ):
            mapping_text = pyopl_generative.generative_observable_mapping(
                "left model text",
                "right model text",
                "left data text",
                "right data text",
                model_name="test-model",
                llm_provider="openai",
            )

        self.assertEqual(
            json.loads(mapping_text),
            [
                {
                    "name": "flow",
                    "left": {"coefficients": {"x[1]": 1}},
                    "right": {"coefficients": {"y[1]": 1}},
                }
            ],
        )
        prompt = llm.call_args.args[2]
        self.assertIn("left model text", prompt)
        self.assertIn("right model text", prompt)
        self.assertIn("left data text", prompt)
        self.assertIn("right data text", prompt)
        self.assertIn("concrete supplied-instance mapping", prompt)
        self.assertIn("order_t", prompt)
        self.assertIn("x_1_2 + x_1_3 + x_1_4 + x_1_5 + x_1_6", prompt)
        self.assertIn("do not add inventory or backorder pairs", prompt)
        self.assertTrue(llm.call_args.kwargs["expected_json"])

    def test_generate_observable_mapping_rejects_missing_mapping_list(self) -> None:
        with patch.object(pyopl_generative, "_llm_generate_text", return_value='{"observables": []}'):
            with self.assertRaisesRegex(ValueError, "mapping.*list"):
                pyopl_generative.generative_observable_mapping("left", "right")

    def test_generate_observable_mapping_rejects_incomplete_entry(self) -> None:
        response = '{"mapping": [{"name": "flow", "left": {}}]}'
        with patch.object(pyopl_generative, "_llm_generate_text", return_value=response):
            with self.assertRaisesRegex(ValueError, "left and right expressions"):
                pyopl_generative.generative_observable_mapping("left", "right")

    def test_generate_indexed_observable_mapping_validates_and_canonicalizes_candidate(self) -> None:
        response = json.dumps(
            {
                "schema": "pyopl.uniform_indexed_observable_mapping",
                "version": 1,
                "observables": [
                    {
                        "kind": "indexed",
                        "name": "x[{t}]",
                        "left": {"expression": "x[t]", "domains": [{"name": "t", "range": "1..N"}]},
                        "right": {"expression": "x[t]", "domains": [{"name": "t", "range": "1..N"}]},
                    }
                ],
                "parameter_contract": {"facts": [], "indexed_facts": []},
            }
        )
        with patch.object(pyopl_generative, "_llm_generate_text", return_value=response) as llm:
            mapping_text = pyopl_generative.generative_indexed_observable_mapping(
                "left model text",
                "right model text",
                model_name="test-model",
                llm_provider="openai",
            )

        payload = json.loads(mapping_text)
        self.assertEqual(payload["schema"], "pyopl.uniform_indexed_observable_mapping")
        self.assertEqual(payload["observables"][0]["name"], "x[{t}]")
        self.assertIn("parameter_contract", payload)
        self.assertIn("left model text", llm.call_args.args[2])
        self.assertTrue(llm.call_args.kwargs["expected_json"])

        indexed_prompt = llm.call_args.args[2]
        self.assertIn("compact source form", indexed_prompt)
        self.assertIn("The observable list defines the projection being compared", indexed_prompt)
        self.assertIn("discrete choice and the other", indexed_prompt)
        self.assertIn("Do not add continuous quantity, flow", indexed_prompt)
        self.assertIn("smallest set of directly corresponding decision families", indexed_prompt)
        self.assertIn('such as "1..N" alone does not establish', indexed_prompt)
        self.assertIn('"N >= 1"', indexed_prompt)
        self.assertIn('"nonempty": false', indexed_prompt)
        self.assertIn('"facts": []', indexed_prompt)

    def test_generate_indexed_observable_mapping_rejects_invalid_schema(self) -> None:
        with patch.object(pyopl_generative, "_llm_generate_text", return_value='{"schema": "wrong"}'):
            with self.assertRaisesRegex(ValueError, "unsupported uniform indexed mapping schema"):
                pyopl_generative.generative_indexed_observable_mapping("left", "right")

    def test_strategy_json_extractors_parse_fenced_objects(self) -> None:
        payload = 'lead in```json\n{"model": "m", "data": "d"}\n```tail'

        for module_name in STRATEGY_MODULE_NAMES:
            with self.subTest(module=module_name):
                module = import_module(module_name)
                self.assertEqual(module.extract_json_from_markdown(payload), '{"model": "m", "data": "d"}')
                self.assertEqual(module._json_loads_relaxed(payload), {"model": "m", "data": "d"})

    def test_cafa_rejects_non_object_json_payload(self) -> None:
        cafa = import_module("pyopl.genai.pyopl_cafa")

        with self.assertRaisesRegex(ValueError, "Expected a JSON object"):
            cafa._json_loads_relaxed("[1, 2, 3]")

    def test_tree_of_thoughts_relaxed_json_prefers_model_data_array(self) -> None:
        text = 'bad candidate {"note": "ignore"}\n```json\n[{"model": "m1", "data": "d1"}, {"model": "m2", "data": "d2"}]\n```'

        self.assertEqual(
            pyopl_tree_of_thoughts._json_loads_relaxed(text),
            [{"model": "m1", "data": "d1"}, {"model": "m2", "data": "d2"}],
        )

    def test_strategy_provider_and_llm_wrappers_delegate_to_base(self) -> None:
        for module_name in STRATEGY_MODULE_NAMES:
            module = import_module(module_name)
            with self.subTest(module=module_name):
                self.assertEqual(module._infer_provider("google", "gemini-test"), module.LLMProvider.GOOGLE)

                with patch.object(
                    module._BASE, "llm_generate_text", return_value=("{}", {"prompt_tokens": 1, "completion_tokens": 2})
                ) as llm:
                    content, usage = module._llm_generate_text(
                        module.LLMProvider.OPENAI,
                        "gpt-test",
                        "prompt",
                        max_tokens=10,
                        temperature=0.1,
                        stop=["END"],
                        capture_usage=True,
                    )

                self.assertEqual(content, "{}")
                self.assertEqual(usage, {"prompt_tokens": 1, "completion_tokens": 2})
                kwargs = llm.call_args.kwargs
                self.assertEqual(kwargs["provider"].name, "OPENAI")
                self.assertEqual(kwargs["model_name"], "gpt-test")
                self.assertEqual(kwargs["input_text"], "prompt")

    def test_prompt_builders_include_problem_grammar_and_outputs(self) -> None:
        prompt_builders = [
            ("pyopl.genai.pyopl_standard", "_build_standard_generation_prompt"),
            ("pyopl.genai.pyopl_generative", "_build_generation_prompt"),
            ("pyopl.genai.pyopl_chain_of_thought", "_build_cot_generation_prompt"),
            ("pyopl.genai.pyopl_reflexion", "_build_reflexion_generation_prompt"),
            ("pyopl.genai.pyopl_cafa", "_build_cafa_generation_prompt"),
        ]

        for module_name, builder_name in prompt_builders:
            module = import_module(module_name)
            builder = getattr(module, builder_name)
            with self.subTest(module=module_name):
                try:
                    built = builder("capacity problem", "grammar ref", [])
                except TypeError:
                    built = builder("capacity problem", "grammar ref", [], [])

                self.assertIn("capacity problem", built)
                self.assertIn("grammar ref", built)
                self.assertIn("model", built.lower())
                self.assertIn("data", built.lower())

    def test_chain_of_experts_prompt_builders_include_context(self) -> None:
        coe = import_module("pyopl.genai.pyopl_chain_of_experts")
        comments = [{"expert": "Modeling Expert", "comment": "Use binary assignment variables"}]
        few_shots = [{"description": "sample", "model": "dvar int x;", "data": "x=1;", "desc_path": "a.md"}]

        knowledge = coe._format_few_shots_knowledge(few_shots)
        conductor = coe._build_conductor_prompt(
            "routing problem", ["Modeling Expert", "Data Builder"], comments, remaining_steps=2
        )
        expert = coe._build_expert_prompt("Modeling Expert", "routing problem", "grammar ref", comments, few_shots)
        reducer = coe._build_reducer_prompt("routing problem", "grammar ref", comments, few_shots)

        self.assertIn("sample", knowledge)
        self.assertIn("Data Builder", conductor)
        self.assertIn("Use binary assignment variables", expert)
        self.assertIn("routing problem", reducer)

    def test_generative_few_shot_renderer_and_prompt_normalizer(self) -> None:
        rendered = pyopl_generative._render_few_shots_section(
            [{"description": "desc", "model": "model text", "data": "data text", "model_path": "m.mod"}]
        )
        text, images = pyopl_generative._normalize_prompt_input({"text": "with image", "image": "plot.png"})

        self.assertIn("few_shot_examples", rendered)
        self.assertIn("model text", rendered)
        self.assertEqual(text, "with image")
        self.assertEqual(images, [{"path": "plot.png"}])

    def test_generative_image_description_helper_is_offline_mockable(self) -> None:
        messages: list[str] = []

        self.assertEqual(
            pyopl_generative._describe_images_for_rag(
                provider=pyopl_generative.LLMProvider.OPENAI, model_name="gpt", images=[]
            ),
            "",
        )
        self.assertEqual(
            pyopl_generative._describe_images_for_rag(
                provider=pyopl_generative.LLMProvider.OLLAMA,
                model_name="llama",
                images=[{"path": "image.png"}],
                progress=messages.append,
            ),
            "",
        )
        with patch.object(pyopl_generative, "_llm_generate_text", return_value=" described image "):
            self.assertEqual(
                pyopl_generative._describe_images_for_rag(
                    provider=pyopl_generative.LLMProvider.OPENAI,
                    model_name="gpt",
                    images=[{"path": "image.png"}],
                    progress=messages.append,
                ),
                "described image",
            )

        self.assertTrue(any("Ollama" in message for message in messages))

        with patch.object(pyopl_generative, "_llm_generate_text", return_value=(" tuple description ", {})):
            self.assertEqual(
                pyopl_generative._describe_images_for_rag(
                    provider=pyopl_generative.LLMProvider.OPENAI,
                    model_name="gpt",
                    images=[{"path": "image.png"}],
                ),
                "tuple description",
            )
        with patch.object(pyopl_generative, "_llm_generate_text", side_effect=RuntimeError("provider unavailable")):
            self.assertEqual(
                pyopl_generative._describe_images_for_rag(
                    provider=pyopl_generative.LLMProvider.OPENAI,
                    model_name="gpt",
                    images=[{"path": "image.png"}],
                    progress=messages.append,
                ),
                "",
            )
        self.assertTrue(any("generation failed" in message for message in messages))

    def test_generative_utility_wrappers_delegate_to_base(self) -> None:
        base = pyopl_generative._BASE
        with (
            patch.object(base, "read_file", return_value="file contents") as read_file,
            patch.object(base, "read_pyopl_GBNF", return_value="bnf") as read_bnf,
            patch.object(base, "read_pyopl_grammar", return_value="grammar") as read_grammar,
            patch.object(base, "read_pyopl_code", return_value="code") as read_code,
        ):
            self.assertEqual(pyopl_generative._read_file("model.mod"), "file contents")
            self.assertEqual(pyopl_generative._read_pyopl_GBNF(), "bnf")
            self.assertEqual(pyopl_generative._read_pyopl_grammar(), "grammar")
            self.assertEqual(pyopl_generative._read_pyopl_code(), "code")
        read_file.assert_called_once_with("model.mod")
        read_bnf.assert_called_once_with()
        read_grammar.assert_called_once_with()
        read_code.assert_called_once_with()

        with patch.object(base, "get_grammar_implementation", return_value="implementation") as grammar:
            self.assertEqual(pyopl_generative._get_grammar_implementation(pyopl_generative.Grammar.CODE), "implementation")
        grammar.assert_called_once_with(strategy_base.Grammar.CODE)

        with (
            patch.object(base, "safe_read_text", return_value="text") as read_text,
            patch.object(base, "find_pair_in_folder", return_value=(Path("model.mod"), Path("data.dat"))) as find_pair,
            patch.object(base, "gather_few_shots", return_value=[]) as gather,
            patch.object(base, "render_few_shots_section", return_value="examples") as render,
        ):
            self.assertEqual(pyopl_generative._safe_read_text(Path("desc.md"), max_chars=10), "text")
            self.assertEqual(
                pyopl_generative._find_pair_in_folder(Path("desc.md")),
                (Path("model.mod"), Path("data.dat")),
            )
            self.assertEqual(pyopl_generative._gather_few_shots("query", k=2, models_dir="models"), [])
            self.assertEqual(pyopl_generative._render_few_shots_section([]), "examples")
        read_text.assert_called_once_with(Path("desc.md"), max_chars=10)
        find_pair.assert_called_once_with(Path("desc.md"))
        gather.assert_called_once_with("query", k=2, models_dir="models", progress=None)
        render.assert_called_once_with([])

        with (
            patch.object(base, "_openai_client", return_value="openai") as openai,
            patch.object(base, "_google_client", return_value="google") as google,
            patch.object(base, "_ollama_generate_text", return_value="ollama") as ollama,
        ):
            self.assertEqual(pyopl_generative._openai_client(), "openai")
            self.assertEqual(pyopl_generative._google_client(), "google")
            self.assertEqual(pyopl_generative._ollama_generate_text("llama", "prompt", 10), "ollama")
        openai.assert_called_once_with()
        google.assert_called_once_with()
        ollama.assert_called_once_with(
            model_name="llama",
            prompt="prompt",
            num_predict=10,
            return_usage=False,
            enforce_json=True,
        )

        self.assertEqual(
            pyopl_generative._build_create_params("gpt", "prompt", max_tokens=8, temperature=0.1, stop=["END"]),
            {
                "model": "gpt",
                "input": "prompt",
                "response_format": {"type": "json_object"},
                "max_output_tokens": 8,
                "temperature": 0.1,
                "stop": ["END"],
            },
        )
        self.assertEqual(
            pyopl_generative._coalesce_response_text(SimpleNamespace(output_text="response")),
            "response",
        )

    def test_generative_json_and_response_text_fallback_helpers(self) -> None:
        fenced = 'prefix\n```json\n{"model": "m", "data": "d"}\n```\nsuffix'
        nested = 'extra text {"outer": {"value": 1}} trailing text'

        self.assertEqual(
            pyopl_generative.extract_json_from_markdown(fenced),
            '{"model": "m", "data": "d"}',
        )
        self.assertEqual(
            pyopl_generative.extract_json_from_markdown(nested),
            '{"outer": {"value": 1}}',
        )
        self.assertEqual(pyopl_generative.extract_json_from_markdown("no object here"), "no object here")
        self.assertEqual(
            pyopl_generative._json_loads_relaxed(fenced),
            {"model": "m", "data": "d"},
        )
        self.assertEqual(
            pyopl_generative._coalesce_response_text(
                SimpleNamespace(
                    output=[
                        SimpleNamespace(content=None, text="first"),
                        SimpleNamespace(
                            content=[
                                SimpleNamespace(text="second"),
                                {"text": "third"},
                                {"text": 4},
                                SimpleNamespace(),
                            ]
                        ),
                    ]
                )
            ),
            "firstsecondthird",
        )
        self.assertEqual(pyopl_generative._coalesce_response_text(SimpleNamespace(output=[])), "")

    def test_generative_revision_feedback_and_assessment_prompts(self) -> None:
        syntax_revision = pyopl_generative._build_revision_prompt(
            "question",
            "grammar",
            "old model",
            "old data",
            compile_errors=["missing declaration"],
            few_shots=[],
        )
        alignment_revision = pyopl_generative._build_revision_prompt(
            "question",
            "grammar",
            "old model",
            "old data",
            alignment_assessment="capacity constraint is missing",
        )
        feedback = pyopl_generative._build_feedback_prompt("question", "grammar", "model", "data")
        final_assessment = pyopl_generative._build_final_assessment_prompt(
            "question",
            "grammar",
            "model",
            "data",
            ["syntax error"],
        )

        self.assertIn("missing declaration", syntax_revision)
        self.assertIn("capacity constraint is missing", alignment_revision)
        self.assertIn("question", feedback)
        self.assertIn("model", feedback)
        self.assertIn("data", feedback)
        self.assertIn("SYNTAX ERRORS", final_assessment)
        self.assertIn("syntax error", final_assessment)

    def test_strategy_grammar_wrappers_delegate_to_base_helper(self) -> None:
        for module_name in STRATEGY_MODULE_NAMES:
            module = import_module(module_name)
            with self.subTest(module=module_name):
                self.assertEqual(module._get_grammar_implementation(module.Grammar.NONE), "")

    def test_strategy_notify_uses_progress_and_swallows_callback_errors(self) -> None:
        for module_name in STRATEGY_MODULE_NAMES:
            module = import_module(module_name)
            messages: list[str] = []
            with self.subTest(module=module_name):
                module._notify(messages.append, "hello")
                module._notify(lambda msg: (_ for _ in ()).throw(RuntimeError("boom")), "ignored")

            self.assertEqual(messages, ["hello"])

    def test_common_assessment_prompts_include_inputs(self) -> None:
        for module_name in STRATEGY_MODULE_NAMES:
            module = import_module(module_name)
            with self.subTest(module=module_name):
                alignment = module._build_alignment_prompt("user problem", "grammar ref", "model code", "data code")
                final = module._build_final_assessment_prompt(
                    "user problem", "grammar ref", "model code", "data code", "syntax issue"
                )

                self.assertIn("user problem", alignment)
                self.assertIn("model code", final)
                self.assertIn("SYNTAX ERRORS", final)

    def test_canonical_feedback_prompt_includes_inputs(self) -> None:
        feedback = feedback_validation._build_feedback_prompt("why infeasible?", "grammar ref", "model code", "data code")

        self.assertIn("why infeasible?", feedback)
        self.assertIn("grammar ref", feedback)


class TestGraphChainHelpers(unittest.TestCase):
    def make_context(self, *, model_file: str = "out.mod", data_file: str = "out.dat", alignment_check: bool = True):
        return pyopl_generative_graphchain.ExecutionContext(
            problem_text="minimize cost",
            model_file=model_file,
            data_file=data_file,
            model_name="gpt-test",
            grammar_mode=pyopl_generative_graphchain.Grammar.NONE,
            provider=pyopl_generative_graphchain.LLMProvider.OPENAI,
            max_iterations=2,
            do_alignment_check=alignment_check,
            temperature=None,
            stop=None,
            progress=None,
            few_shots=[],
            grammar_implementation="grammar",
            model_code="model old",
            data_code="data old",
        )

    def test_graphchain_json_repair_prompt_and_validators(self) -> None:
        repair = pyopl_generative_graphchain._build_json_repair_prompt("original", "not json", "bad parse")

        self.assertIn("original", repair)
        self.assertIn("not json", repair)
        self.assertIn("bad parse", repair)
        self.assertIsNone(pyopl_generative_graphchain._validate_model_data_payload({"model": "m", "data": "d"}))
        self.assertEqual(
            pyopl_generative_graphchain._validate_model_data_payload({"model": "m"}),
            'Response JSON is missing string key "data"',
        )
        self.assertIsNone(pyopl_generative_graphchain._validate_alignment_payload({"aligned": True, "assessment": "ok"}))
        self.assertEqual(
            pyopl_generative_graphchain._validate_alignment_payload({"aligned": "yes", "assessment": "ok"}),
            'Response JSON is missing boolean key "aligned"',
        )

    def test_graphchain_generate_json_object_retries_once(self) -> None:
        calls = []

        def fake_llm_generate_text(**kwargs):
            calls.append(kwargs["input_text"])
            if len(calls) == 1:
                return "not json", {"prompt_tokens": 3, "completion_tokens": 4}
            return '{"model": "m", "data": "d"}', {"prompt_tokens": 5, "completion_tokens": 6}

        progress_messages: list[str] = []
        with patch.object(pyopl_generative_graphchain, "_llm_generate_text", side_effect=fake_llm_generate_text):
            payload, usage = pyopl_generative_graphchain._generate_json_object(
                provider=pyopl_generative_graphchain.LLMProvider.OPENAI,
                model_name="gpt-test",
                input_text="make json",
                images=None,
                max_tokens=10,
                temperature=None,
                stop=None,
                progress=progress_messages.append,
                validator=pyopl_generative_graphchain._validate_model_data_payload,
                retry_label="generate",
            )

        self.assertEqual(payload, {"model": "m", "data": "d"})
        self.assertEqual(usage, {"prompt_tokens": 8, "completion_tokens": 10})
        self.assertEqual(len(calls), 2)
        self.assertIn("retry_instruction", calls[1])
        self.assertTrue(progress_messages)

    def test_graphchain_generate_json_object_raises_after_invalid_retry(self) -> None:
        calls = []

        def fake_llm_generate_text(**kwargs):
            calls.append(kwargs["input_text"])
            if len(calls) == 1:
                return "", {"prompt_tokens": 1, "completion_tokens": 2}
            return '{"model": "m"}', {"prompt_tokens": 3, "completion_tokens": 4}

        with (
            patch.object(pyopl_generative_graphchain, "_llm_generate_text", side_effect=fake_llm_generate_text),
            self.assertRaisesRegex(RuntimeError, 'missing string key "data"'),
        ):
            pyopl_generative_graphchain._generate_json_object(
                provider=pyopl_generative_graphchain.LLMProvider.OPENAI,
                model_name="gpt-test",
                input_text="make json",
                images=None,
                max_tokens=10,
                temperature=None,
                stop=None,
                progress=None,
                validator=pyopl_generative_graphchain._validate_model_data_payload,
                retry_label="generate",
            )

        self.assertEqual(len(calls), 2)
        self.assertIn("retry_instruction", calls[1])

    def test_graphchain_context_validation_and_callable_error_wrapper(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_iterations"):
            self.make_context().__class__(
                problem_text="p",
                model_file="m.mod",
                data_file="d.dat",
                model_name="gpt",
                grammar_mode=pyopl_generative_graphchain.Grammar.NONE,
                provider=pyopl_generative_graphchain.LLMProvider.OPENAI,
                max_iterations=0,
                do_alignment_check=True,
                temperature=None,
                stop=None,
                progress=None,
                few_shots=[],
            )

        class FailingNode(pyopl_generative_graphchain.GraphNode):
            async def execute(self, context):
                raise RuntimeError("node boom")

        result = asyncio.run(FailingNode("fail")(self.make_context()))
        self.assertFalse(result.success)
        self.assertIn("node boom", result.error)

    def test_graphchain_generate_alignment_revision_and_final_nodes(self) -> None:
        context = self.make_context()

        with patch.object(
            pyopl_generative_graphchain,
            "_generate_json_object",
            return_value=({"model": "new model", "data": "new data"}, {"prompt_tokens": 2, "completion_tokens": 3}),
        ):
            result = asyncio.run(pyopl_generative_graphchain.GenerateNode("gen").execute(context))
        self.assertTrue(result.success)
        self.assertEqual(context.model_code, "new model")
        self.assertEqual(context.total_prompt_tokens, 2)

        with patch.object(
            pyopl_generative_graphchain,
            "_generate_json_object",
            return_value=({"aligned": False, "assessment": "needs capacity"}, {"prompt_tokens": 5, "completion_tokens": 7}),
        ):
            result = asyncio.run(pyopl_generative_graphchain.CheckAlignmentNode("align").execute(context))
        self.assertTrue(result.success)
        self.assertFalse(context.aligned)
        self.assertEqual(context.alignment_assessment, "needs capacity")

        with patch.object(
            pyopl_generative_graphchain,
            "_generate_json_object",
            return_value=({"model": "syntax model", "data": "syntax data"}, {"prompt_tokens": 1, "completion_tokens": 1}),
        ):
            result = asyncio.run(pyopl_generative_graphchain.ReviseSyntaxNode("revise_syntax").execute(context))
        self.assertTrue(result.success)
        self.assertEqual(context.last_revision_type, "syntax")

        with patch.object(
            pyopl_generative_graphchain,
            "_generate_json_object",
            return_value=({"model": "aligned model", "data": "aligned data"}, {"prompt_tokens": 1, "completion_tokens": 1}),
        ):
            result = asyncio.run(pyopl_generative_graphchain.ReviseAlignmentNode("revise_alignment").execute(context))
        self.assertTrue(result.success)
        self.assertEqual(context.last_revision_type, "alignment")

        with patch.object(
            pyopl_generative_graphchain,
            "_llm_generate_text",
            return_value=(" final assessment ", {"prompt_tokens": 2, "completion_tokens": 2}),
        ):
            result = asyncio.run(pyopl_generative_graphchain.FinalAssessmentNode("final").execute(context))
        self.assertTrue(result.success)
        self.assertEqual(context.alignment_assessment, "final assessment")

    def test_graphchain_nodes_report_errors_and_empty_assessment_fallback(self) -> None:
        context = self.make_context()
        with patch.object(
            pyopl_generative_graphchain,
            "_generate_json_object",
            side_effect=RuntimeError("provider failed"),
        ):
            result = asyncio.run(pyopl_generative_graphchain.GenerateNode("generate").execute(context))
        self.assertFalse(result.success)
        self.assertIn("provider failed", result.error)

        context = self.make_context()
        with patch.object(
            pyopl_generative_graphchain.OPLCompiler,
            "compile_model",
            side_effect=pyopl_generative_graphchain.SemanticError("invalid model"),
        ):
            result = asyncio.run(pyopl_generative_graphchain.CheckSyntaxNode("syntax").execute(context))
        self.assertTrue(result.success)
        self.assertFalse(context.syntax_valid)
        self.assertEqual(context.syntax_errors, ["Semantic Error: invalid model"])

        for node in (
            pyopl_generative_graphchain.CheckAlignmentNode("alignment"),
            pyopl_generative_graphchain.ReviseSyntaxNode("revise_syntax"),
            pyopl_generative_graphchain.ReviseAlignmentNode("revise_alignment"),
        ):
            with (
                self.subTest(node=node.name),
                patch.object(
                    pyopl_generative_graphchain,
                    "_generate_json_object",
                    side_effect=RuntimeError("provider failed"),
                ),
            ):
                result = asyncio.run(node.execute(self.make_context()))
                self.assertFalse(result.success)
                self.assertIn("provider failed", result.error)

        context = self.make_context()
        with patch.object(pyopl_generative_graphchain, "_llm_generate_text", return_value=("", {})):
            result = asyncio.run(pyopl_generative_graphchain.FinalAssessmentNode("assessment").execute(context))
        self.assertTrue(result.success)
        self.assertEqual(context.alignment_assessment, "Unable to generate assessment.")

    def test_graphchain_syntax_and_save_nodes_are_offline(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_file = str(Path(td) / "nested" / "model.mod")
            data_file = str(Path(td) / "nested" / "data.dat")
            context = self.make_context(model_file=model_file, data_file=data_file)
            context.model_code = "dvar int x;\nminimize x;\nsubject to { c: x >= 1; }"
            context.data_code = ""

            with patch.object(pyopl_generative_graphchain.OPLCompiler, "compile_model", return_value=None):
                syntax_result = asyncio.run(pyopl_generative_graphchain.CheckSyntaxNode("syntax").execute(context))
            save_result = asyncio.run(pyopl_generative_graphchain.SaveFilesNode("save").execute(context))

            self.assertTrue(syntax_result.success)
            self.assertTrue(context.syntax_valid)
            self.assertTrue(save_result.success)
            self.assertEqual(Path(model_file).read_text(encoding="utf-8"), context.model_code)
            self.assertEqual(Path(data_file).read_text(encoding="utf-8"), context.data_code)

    def test_graphchain_save_node_cleans_up_after_atomic_write_failure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_file = str(Path(td) / "model.mod")
            data_file = str(Path(td) / "data.dat")
            context = self.make_context(model_file=model_file, data_file=data_file)
            context.model_code = "model"
            context.data_code = "data"

            with patch.object(pyopl_generative_graphchain.os, "replace", side_effect=OSError("disk full")):
                result = asyncio.run(pyopl_generative_graphchain.SaveFilesNode("save").execute(context))

            self.assertFalse(result.success)
            self.assertIn("disk full", result.error)
            self.assertEqual(list(Path(td).iterdir()), [])

    def test_graphchain_executor_happy_path_with_mocked_nodes(self) -> None:
        context = self.make_context(alignment_check=False)

        async def fake_generate(ctx):
            ctx.model_code = "m"
            ctx.data_code = "d"
            return pyopl_generative_graphchain.NodeExecutionResult(ctx)

        async def fake_syntax(ctx):
            ctx.syntax_valid = True
            return pyopl_generative_graphchain.NodeExecutionResult(ctx)

        async def fake_final(ctx):
            ctx.alignment_assessment = "done"
            return pyopl_generative_graphchain.NodeExecutionResult(ctx)

        async def fake_save(ctx):
            ctx.saved = True
            return pyopl_generative_graphchain.NodeExecutionResult(ctx)

        executor = pyopl_generative_graphchain.GraphChainExecutor(max_iterations=2)
        with (
            patch.object(pyopl_generative_graphchain, "_get_grammar_implementation", return_value="grammar"),
            patch.object(executor.gen, "execute", side_effect=fake_generate),
            patch.object(executor.check_syntax, "execute", side_effect=fake_syntax),
            patch.object(executor.final_assessment, "execute", side_effect=fake_final),
            patch.object(executor.save_files, "execute", side_effect=fake_save),
        ):
            result = asyncio.run(executor.execute(context))

        self.assertEqual(result.iteration, 1)
        self.assertEqual(result.alignment_assessment, "done")

    def test_tree_of_thoughts_expand_prompt_with_parent_and_examples(self) -> None:
        prompt = pyopl_tree_of_thoughts._build_tot_expand_prompt(
            "routing problem",
            "grammar ref",
            few_shots=[{"description": "sample", "model": "m", "data": "d"}],
            k=2,
            parent_model="old m",
            parent_data="old d",
        )

        self.assertIn("Generate 2 diverse", prompt)
        self.assertIn("parent_candidate", prompt)
        self.assertIn("sample", prompt)

    def test_graphchain_public_async_and_sync_wrappers_with_mocked_executor(self) -> None:
        async def fake_execute(self, context):
            context.iteration = 1
            context.alignment_assessment = "wrapped assessment"
            context.syntax_errors = []
            context.total_prompt_tokens = 11
            context.total_completion_tokens = 13
            return context

        with (
            patch.object(pyopl_generative_graphchain.GraphChainExecutor, "execute", fake_execute),
            patch.object(pyopl_generative_graphchain, "_estimate_costs", return_value={"total_cost": 0.25}),
        ):
            stats = cast(
                Dict[str, Any],
                asyncio.run(
                    pyopl_generative_graphchain.generative_solve_async(
                        "problem text",
                        "model.mod",
                        "data.dat",
                        model_name="gpt-test",
                        mode=pyopl_generative_graphchain.Grammar.NONE,
                        iterations=0,
                        return_statistics=True,
                        alignment_check=False,
                        few_shot=False,
                    )
                ),
            )
            assessment = pyopl_generative_graphchain.generative_solve_graphchain(
                "problem text",
                "model.mod",
                "data.dat",
                model_name="gpt-test",
                mode=pyopl_generative_graphchain.Grammar.NONE,
                iterations=1,
                return_statistics=False,
                alignment_check=False,
                few_shot=False,
            )

        self.assertEqual(stats["iterations"], 1)
        self.assertEqual(stats["assessment"], "wrapped assessment")
        self.assertEqual(stats["cost"]["usage"], {"prompt_tokens": 11, "completion_tokens": 13})
        self.assertEqual(stats["cost"]["estimated_costs"], {"total_cost": 0.25})
        self.assertEqual(assessment, "wrapped assessment")

    def test_graphchain_async_alignment_revision_uses_image_context(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_file = str(Path(td) / "model.mod")
            data_file = str(Path(td) / "data.dat")
            responses = [
                (json.dumps({"model": "first model", "data": "first data"}), {"prompt_tokens": 1, "completion_tokens": 2}),
                (json.dumps({"aligned": False, "assessment": "add capacity"}), {"prompt_tokens": 3, "completion_tokens": 4}),
                (json.dumps({"model": "revised model", "data": "revised data"}), {"prompt_tokens": 5, "completion_tokens": 6}),
                (json.dumps({"aligned": True, "assessment": "aligned"}), {"prompt_tokens": 7, "completion_tokens": 8}),
            ]

            with (
                patch.object(pyopl_generative_graphchain, "_llm_generate_text", side_effect=responses) as llm,
                patch.object(pyopl_generative_graphchain, "_describe_images_for_rag", return_value="image context"),
                patch.object(pyopl_generative_graphchain, "_gather_few_shots", return_value=[]) as gather,
                patch.object(pyopl_generative_graphchain.OPLCompiler, "compile_model", return_value=None),
                patch.object(pyopl_generative_graphchain, "_estimate_costs", side_effect=RuntimeError("pricing unavailable")),
            ):
                stats = cast(
                    Dict[str, Any],
                    asyncio.run(
                        pyopl_generative_graphchain.generative_solve_async(
                            cast(
                                strategy_base.PromptWithImages,
                                {"text": "solve the pictured problem", "image": "problem.png"},
                            ),
                            model_file,
                            data_file,
                            model_name="gpt-test",
                            mode=pyopl_generative_graphchain.Grammar.NONE,
                            iterations=2,
                            return_statistics=True,
                            few_shot=True,
                        )
                    ),
                )

            generated_model = Path(model_file).read_text(encoding="utf-8")
            generated_data = Path(data_file).read_text(encoding="utf-8")

        self.assertEqual(stats["assessment"], "aligned")
        self.assertEqual(stats["iterations"], 2)
        self.assertEqual(stats["syntax_errors"], [])
        self.assertEqual(stats["cost"]["usage"], {"prompt_tokens": 16, "completion_tokens": 20})
        self.assertEqual(stats["cost"]["estimated_costs"], {})
        self.assertIn("image context", gather.call_args.args[0])
        self.assertEqual(llm.call_count, 4)
        self.assertEqual(generated_model, "revised model")
        self.assertEqual(generated_data, "revised data")

    def test_graphchain_async_recovers_syntax_and_returns_final_assessment(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_file = str(Path(td) / "model.mod")
            data_file = str(Path(td) / "data.dat")
            responses = [
                (json.dumps({"model": "invalid model", "data": "data"}), {"prompt_tokens": 1, "completion_tokens": 1}),
                (
                    json.dumps({"model": "repaired model", "data": "repaired data"}),
                    {"prompt_tokens": 2, "completion_tokens": 2},
                ),
                ("syntax repaired", {"prompt_tokens": 3, "completion_tokens": 3}),
            ]

            with (
                patch.object(pyopl_generative_graphchain, "_llm_generate_text", side_effect=responses) as llm,
                patch.object(
                    pyopl_generative_graphchain.OPLCompiler,
                    "compile_model",
                    side_effect=[
                        pyopl_generative_graphchain.SemanticError("first syntax error"),
                        None,
                    ],
                ),
                patch.object(pyopl_generative_graphchain, "_estimate_costs", return_value={}),
            ):
                stats = cast(
                    Dict[str, Any],
                    asyncio.run(
                        pyopl_generative_graphchain.generative_solve_async(
                            "repair this model",
                            model_file,
                            data_file,
                            model_name="gpt-test",
                            mode=pyopl_generative_graphchain.Grammar.NONE,
                            iterations=2,
                            return_statistics=True,
                            alignment_check=False,
                            few_shot=False,
                        )
                    ),
                )

            generated_model = Path(model_file).read_text(encoding="utf-8")

        self.assertEqual(stats["assessment"], "syntax repaired")
        self.assertEqual(stats["iterations"], 2)
        self.assertEqual(stats["syntax_errors"], [])
        self.assertIn("first syntax error", llm.call_args_list[1].kwargs["input_text"])
        self.assertEqual(generated_model, "repaired model")

    def test_graphchain_public_wrappers_surface_execution_errors(self) -> None:
        async def fail_execute(self, context):
            raise ValueError("executor failed")

        with patch.object(pyopl_generative_graphchain.GraphChainExecutor, "execute", fail_execute):
            with self.assertRaisesRegex(RuntimeError, "Generative solve failed: executor failed"):
                asyncio.run(
                    pyopl_generative_graphchain.generative_solve_async(
                        "problem",
                        "model.mod",
                        "data.dat",
                        few_shot=False,
                    )
                )

        async def call_sync_wrapper_inside_loop():
            with self.assertWarnsRegex(UserWarning, "use generative_solve_async instead"):
                pyopl_generative_graphchain.generative_solve_graphchain("problem", "model.mod", "data.dat")

        with self.assertRaisesRegex(RuntimeError, "Cannot use sync wrapper from async context"):
            asyncio.run(call_sync_wrapper_inside_loop())


class TestStrategySolveLoops(unittest.TestCase):
    def test_common_strategy_solve_loops_with_mocked_llm(self) -> None:
        module_names = [
            "pyopl.genai.pyopl_standard",
            "pyopl.genai.pyopl_chain_of_thought",
            "pyopl.genai.pyopl_reflexion",
            "pyopl.genai.pyopl_cafa",
        ]

        for module_name in module_names:
            module = import_module(module_name)
            with self.subTest(module=module_name), tempfile.TemporaryDirectory() as td:
                model_file = str(Path(td) / "model.mod")
                data_file = str(Path(td) / "data.dat")
                llm_responses = [
                    (json.dumps({"model": "dvar int x;", "data": "x=1;"}), {"prompt_tokens": 2, "completion_tokens": 3}),
                    ("offline assessment", {"prompt_tokens": 5, "completion_tokens": 7}),
                ]
                if module_name == "pyopl.genai.pyopl_reflexion":
                    llm_responses.insert(
                        1, (json.dumps({"reflection": "try simpler domains"}), {"prompt_tokens": 1, "completion_tokens": 1})
                    )

                with (
                    patch.object(module.OPLCompiler, "compile_model", return_value=None),
                    patch.object(module, "_llm_generate_text", side_effect=llm_responses) as llm,
                    patch.object(module, "_estimate_costs", return_value={"total_cost": 0.5}),
                ):
                    stats = module.generative_solve(
                        "make a tiny model",
                        model_file,
                        data_file,
                        model_name="gpt-test",
                        mode=module.Grammar.NONE,
                        iterations=1,
                        return_statistics=True,
                        alignment_check=False,
                        few_shot=False,
                    )

                self.assertEqual(Path(model_file).read_text(encoding="utf-8"), "dvar int x;")
                self.assertEqual(Path(data_file).read_text(encoding="utf-8"), "x=1;")
                self.assertEqual(stats["assessment"], "offline assessment")
                self.assertEqual(stats["syntax_errors"], [])
                self.assertEqual(stats["cost"]["estimated_costs"], {"total_cost": 0.5})
                self.assertGreaterEqual(llm.call_count, 2)

    def test_generative_legacy_loop_with_mocked_llm(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_file = str(Path(td) / "model.mod")
            data_file = str(Path(td) / "data.dat")

            with (
                patch.object(pyopl_generative.OPLCompiler, "compile_model", return_value=None),
                patch.object(
                    pyopl_generative,
                    "_llm_generate_text",
                    side_effect=[
                        (json.dumps({"model": "dvar int x;", "data": "x=1;"}), {"prompt_tokens": 2, "completion_tokens": 3}),
                        ("legacy assessment", {"prompt_tokens": 5, "completion_tokens": 7}),
                    ],
                ),
                patch.object(pyopl_generative, "_estimate_costs", return_value={"total_cost": 0.6}),
            ):
                stats = pyopl_generative.generative_solve(
                    "make a tiny model",
                    model_file,
                    data_file,
                    model_name="gpt-test",
                    mode=pyopl_generative.Grammar.NONE,
                    iterations=1,
                    return_statistics=True,
                    alignment_check=False,
                    few_shot=False,
                    use_graphchain=False,
                )

        self.assertEqual(stats["assessment"], "legacy assessment")
        self.assertEqual(stats["cost"]["estimated_costs"], {"total_cost": 0.6})

    def test_generative_alignment_success_aggregates_usage(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_file = str(Path(td) / "model.mod")
            data_file = str(Path(td) / "data.dat")
            responses = [
                (json.dumps({"model": "dvar int x;", "data": "x=1;"}), {"prompt_tokens": 2, "completion_tokens": 3}),
                (
                    json.dumps({"aligned": True, "assessment": "  aligned well  "}),
                    {"prompt_tokens": 5, "completion_tokens": 7},
                ),
            ]

            with (
                patch.object(pyopl_generative.OPLCompiler, "compile_model", return_value=None),
                patch.object(pyopl_generative, "_llm_generate_text", side_effect=responses) as llm,
                patch.object(pyopl_generative, "_estimate_costs", return_value={"total_cost": 0.4}),
            ):
                stats = pyopl_generative.generative_solve(
                    "make a tiny model",
                    model_file,
                    data_file,
                    model_name="gpt-test",
                    mode=pyopl_generative.Grammar.NONE,
                    iterations=1,
                    return_statistics=True,
                    alignment_check=True,
                    few_shot=False,
                    use_graphchain=False,
                )

        self.assertEqual(stats["assessment"], "aligned well")
        self.assertEqual(stats["syntax_errors"], [])
        self.assertEqual(stats["cost"]["usage"], {"prompt_tokens": 7, "completion_tokens": 10})
        self.assertEqual(llm.call_count, 2)

    def test_generative_alignment_failure_revises_before_success(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_file = str(Path(td) / "model.mod")
            data_file = str(Path(td) / "data.dat")
            responses = [
                (json.dumps({"model": "first model", "data": "first data"}), {"prompt_tokens": 1, "completion_tokens": 2}),
                (
                    json.dumps({"aligned": False, "assessment": "missing a capacity constraint"}),
                    {"prompt_tokens": 3, "completion_tokens": 4},
                ),
                (json.dumps({"model": "revised model", "data": "revised data"}), {"prompt_tokens": 5, "completion_tokens": 6}),
                (json.dumps({"aligned": True, "assessment": "now aligned"}), {"prompt_tokens": 7, "completion_tokens": 8}),
            ]

            with (
                patch.object(pyopl_generative.OPLCompiler, "compile_model", return_value=None),
                patch.object(pyopl_generative, "_llm_generate_text", side_effect=responses) as llm,
                patch.object(pyopl_generative, "_estimate_costs", return_value={}),
            ):
                stats = pyopl_generative.generative_solve(
                    "make a tiny model",
                    model_file,
                    data_file,
                    model_name="gpt-test",
                    mode=pyopl_generative.Grammar.NONE,
                    iterations=2,
                    return_statistics=True,
                    alignment_check=True,
                    few_shot=False,
                    use_graphchain=False,
                )
            generated_model = Path(model_file).read_text(encoding="utf-8")

        self.assertEqual(stats["assessment"], "now aligned")
        self.assertEqual(stats["iterations"], 2)
        self.assertIn("missing a capacity constraint", llm.call_args_list[2].kwargs["input_text"])
        self.assertEqual(generated_model, "revised model")

    def test_generative_compile_errors_trigger_revision_and_final_assessment(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_file = str(Path(td) / "model.mod")
            data_file = str(Path(td) / "data.dat")
            responses = [
                (json.dumps({"model": "first model", "data": "first data"}), {"prompt_tokens": 1, "completion_tokens": 2}),
                (json.dumps({"model": "second model", "data": "second data"}), {"prompt_tokens": 3, "completion_tokens": 4}),
                ("final assessment", {"prompt_tokens": 5, "completion_tokens": 6}),
            ]
            compile_errors = [
                strategy_base.SemanticError("first semantic failure"),
                RuntimeError("second parse failure"),
            ]

            with (
                patch.object(pyopl_generative.OPLCompiler, "compile_model", side_effect=compile_errors),
                patch.object(pyopl_generative, "_llm_generate_text", side_effect=responses) as llm,
                patch.object(pyopl_generative, "_estimate_costs", return_value={}),
            ):
                stats = pyopl_generative.generative_solve(
                    "make a tiny model",
                    model_file,
                    data_file,
                    model_name="gpt-test",
                    mode=pyopl_generative.Grammar.NONE,
                    iterations=2,
                    return_statistics=True,
                    alignment_check=True,
                    few_shot=False,
                    use_graphchain=False,
                )

        self.assertEqual(stats["assessment"], "final assessment")
        self.assertEqual(stats["syntax_errors"], ["RuntimeError: second parse failure"])
        self.assertIn("first semantic failure", llm.call_args_list[1].kwargs["input_text"])
        self.assertIn("second parse failure", llm.call_args_list[2].kwargs["input_text"])

    def test_generative_image_context_enriches_few_shot_query(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_file = str(Path(td) / "model.mod")
            data_file = str(Path(td) / "data.dat")
            responses = [
                (json.dumps({"model": "dvar int x;", "data": "x=1;"}), {"prompt_tokens": 1, "completion_tokens": 2}),
                ("image assessment", {"prompt_tokens": 3, "completion_tokens": 4}),
            ]

            with (
                patch.object(pyopl_generative.OPLCompiler, "compile_model", return_value=None),
                patch.object(pyopl_generative, "_describe_images_for_rag", return_value="image-derived context"),
                patch.object(pyopl_generative, "_gather_few_shots", return_value=[]) as gather,
                patch.object(pyopl_generative, "_llm_generate_text", side_effect=responses),
                patch.object(pyopl_generative, "_estimate_costs", return_value={}),
            ):
                pyopl_generative.generative_solve(
                    {"text": "describe this problem", "image": "problem.png"},
                    model_file,
                    data_file,
                    model_name="gpt-test",
                    mode=pyopl_generative.Grammar.NONE,
                    iterations=1,
                    alignment_check=False,
                    few_shot=True,
                    use_graphchain=False,
                )

        rag_query = gather.call_args.args[0]
        self.assertIn("describe this problem", rag_query)
        self.assertIn("image-derived context", rag_query)

    def test_generative_solve_delegates_to_graphchain_by_default(self) -> None:
        with patch.object(pyopl_generative, "generative_solve_graphchain", return_value="delegated") as graphchain:
            result = pyopl_generative.generative_solve("prompt", "model.mod", "data.dat")

        self.assertEqual(result, "delegated")
        self.assertEqual(graphchain.call_args.kwargs["prompt"], "prompt")
        self.assertEqual(graphchain.call_args.kwargs["model_file"], "model.mod")

    def test_generative_legacy_loop_rejects_empty_generation_and_invalid_alignment(self) -> None:
        for responses, expected_error in [
            ([""], "Empty model response"),
            (["not json"], "Failed to parse model response as JSON"),
            (
                [
                    json.dumps({"model": "dvar int x;", "data": "x=1;"}),
                    "{}",
                ],
                "Invalid alignment response JSON",
            ),
            (
                [
                    json.dumps({"model": "dvar int x;", "data": "x=1;"}),
                    "",
                ],
                "Empty alignment response",
            ),
        ]:
            with self.subTest(expected_error=expected_error), tempfile.TemporaryDirectory() as td:
                model_file = str(Path(td) / "model.mod")
                data_file = str(Path(td) / "data.dat")
                with (
                    patch.object(pyopl_generative.OPLCompiler, "compile_model", return_value=None),
                    patch.object(
                        pyopl_generative,
                        "_llm_generate_text",
                        side_effect=[(response, {}) for response in responses],
                    ),
                    patch.object(pyopl_generative, "_estimate_costs", return_value={}),
                    self.assertRaisesRegex(RuntimeError, expected_error),
                ):
                    pyopl_generative.generative_solve(
                        "make a tiny model",
                        model_file,
                        data_file,
                        model_name="gpt-test",
                        mode=pyopl_generative.Grammar.NONE,
                        iterations=1,
                        alignment_check=True,
                        few_shot=False,
                        use_graphchain=False,
                    )

    def test_tree_of_thoughts_solve_loop_with_mocked_llm(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_file = str(Path(td) / "model.mod")
            data_file = str(Path(td) / "data.dat")

            with (
                patch.object(pyopl_tree_of_thoughts.OPLCompiler, "compile_model", return_value=None),
                patch.object(
                    pyopl_tree_of_thoughts,
                    "_llm_generate_text",
                    side_effect=[
                        (json.dumps([{"model": "dvar int x;", "data": "x=1;"}]), {"prompt_tokens": 2, "completion_tokens": 3}),
                        ("tot assessment", {"prompt_tokens": 5, "completion_tokens": 7}),
                    ],
                ),
                patch.object(pyopl_tree_of_thoughts, "_estimate_costs", return_value={"total_cost": 0.7}),
            ):
                stats = pyopl_tree_of_thoughts.generative_solve(
                    "make a tiny model",
                    model_file,
                    data_file,
                    model_name="gpt-test",
                    mode=pyopl_tree_of_thoughts.Grammar.NONE,
                    iterations=1,
                    return_statistics=True,
                    alignment_check=False,
                    few_shot=False,
                )

            self.assertEqual(Path(model_file).read_text(encoding="utf-8"), "dvar int x;")
            self.assertEqual(stats["assessment"], "tot assessment")
            self.assertEqual(stats["cost"]["estimated_costs"], {"total_cost": 0.7})

    def test_chain_of_experts_solve_loop_with_mocked_chain(self) -> None:
        coe = import_module("pyopl.genai.pyopl_chain_of_experts")

        with tempfile.TemporaryDirectory() as td:
            model_file = str(Path(td) / "model.mod")
            data_file = str(Path(td) / "data.dat")

            with (
                patch.object(
                    coe,
                    "_run_chain_of_experts",
                    return_value=(
                        "dvar int x;",
                        "x=1;",
                        "chain assessment",
                        [],
                        {"prompt_tokens": 2, "completion_tokens": 3},
                        1,
                    ),
                ),
                patch.object(
                    coe,
                    "_llm_generate_text",
                    return_value=("final chain assessment", {"prompt_tokens": 5, "completion_tokens": 7}),
                ),
                patch.object(coe, "_estimate_costs", return_value={"total_cost": 0.8}),
            ):
                stats = coe.generative_solve(
                    "make a tiny model",
                    model_file,
                    data_file,
                    model_name="gpt-test",
                    mode=coe.Grammar.NONE,
                    iterations=1,
                    return_statistics=True,
                    alignment_check=False,
                    few_shot=False,
                )

            self.assertEqual(Path(model_file).read_text(encoding="utf-8"), "dvar int x;")
            self.assertEqual(stats["assessment"], "final chain assessment")
            self.assertEqual(stats["cost"]["estimated_costs"], {"total_cost": 0.8})

    def test_chain_of_experts_call_json_parses_and_wraps_errors(self) -> None:
        coe = import_module("pyopl.genai.pyopl_chain_of_experts")

        with patch.object(coe, "_llm_generate_text", return_value=(json.dumps({"comment": "ok"}), {"prompt_tokens": 1})):
            obj, usage = coe._call_json(coe.LLMProvider.OPENAI, "gpt-test", "prompt", None)

        self.assertEqual(obj, {"comment": "ok"})
        self.assertEqual(usage, {"prompt_tokens": 1})

        with patch.object(coe, "_llm_generate_text", return_value=("not json", {})):
            with self.assertRaisesRegex(RuntimeError, "Failed to parse LLM JSON response"):
                coe._call_json(coe.LLMProvider.OPENAI, "gpt-test", "prompt", None)

    def test_chain_of_experts_internal_loop_aligned_first_try(self) -> None:
        coe = import_module("pyopl.genai.pyopl_chain_of_experts")
        responses = iter(
            [
                ({"next_expert": "Modeling Expert"}, {"prompt_tokens": 1, "completion_tokens": 1}),
                ({"comment": "Use one integer decision variable."}, {"prompt_tokens": 2, "completion_tokens": 2}),
                ({"model": "dvar int x;", "data": "x=1;"}, {"prompt_tokens": 3, "completion_tokens": 3}),
                ({"aligned": True, "assessment": "aligned"}, {"prompt_tokens": 4, "completion_tokens": 4}),
            ]
        )

        with (
            patch.object(coe, "_call_json", side_effect=lambda *args, **kwargs: next(responses)),
            patch.object(coe.OPLCompiler, "compile_model", return_value=None),
        ):
            model, data, assessment, errors, usage, trials = coe._run_chain_of_experts(
                problem="make a tiny model",
                grammar_implementation="grammar",
                provider=coe.LLMProvider.OPENAI,
                model_name="gpt-test",
                progress=None,
                temperature=None,
                stop=None,
                few_shots=[],
                max_forward_steps=1,
                max_trials=2,
                do_alignment=True,
            )

        self.assertEqual(model, "dvar int x;")
        self.assertEqual(data, "x=1;")
        self.assertEqual(assessment, "aligned")
        self.assertEqual(errors, [])
        self.assertEqual(usage, {"prompt_tokens": 10, "completion_tokens": 10})
        self.assertEqual(trials, 1)

    def test_chain_of_experts_internal_loop_reflects_after_alignment_failure(self) -> None:
        coe = import_module("pyopl.genai.pyopl_chain_of_experts")
        responses = iter(
            [
                ({"next_expert": "Made Up Expert"}, {"prompt_tokens": 1, "completion_tokens": 0}),
                ({"comment": "Initial modeling comment."}, {"prompt_tokens": 1, "completion_tokens": 1}),
                ({"model": "bad model", "data": "bad data"}, {"prompt_tokens": 2, "completion_tokens": 2}),
                ({"aligned": False, "assessment": "missing demand"}, {"prompt_tokens": 3, "completion_tokens": 3}),
                ({"comment": "Add demand balance."}, {"prompt_tokens": 4, "completion_tokens": 4}),
                ({"model": "fixed model", "data": "fixed data"}, {"prompt_tokens": 5, "completion_tokens": 5}),
                ({"aligned": True, "assessment": "fixed"}, {"prompt_tokens": 6, "completion_tokens": 6}),
            ]
        )

        with (
            patch.object(coe, "_call_json", side_effect=lambda *args, **kwargs: next(responses)),
            patch.object(coe.OPLCompiler, "compile_model", return_value=None),
        ):
            model, data, assessment, errors, usage, trials = coe._run_chain_of_experts(
                problem="make a tiny model",
                grammar_implementation="grammar",
                provider=coe.LLMProvider.OPENAI,
                model_name="gpt-test",
                progress=None,
                temperature=None,
                stop=None,
                few_shots=[],
                max_forward_steps=1,
                max_trials=1,
                do_alignment=True,
            )

        self.assertEqual(model, "fixed model")
        self.assertEqual(data, "fixed data")
        self.assertEqual(assessment, "fixed")
        self.assertEqual(errors, [])
        self.assertEqual(usage, {"prompt_tokens": 22, "completion_tokens": 21})
        self.assertEqual(trials, 1)

    def test_chain_of_experts_internal_loop_handles_compile_errors_without_alignment(self) -> None:
        coe = import_module("pyopl.genai.pyopl_chain_of_experts")
        responses = iter(
            [
                ({"next_expert": "Modeling Expert"}, {"prompt_tokens": 1, "completion_tokens": 1}),
                ({"comment": "Initial modeling comment."}, {"prompt_tokens": 1, "completion_tokens": 1}),
                ({"model": "bad model", "data": "bad data"}, {"prompt_tokens": 1, "completion_tokens": 1}),
                ({"comment": "Fix syntax."}, {"prompt_tokens": 1, "completion_tokens": 1}),
                ({"model": "fixed model", "data": "fixed data"}, {"prompt_tokens": 1, "completion_tokens": 1}),
            ]
        )

        compile_calls = {"count": 0}

        def fake_compile(model, data):
            compile_calls["count"] += 1
            if compile_calls["count"] == 1:
                raise RuntimeError("syntax bad")
            return None

        with (
            patch.object(coe, "_call_json", side_effect=lambda *args, **kwargs: next(responses)),
            patch.object(coe.OPLCompiler, "compile_model", side_effect=fake_compile),
        ):
            model, data, assessment, errors, usage, trials = coe._run_chain_of_experts(
                problem="make a tiny model",
                grammar_implementation="grammar",
                provider=coe.LLMProvider.OPENAI,
                model_name="gpt-test",
                progress=None,
                temperature=None,
                stop=None,
                few_shots=[],
                max_forward_steps=1,
                max_trials=1,
                do_alignment=False,
            )

        self.assertEqual(model, "fixed model")
        self.assertEqual(data, "fixed data")
        self.assertEqual(assessment, "")
        self.assertEqual(errors, [])
        self.assertEqual(usage, {"prompt_tokens": 5, "completion_tokens": 5})
        self.assertEqual(trials, 1)

    def test_strategy_feedback_helpers_with_mocked_llm(self) -> None:
        for module_name in STRATEGY_MODULE_NAMES:
            module = import_module(module_name)
            with self.subTest(module=module_name), tempfile.TemporaryDirectory() as td:
                self.assertIs(module.generative_feedback, feedback_validation.generative_feedback)
                model_file = Path(td) / "model.mod"
                data_file = Path(td) / "data.dat"
                model_file.write_text("dvar int x;", encoding="utf-8")
                data_file.write_text("x=1;", encoding="utf-8")

                with patch.object(
                    feedback_validation._BASE,
                    "llm_generate_text",
                    return_value=json.dumps({"feedback": "looks ok"}),
                ):
                    feedback = module.generative_feedback(
                        "review this",
                        str(model_file),
                        str(data_file),
                        model_name="gpt-test",
                        mode=module.Grammar.NONE,
                    )

                self.assertEqual(feedback, {"feedback": "looks ok"})


if __name__ == "__main__":
    unittest.main()
