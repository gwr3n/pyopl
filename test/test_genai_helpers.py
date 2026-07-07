# mypy: disable-error-code="attr-defined,name-defined,call-arg"
import asyncio
import json
import sys
import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, cast
from unittest.mock import patch

import pyopl.genai._strategy_base as strategy_base
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


class TestGenAIStrategyBaseHelpers(unittest.TestCase):
    def test_extract_json_from_fence_and_prose(self) -> None:
        fenced = 'prefix```json\n{"model": {"name": "x"}, "ok": true}\n```suffix'
        prose = 'Here is the answer: {"outer": {"inner": 3}, "items": [1, 2]}. Done.'

        self.assertEqual(GenAIStrategyBase.extract_json_from_markdown(fenced), '{"model": {"name": "x"}, "ok": true}')
        self.assertEqual(GenAIStrategyBase.json_loads_relaxed(prose), {"outer": {"inner": 3}, "items": [1, 2]})

    def test_normalize_prompt_input_accepts_text_and_images(self) -> None:
        text, images = GenAIStrategyBase.normalize_prompt_input(
            {
                "text": "build a model",
                "images": [
                    Path("diagram.png"),
                    {"url": "https://example.test/img.png", "mime_type": "image/png", "extra": "ignored"},
                ],
            }
        )

        self.assertEqual(text, "build a model")
        self.assertEqual(images, [{"path": "diagram.png"}, {"url": "https://example.test/img.png", "mime_type": "image/png"}])

    def test_get_grammar_implementation_rejects_invalid_mode(self) -> None:
        self.assertEqual(GenAIStrategyBase.get_grammar_implementation(Grammar.NONE), "")
        with self.assertRaisesRegex(ValueError, "Invalid mode"):
            GenAIStrategyBase.get_grammar_implementation(object())  # type: ignore[arg-type]

    def test_find_pair_in_folder_prefers_matching_stems_then_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            desc = folder / "problem.txt"
            desc.write_text("description", encoding="utf-8")
            (folder / "fallback.mod").write_text("model", encoding="utf-8")
            (folder / "fallback.dat").write_text("data", encoding="utf-8")

            mod, dat = GenAIStrategyBase.find_pair_in_folder(desc)

            self.assertEqual(mod, folder / "fallback.mod")
            self.assertEqual(dat, folder / "fallback.dat")

    def test_usage_and_response_helpers_cover_fallback_shapes(self) -> None:
        usage = Usage()
        usage.add({"prompt_tokens": 4, "completion_tokens": 9})
        response = SimpleNamespace(output=[SimpleNamespace(text="loose"), SimpleNamespace(content=[{"text": " dict"}])])

        self.assertEqual(usage.as_dict(), {"prompt_tokens": 4, "completion_tokens": 9})
        self.assertEqual(GenAIStrategyBase._coalesce_response_text(response), "loose dict")

    def test_base_gather_and_render_few_shots(self) -> None:
        base = GenAIStrategyBase(logger=genai_pricing.logger, few_shot_max_chars=100)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            desc = root / "sample.txt"
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


class TestRagHelper(unittest.TestCase):
    def test_iter_description_files_and_read_text(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            nested = root / "nested"
            nested.mkdir()
            first = root / "a.txt"
            second = nested / "b.txt"
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
            (root / "empty.txt").write_text("   ", encoding="utf-8")
            self.assertEqual(rag_helper.rank_problem_descriptions("query", root), [])

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
            low = root / "low.txt"
            high = root / "high.txt"
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

    def test_strategy_openai_create_params_delegate_expected_json_flags(self) -> None:
        expected_json_by_module = {
            "pyopl.genai.pyopl_standard": False,
            "pyopl.genai.pyopl_tree_of_thoughts": False,
        }
        for module_name in STRATEGY_MODULE_NAMES:
            module = import_module(module_name)
            with self.subTest(module=module_name):
                with patch.object(module._BASE, "_build_openai_create_params", return_value={"ok": True}) as build:
                    self.assertEqual(module._build_create_params("gpt-test", "prompt", max_tokens=7), {"ok": True})
                self.assertEqual(build.call_args.kwargs["model_name"], "gpt-test")
                self.assertEqual(build.call_args.kwargs["input_text"], "prompt")
                self.assertEqual(build.call_args.kwargs["expected_json"], expected_json_by_module.get(module_name, True))

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
        few_shots = [{"description": "sample", "model": "dvar int x;", "data": "x=1;", "desc_path": "a.txt"}]

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

    def test_strategy_file_and_pair_wrappers_delegate_to_base_helpers(self) -> None:
        for module_name in STRATEGY_MODULE_NAMES:
            module = import_module(module_name)
            with self.subTest(module=module_name), tempfile.TemporaryDirectory() as td:
                folder = Path(td)
                text_path = folder / "note.txt"
                desc_path = folder / "problem.txt"
                mod_path = folder / "problem.mod"
                dat_path = folder / "problem.dat"
                text_path.write_text("abcdef", encoding="utf-8")
                desc_path.write_text("description", encoding="utf-8")
                mod_path.write_text("model", encoding="utf-8")
                dat_path.write_text("data", encoding="utf-8")

                self.assertEqual(module._read_file(str(text_path)), "abcdef")
                self.assertEqual(module._safe_read_text(text_path, max_chars=3), "abc")
                self.assertEqual(module._find_pair_in_folder(desc_path), (mod_path, dat_path))
                self.assertEqual(module._get_grammar_implementation(module.Grammar.NONE), "")

    def test_strategy_notify_uses_progress_and_swallows_callback_errors(self) -> None:
        for module_name in STRATEGY_MODULE_NAMES:
            module = import_module(module_name)
            messages: list[str] = []
            with self.subTest(module=module_name):
                module._notify(messages.append, "hello")
                module._notify(lambda msg: (_ for _ in ()).throw(RuntimeError("boom")), "ignored")

            self.assertEqual(messages, ["hello"])

    def test_strategy_coalesce_and_ollama_wrappers_delegate(self) -> None:
        enforce_json_by_module = {
            "pyopl.genai.pyopl_standard": False,
            "pyopl.genai.pyopl_tree_of_thoughts": False,
            "pyopl.genai.pyopl_cafa": False,
        }

        for module_name in STRATEGY_MODULE_NAMES:
            module = import_module(module_name)
            response = SimpleNamespace(output=[SimpleNamespace(content=[SimpleNamespace(text="part1"), {"text": "part2"}])])
            with self.subTest(module=module_name):
                self.assertEqual(module._coalesce_response_text(response), "part1part2")

                with patch.object(module._BASE, "_ollama_generate_text", return_value=("{}", {"prompt_tokens": 1})) as ollama:
                    self.assertEqual(
                        module._ollama_generate_text("llama", "prompt", num_predict=3, return_usage=True),
                        ("{}", {"prompt_tokens": 1}),
                    )

                self.assertEqual(ollama.call_args.kwargs["model_name"], "llama")
                self.assertEqual(ollama.call_args.kwargs["prompt"], "prompt")
                self.assertEqual(ollama.call_args.kwargs["num_predict"], 3)
                self.assertEqual(ollama.call_args.kwargs["enforce_json"], enforce_json_by_module.get(module_name, True))

    def test_common_assessment_and_feedback_prompts_include_inputs(self) -> None:
        for module_name in STRATEGY_MODULE_NAMES:
            module = import_module(module_name)
            with self.subTest(module=module_name):
                alignment = module._build_alignment_prompt("user problem", "grammar ref", "model code", "data code")
                final = module._build_final_assessment_prompt(
                    "user problem", "grammar ref", "model code", "data code", "syntax issue"
                )
                feedback = module._build_feedback_prompt("why infeasible?", "grammar ref", "model code", "data code")

                self.assertIn("user problem", alignment)
                self.assertIn("model code", final)
                self.assertIn("SYNTAX ERRORS", final)
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
                model_file = Path(td) / "model.mod"
                data_file = Path(td) / "data.dat"
                model_file.write_text("dvar int x;", encoding="utf-8")
                data_file.write_text("x=1;", encoding="utf-8")

                with patch.object(module, "_llm_generate_text", return_value=json.dumps({"feedback": "looks ok"})):
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
