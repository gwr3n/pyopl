import unittest
from pathlib import Path
from unittest import mock

from pyopl.genai import exemplar_distillation


class TestExemplarDistillation(unittest.TestCase):
    def test_reviewed_content_replaces_only_the_reviewed_fields(self):
        draft = exemplar_distillation.ExemplarDraft(
            name="name",
            model="old model",
            data="old data",
            description="old description",
            source_session="session",
        )

        reviewed = draft.with_reviewed_content(description="new description", model="new model", data="new data")

        self.assertEqual(reviewed.name, "name")
        self.assertEqual(reviewed.source_session, "session")
        self.assertEqual(reviewed.description, "new description")
        self.assertEqual(reviewed.model, "new model")
        self.assertEqual(reviewed.data, "new data")
        self.assertEqual(draft.model, "old model")

    def test_prompt_contains_three_style_examples_and_current_inputs(self):
        examples = [
            {"description": f"description {index}", "model": f"model {index}", "data": f"data {index}"} for index in range(3)
        ]

        prompt = exemplar_distillation.build_distillation_prompt(
            model="current model",
            data="current data",
            source_session="current session",
            examples=examples,
        )

        self.assertEqual(prompt.count('<example index="'), 3)
        self.assertIn("style references only", prompt)
        self.assertIn("<current_model>\ncurrent model", prompt)
        self.assertIn("<current_data>\ncurrent data", prompt)
        self.assertIn("<current_session>\ncurrent session", prompt)

    def test_distillation_retrieves_three_examples_and_normalizes_response(self):
        examples = [{"description": "d", "model": "m", "data": "x"}] * 3
        with (
            mock.patch.object(exemplar_distillation._STRATEGY, "gather_few_shots", return_value=examples) as gather,
            mock.patch.object(
                exemplar_distillation._STRATEGY,
                "llm_generate_text",
                return_value="```text\nDescription: Assign crews while minimizing cost.\n```",
            ) as generate,
        ):
            result = exemplar_distillation.distill_exemplar_description(
                model="model",
                data="data",
                source_session="session",
                provider="openai",
                model_name="test-model",
            )

        self.assertEqual(result, "Assign crews while minimizing cost.")
        self.assertEqual(gather.call_args.kwargs["k"], 3)
        self.assertIn("<current_session>\nsession", generate.call_args.kwargs["input_text"])

    def test_normalize_description_removes_headers_and_truncates(self):
        response = " \n```MARKDOWN\nProblem Description: Assign items to bins.\n``` \n"
        self.assertEqual(exemplar_distillation.normalize_description(response), "Assign items to bins.")
        self.assertEqual(exemplar_distillation.normalize_description("  Answer: Minimize cost.  "), "Minimize cost.")

        truncated = exemplar_distillation.normalize_description("x" * 1199 + " " + "y")
        self.assertEqual(len(truncated), exemplar_distillation.DESCRIPTION_MAX_CHARS - 1)

    def test_distillation_warns_when_examples_are_sparse_and_rejects_empty_response(self):
        progress_messages = []
        with (
            mock.patch.object(exemplar_distillation._STRATEGY, "gather_few_shots", return_value=[]) as gather,
            mock.patch.object(exemplar_distillation._STRATEGY, "llm_generate_text", return_value="  ") as generate,
            self.assertRaisesRegex(RuntimeError, "empty exemplar description"),
        ):
            exemplar_distillation.distill_exemplar_description(
                model="model",
                data="data",
                source_session="session",
                provider=" OPENAI ",
                model_name="test-model",
                models_dir=Path("models"),
                progress=progress_messages.append,
            )

        self.assertEqual(gather.call_args.args[0], "session\n\nmodel\n\ndata")
        self.assertEqual(gather.call_args.kwargs["k"], 3)
        self.assertEqual(gather.call_args.kwargs["models_dir"], Path("models"))
        self.assertTrue(any("using 0 of 3" in message for message in progress_messages))
        self.assertEqual(generate.call_args.kwargs["provider"], exemplar_distillation.LLMProvider.OPENAI)

    def test_sparse_examples_work_without_a_progress_callback(self):
        with (
            mock.patch.object(exemplar_distillation._STRATEGY, "gather_few_shots", return_value=[{}]),
            mock.patch.object(
                exemplar_distillation._STRATEGY, "llm_generate_text", return_value="Description: Minimize cost."
            ),
        ):
            result = exemplar_distillation.distill_exemplar_description(
                model="model",
                data="data",
                source_session="session",
                provider="openai",
                model_name="test-model",
            )

        self.assertEqual(result, "Minimize cost.")

    def test_distillation_rejects_unsupported_provider(self):
        with self.assertRaisesRegex(ValueError, "Unsupported GenAI provider: unknown"):
            exemplar_distillation.distill_exemplar_description(
                model="model",
                data="data",
                source_session="session",
                provider="unknown",
                model_name="test-model",
            )


if __name__ == "__main__":
    unittest.main()
