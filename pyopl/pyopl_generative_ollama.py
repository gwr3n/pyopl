import json
import os
import re
from enum import Enum, auto

from ollama import generate

from .pyopl_core import OPLCompiler, SemanticError

MAX_ITERATIONS = 5
MAX_OUTPUT_TOKENS = 4096 * 2  # used as num_predict for Ollama


class Grammar(Enum):
    NONE = auto()
    BNF = auto()
    CODE = auto()


def _read_pyopl_grammar():
    grammar_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "grammars", "PyOPL grammar.md")
    with open(grammar_path, "r", encoding="utf-8") as f:
        return f.read()


def _read_pyopl_code():
    code_path = os.path.join(os.path.dirname(__file__), "pyopl_core.py")
    with open(code_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_json_from_markdown(text):
    """
    Extract JSON object from a Markdown code block if present.
    """
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text


def _ollama_generate_text(model_name: str, prompt: str, num_predict: int = MAX_OUTPUT_TOKENS) -> str:
    """
    Call Ollama's generate and return the response text.
    """
    resp = generate(model=model_name, prompt=prompt, options={"num_predict": num_predict})
    try:
        return resp["response"]
    except (TypeError, KeyError) as e:
        raise RuntimeError(f"Failed to retrieve response text from Ollama response: {e}")


# https://ollama.com/library/gpt-oss
def generative_solve(
    prompt,
    model_file,
    data_file,
    model_name="gpt-oss:120b",
    mode=Grammar.CODE,
    iterations=MAX_ITERATIONS,
    return_statistics=False,
):
    """
    Generate a PyOPL model and data file from a prompt using Ollama, validate with pyopl,
    iterate on errors, and provide an alignment assessment.

    Args:
        prompt (str): Textual description of the optimization problem.
        model_file (str): Path to save the generated model file.
        data_file (str): Path to save the generated data file.
        model_name (str): Ollama model name.
        mode (Grammar): Grammar mode for generation (NONE, BNF, CODE).
        iterations (int): Maximum number of refinement iterations.
        return_statistics (bool): If True, return a dict with stats and assessment.
    Returns:
        str | dict: Assessment text or dict with iterations, assessment, and syntax_errors.
    """
    if mode == Grammar.NONE:
        grammar_implementation = ""
    elif mode == Grammar.BNF:
        grammar_implementation = _read_pyopl_grammar()
    elif mode == Grammar.CODE:
        grammar_implementation = _read_pyopl_code()
    else:
        raise ValueError(f"Invalid mode: {mode}")

    try:
        iterations = max(1, int(iterations))
    except Exception:
        iterations = MAX_ITERATIONS

    user_prompt = (
        "<role>\n"
        "You are an expert in mathematical optimization and PyOPL.\n"
        "</role>\n\n"
        "<task>\n"
        "Generate a valid PyOPL model (.mod) and a matching data file (.dat) for the given problem.\n"
        "If data are missing, create a small, plausible mock instance consistent with the model.\n"
        "Validate all syntax against the provided PyOPL implementation reference only.\n"
        "</task>\n\n"
        "<grammar_reference>\n"
        "--- BEGIN REFERENCE ---\n"
        f"{grammar_implementation}\n"
        "--- END REFERENCE ---\n"
        "</grammar_reference>\n\n"
        "<problem_prompt>\n"
        f"{prompt}\n"
        "</problem_prompt>\n\n"
        "<output_requirements>\n"
        '- Return ONLY a JSON object with exactly two keys: "model" (the PyOPL model) and "data" (the matching data file).\n'
        "- The values must be single JSON strings (no arrays/objects inside them).\n"
        "- Escape all double quotes and backslashes; encode newlines as \\n.\n"
        "- No trailing commas. No additional keys. No commentary.\n"
        "- Optional: you MAY wrap the JSON in a ```json fenced block; if you do, the fence must contain only the JSON.\n"
        "</output_requirements>\n\n"
        "<json_schema>\n"
        "{\n"
        '  "type": "object",\n'
        '  "additionalProperties": false,\n'
        '  "required": ["model", "data"],\n'
        '  "properties": {\n'
        '    "model": {"type": "string"},\n'
        '    "data":  {"type": "string"}\n'
        "  }\n"
        "}\n"
        "</json_schema>\n\n"
        "<example_output>\n"
        "{\n"
        '  "model": "float a;\\nfloat b;\\ndvar float x;\\nminimize z: a*x;\\nsubject to { b*x >= 0; }",'
        '  "data":  "a = 10;\\nb= 5;"\n'
        "}\n"
        "</example_output>\n"
    )

    for iteration in range(iterations):
        print(f"Iteration {iteration + 1}/{iterations}")
        content = _ollama_generate_text(model_name, user_prompt, num_predict=MAX_OUTPUT_TOKENS)
        if not content:
            raise RuntimeError("Empty model response from Ollama.")
        try:
            result = json.loads(extract_json_from_markdown(content))
            model_code = result["model"]
            data_code = result["data"]
        except Exception as e:
            raise RuntimeError(f"Failed to parse Ollama response as JSON: {e}\nResponse: {content}")

        compiler = OPLCompiler()
        syntax_errors = []
        try:
            compiler.compile_model(model_code, data_code)
        except SemanticError as e:
            syntax_errors.append(str(e))
            print(f"Semantic error in model: {e}")
        except Exception as e:
            syntax_errors.append(f"Unexpected error: {e}")

        # Ensure output folders exist
        model_dir = os.path.dirname(model_file)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        data_dir = os.path.dirname(data_file)
        if data_dir:
            os.makedirs(data_dir, exist_ok=True)

        # Write files
        with open(model_file, "w", encoding="utf-8") as f:
            f.write(model_code)
        with open(data_file, "w", encoding="utf-8") as f:
            f.write(data_code)

        if not syntax_errors:
            break
        else:
            # Feedback errors and retry
            user_prompt = (
                "<role>\n"
                "You are an expert in mathematical optimization and PyOPL.\n"
                "</role>\n\n"
                "<task>\n"
                "The previous attempt to generate a PyOPL model and data file failed due to syntax errors.\n"
                "Revise the model and data to fix the errors while retaining alignment with the original intent.\n"
                "Validate all syntax against the provided PyOPL implementation reference only.\n"
                "Change only what is necessary to fix the errors.\n"
                "</task>\n\n"
                "<grammar_reference>\n"
                "--- BEGIN REFERENCE ---\n"
                f"{grammar_implementation}\n"
                "--- END REFERENCE ---\n"
                "</grammar_reference>\n\n"
                "<problem_prompt>\n"
                f"{prompt}\n"
                "</problem_prompt>\n\n"
                "<previous_attempt>\n"
                "<model>\n"
                f"{model_code}\n"
                "</model>\n\n"
                "<data>\n"
                f"{data_code}\n"
                "</data>\n"
                "</previous_attempt>\n\n"
                "<errors>\n"
                f"{syntax_errors}\n"
                "</errors>\n\n"
                "<revision_guidelines>\n"
                "- Fix the listed syntax/semantic errors.\n"
                "- Preserve the original modeling intent and structure when possible.\n"
                "- Ensure the model compiles with the data under the given implementation.\n"
                "- Return complete model and data strings; do not return diffs.\n"
                "</revision_guidelines>\n\n"
                "<output_requirements>\n"
                '- Return ONLY a JSON object with exactly two keys: "model" (the PyOPL model) and "data" (the matching data file).\n'
                "- The values must be single JSON strings (no arrays/objects inside them).\n"
                "- Escape all double quotes and backslashes; encode newlines as \\n.\n"
                "- No trailing commas. No additional keys. No commentary.\n"
                "- Optional: you MAY wrap the JSON in a ```json fenced block; if you do, the fence must contain only the JSON.\n"
                "</output_requirements>\n\n"
                "<json_schema>\n"
                "{\n"
                '  "type": "object",\n'
                '  "additionalProperties": false,\n'
                '  "required": ["model", "data"],\n'
                '  "properties": {\n'
                '    "model": {"type": "string"},\n'
                '    "data":  {"type": "string"}\n'
                "  }\n"
                "}\n"
                "</json_schema>\n\n"
                "<example_output>\n"
                "{\n"
                '  "model": "float a;\\nfloat b;\\ndvar float x;\\nminimize z: a*x;\\nsubject to { b*x >= 0; }",'
                '  "data":  "a = 10;\\nb= 5;"\n'
                "}\n"
                "</example_output>\n"
            )

    # Load latest version of the model and data files (ensure we assess what's written)
    with open(model_file, "r", encoding="utf-8") as f:
        model_code = f.read()
    with open(data_file, "r", encoding="utf-8") as f:
        data_code = f.read()

    # Final assessment
    syntax_errors_str = f"SYNTAX ERRORS:\n{syntax_errors}\n\n" if syntax_errors else ""
    assessment_prompt = (
        "<role>\n"
        "You are an expert in mathematical optimization and PyOPL.\n"
        "</role>\n\n"
        "<task>\n"
        "Assess how well the generated PyOPL model and data align with the original problem intent.\n"
        "Be critical and specific about modeling choices, feasibility, and consistency.\n"
        "Reference only the provided PyOPL implementation for syntax validity.\n"
        "</task>\n\n"
        "<grammar_reference>\n"
        "--- BEGIN REFERENCE ---\n"
        f"{grammar_implementation}\n"
        "--- END REFERENCE ---\n"
        "</grammar_reference>\n\n"
        "<inputs>\n"
        "<problem_prompt>\n"
        f"{prompt}\n"
        "</problem_prompt>\n\n"
        "<model>\n"
        f"{model_code}\n"
        "</model>\n\n"
        "<data>\n"
        f"{data_code}\n"
        "</data>\n\n"
        f"{syntax_errors_str}"
        "</inputs>\n\n"
        "<assessment_focus>\n"
        "- Objective and constraints reflect the prompt intent.\n"
        "- Decision variables have correct domains and indices.\n"
        "- Data is consistent with sets/parameters used by the model.\n"
        "- Signs, units, and indexing are correct; no missing links.\n"
        "- Any syntax/semantic issues relative to the implementation reference.\n"
        "- Most impactful improvements if misaligned.\n"
        "</assessment_focus>\n\n"
        "<output_requirements>\n"
        "- Return a single short paragraph (3–6 sentences) of plain text.\n"
        "- No Markdown, no bullet lists, no code fences.\n"
        "- If issues exist, mention the most critical fixes.\n"
        "</output_requirements>\n"
    )
    assessment_text = _ollama_generate_text(model_name, assessment_prompt, num_predict=MAX_OUTPUT_TOKENS).strip()
    if not assessment_text:
        raise RuntimeError("Empty assessment response from Ollama.")

    if return_statistics:
        return {
            "iterations": iteration + 1,
            "assessment": assessment_text,
            "syntax_errors": syntax_errors,
        }
    else:
        return assessment_text


# https://ollama.com/library/gpt-oss
def generative_feedback(prompt, model_file, data_file, model_name="gpt-oss:120b", mode=Grammar.CODE):
    """
    Ask questions or request revisions about a given PyOPL model and data using Ollama.
    Returns a JSON object with:
      - 'feedback' (str, mandatory)
      - 'revised_model' (str, optional)
      - 'revised_data' (str, optional)
    """
    if mode == Grammar.NONE:
        grammar_implementation = ""
    elif mode == Grammar.BNF:
        grammar_implementation = _read_pyopl_grammar()
    elif mode == Grammar.CODE:
        grammar_implementation = _read_pyopl_code()
    else:
        raise ValueError(f"Invalid mode: {mode}")

    with open(model_file, "r", encoding="utf-8") as fh:
        model_code = fh.read()
    with open(data_file, "r", encoding="utf-8") as fh:
        data_code = fh.read()

    user_prompt = (
        "<role>\n"
        "You are an expert in mathematical optimization and PyOPL.\n"
        "</role>\n\n"
        "<task>\n"
        "Answer the user's question about the provided PyOPL model and data.\n"
        "Provide critical, specific feedback. If revisions are necessary for correctness,\n"
        "semantics, or consistency with the grammar reference, propose minimal changes.\n"
        "Only change what is necessary.\n"
        "</task>\n\n"
        "<grammar_reference>\n"
        "--- BEGIN REFERENCE ---\n"
        f"{grammar_implementation}\n"
        "--- END REFERENCE ---\n"
        "</grammar_reference>\n\n"
        "<inputs>\n"
        "<prompt>\n"
        f"{prompt}\n"
        "</prompt>\n\n"
        "<model>\n"
        f"{model_code}\n"
        "</model>\n\n"
        "<data>\n"
        f"{data_code}\n"
        "</data>\n"
        "</inputs>\n\n"
        "<output_requirements>\n"
        "- Return ONLY a JSON object with 1 required key and up to 2 optional keys:\n"
        '  "feedback" (required), "revised_model" (optional), "revised_data" (optional).\n'
        "- Each value must be a single JSON string. Escape all double quotes and backslashes;\n"
        "  encode newlines as \\n.\n"
        '- If no changes are needed, omit "revised_model" and "revised_data".\n'
        "- If changes are needed, return complete model and data strings; do not return diffs.\n"
        "- No trailing commas. No additional keys. No commentary.\n"
        "- Optional: you MAY wrap the JSON in a ```json fenced block; if you do, the fence must contain only the JSON.\n"
        "</output_requirements>\n\n"
        "<json_schema>\n"
        "{\n"
        '  "type": "object",\n'
        '  "additionalProperties": false,\n'
        '  "required": ["feedback"],\n'
        '  "properties": {\n'
        '    "feedback": {"type": "string"},\n'
        '    "revised_model": {"type": "string"},\n'
        '    "revised_data": {"type": "string"}\n'
        "  }\n"
        "}\n"
        "</json_schema>\n\n"
        "<example_output>\n"
        "{\n"
        '  "feedback": "The model was missing coefficients a and b.",\n'
        '  "revised_model": "// minimal fix\\nfloat a;\\nfloat b;\\ndvar float x;\\nminimize z: a*x;\\nsubject to { b*x >= 0; }",'
        '  "revised_data":  "a = 10;\\nb= 5;"\n'
        "}\n"
        "</example_output>\n"
    )

    content = _ollama_generate_text(model_name, user_prompt, num_predict=MAX_OUTPUT_TOKENS)
    if not content:
        raise RuntimeError("Empty model response from Ollama.")
    try:
        result = json.loads(extract_json_from_markdown(content))
        return result
    except Exception as e:
        raise RuntimeError(f"Failed to parse Ollama response as JSON: {e}\nResponse: {content}")
