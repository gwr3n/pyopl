import json
import os
import re
from enum import Enum, auto

# import google.generativeai as genai
from openai import OpenAI

from .pyopl_core import OPLCompiler, SemanticError

MAX_ITERATIONS = 5
MAX_OUTPUT_TOKENS = 4096 * 2
REASONING_EFFORT = "medium"  # "low", "medium", "high"
ALIGNMENT_CHECK = True  # Whether to check alignment with original prompt


class Grammar(Enum):
    NONE = auto()
    BNF = auto()
    CODE = auto()


def _read_pyopl_grammar():
    grammar_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "grammars", "PyOPL grammar.md")
    with open(grammar_path, "r") as f:
        return f.read()


def _read_pyopl_code():
    code_path = os.path.join(os.path.dirname(__file__), "pyopl_core.py")
    with open(code_path, "r") as f:
        return f.read()


def extract_json_from_markdown(text):
    """
    Extract JSON object from a Markdown code block if present.
    """
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text


def _coalesce_response_text(resp) -> str:
    # Prefer SDK convenience if present
    if getattr(resp, "output_text", None):
        return resp.output_text

    # Structured fallback: response.output -> items -> item.content -> blocks
    try:
        chunks = []
        for item in getattr(resp, "output", []) or []:
            # item may have .content (list of blocks) or be text-like itself
            content_blocks = getattr(item, "content", None)
            if content_blocks is None:
                # Try common attributes
                if hasattr(item, "text"):
                    chunks.append(getattr(item, "text") or "")
                continue
            for block in content_blocks:
                # Blocks often have .text; be permissive
                if hasattr(block, "text"):
                    chunks.append(getattr(block, "text") or "")
                elif isinstance(block, dict):
                    # Some SDKs use dict-like blocks
                    if "text" in block and isinstance(block["text"], str):
                        chunks.append(block["text"])
        return "".join(chunks)
    except Exception:
        pass
    # Last-resort fallbacks
    try:
        first = getattr(resp, "output", [])[0]
        first_content = getattr(first, "content", [])[0]
        if hasattr(first_content, "text"):
            return first_content.text or ""
    except Exception:
        pass
    return ""


# Use GPT-5 model
# model_name = "gpt-5"
# model_name = "gpt-5-mini"
# model_name = "gpt-5-nano"
# model_name = "gpt-4.1"
def generative_solve(
    prompt, model_file, data_file, model_name="gpt-5", mode=Grammar.CODE, iterations=MAX_ITERATIONS, return_statistics=False
):
    """
    Generate a PyOPL model and data file from a prompt using OpenAI GPT-5, validate with pyopl, iterate on errors, and assess alignment.
    Args:
            prompt (str): Textual description of the optimization problem.
            model_file (str): Path to save the generated model file.
            data_file (str): Path to save the generated data file.
            model_name (str): OpenAI GPT model to use (e.g., "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4.1").
            mode (Grammar): Grammar mode for generation (NONE, BNF, CODE).
            iterations (int): Maximum number of iterations to attempt generation and correction.
            return_statistics (bool): If True, return detailed statistics including assessment and syntax errors.
    Returns:
            str: GPT-5's assessment of alignment between model/data and prompt.
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

    # Use API key from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")
    client = OpenAI(api_key=api_key)

    user_prompt = (
        "<role>\n"
        "You are an expert in mathematical optimization and PyOPL.\n"
        "</role>\n\n"
        "<task>\n"
        "Generate a valid PyOPL model (.mod) and a matching data file (.dat) for the given problem description.\n"
        "Ensure the model decision variables, objective function, and constraints fully align with the provided problem description.\n"
        "If data are missing, create a small, plausible mock instance consistent with the model.\n"
        "Validate all syntax against the provided PyOPL grammar implementation reference only.\n"
        "</task>\n\n"
        "<grammar_reference>\n"
        "--- BEGIN REFERENCE ---\n"
        f"{grammar_implementation}\n"
        "--- END REFERENCE ---\n"
        "</grammar_reference>\n\n"
        "<problem_description>\n"
        f"{prompt}\n"
        "</problem_description>\n\n"
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
        print("Prompting model...")
        create_params = {
            "model": model_name,
            "input": user_prompt,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        }
        if "gpt-5" in model_name:
            create_params["reasoning"] = {"effort": REASONING_EFFORT}
        response = client.responses.create(**create_params)
        content = _coalesce_response_text(response)
        if not content:
            raise RuntimeError(f"Empty model response. Full response: {response}")
        try:
            result = json.loads(extract_json_from_markdown(content))
            model_code = result["model"]
            data_code = result["data"]
            print("Model and data generated.")
        except Exception as e:
            raise RuntimeError(f"Failed to parse GPT-5 response as JSON: {e}\nResponse: {content}")

        compiler = OPLCompiler()
        syntax_errors = []
        # Validate model
        try:
            compiler.compile_model(model_code, data_code)
        except SemanticError as e:
            syntax_errors.append(str(e))
            print(f"Semantic error in model: {e}")
        except Exception as e:
            syntax_errors.append(f"Unexpected error: {e}")

        # Ensure output folder exists
        model_dir = os.path.dirname(model_file)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        data_dir = os.path.dirname(data_file)
        if data_dir:
            os.makedirs(data_dir, exist_ok=True)
        # Write files
        with open(model_file, "w") as f:
            f.write(model_code)
        with open(data_file, "w") as f:
            f.write(data_code)

        if not syntax_errors:
            if not ALIGNMENT_CHECK:
                break  # Success: the model is syntactically correct; exit loop

            # Alignment check with original intent
            alignment_prompt = (
                "<role>\n"
                "You are an expert in mathematical optimization and PyOPL.\n"
                "</role>\n\n"
                "<task>\n"
                "Assess whether the generated PyOPL model and data fully align with the original problem description.\n"
                "Alignment means the objective, constraints, decision variables, and data fully capture the user's specifications.\n"
                "Use the provided PyOPL grammar implementation to support your analysis.\n"
                "</task>\n\n"
                "<grammar_reference>\n"
                "--- BEGIN REFERENCE ---\n"
                f"{grammar_implementation}\n"
                "--- END REFERENCE ---\n"
                "</grammar_reference>\n\n"
                "<inputs>\n"
                "<problem_description>\n"
                f"{prompt}\n"
                "</problem_description>\n\n"
                "<model>\n"
                f"{model_code}\n"
                "</model>\n\n"
                "<data>\n"
                f"{data_code}\n"
                "</data>\n"
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
                '- Return ONLY a JSON object with exactly two keys: "aligned" (boolean) and "assessment" (string).\n'
                '- If issues exist, mention the most critical fixes in "assessment", a single short paragraph (3–6 sentences) of plain text.\n'
                "- No Markdown. No bullet lists. No commentary. No additional keys. No trailing commas.\n"
                "- Optional: you MAY wrap the JSON in a ```json fenced block; if you do, the fence must contain only the JSON.\n"
                "</output_requirements>\n\n"
                "<json_schema>\n"
                "{\n"
                '  "type": "object",\n'
                '  "additionalProperties": false,\n'
                '  "required": ["aligned", "assessment"],\n'
                '  "properties": {\n'
                '    "aligned": {"type": "boolean"},\n'
                '    "assessment": {"type": "string"}\n'
                "  }\n"
                "}\n"
                "</json_schema>\n\n"
                "<example_output>\n"
                '{ "aligned": false, "assessment": "The model objective function does not include fixed costs." }\n'
                "</example_output>\n"
            )

            print("Checking alignment with original prompt...")

            create_params = {
                "model": model_name,
                "input": alignment_prompt,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
            }
            if "gpt-5" in model_name:
                create_params["reasoning"] = {"effort": REASONING_EFFORT}
            alignment_response = client.responses.create(**create_params)
            alignment_content = _coalesce_response_text(alignment_response)
            if not alignment_content:
                raise RuntimeError(f"Empty alignment response. Full response: {alignment_response}")
            try:
                alignment_obj = json.loads(extract_json_from_markdown(alignment_content))
            except Exception as e:
                raise RuntimeError(f"Failed to parse alignment response JSON: {e}\nResponse: {alignment_content}")
            if (
                isinstance(alignment_obj, dict)
                and isinstance(alignment_obj.get("aligned"), bool)
                and isinstance(alignment_obj.get("assessment"), str)
            ):
                if alignment_obj["aligned"]:
                    print("Model and data are syntactically valid and aligned with the prompt.")
                    break
                else:
                    assessment_text = alignment_obj.get("assessment", "").strip()
                    print(
                        f"Model and data are syntactically valid but NOT aligned with the prompt. Assessment: {assessment_text}"
                    )
                    # Not aligned; continue iterating for potential revisions
                    user_prompt = (
                        "<role>\n"
                        "You are an expert in mathematical optimization and PyOPL.\n"
                        "</role>\n\n"
                        "<task>\n"
                        "The previous attempt produced a syntactically valid PyOPL model and data, but they are NOT fully aligned with the problem description.\n"
                        "<assessment>\n"
                        f"{assessment_text}\n"
                        "</assessment>\n"
                        "Revise the model and data so that they fully align with the user's specifications while preserving syntactic validity under the provided PyOPL grammar implementation reference.\n"
                        "Change only what is necessary to achieve alignment (objective, constraints, variables, sets/parameters, and data consistency).\n"
                        "</task>\n\n"
                        "<grammar_reference>\n"
                        "--- BEGIN REFERENCE ---\n"
                        f"{grammar_implementation}\n"
                        "--- END REFERENCE ---\n"
                        "</grammar_reference>\n\n"
                        "<problem_description>\n"
                        f"{prompt}\n"
                        "</problem_description>\n\n"
                        "<previous_attempt>\n"
                        "<model>\n"
                        f"{model_code}\n"
                        "</model>\n\n"
                        "<data>\n"
                        f"{data_code}\n"
                        "</data>\n"
                        "</previous_attempt>\n\n"
                        "<revision_guidelines>\n"
                        "- Ensure the objective, constraints, indices, and variable domains reflect the problem description.\n"
                        "- Make the minimal set of changes necessary to correct misalignment.\n"
                        "- Keep syntax strictly valid per the provided implementation reference.\n"
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
            else:
                raise RuntimeError(f"Invalid alignment response JSON: {alignment_content}")
        else:
            print("Model or data has syntax errors; revising...")
            # Feedback errors to GPT-5 and retry
            user_prompt = (
                "<role>\n"
                "You are an expert in mathematical optimization and PyOPL.\n"
                "</role>\n\n"
                "<task>\n"
                "The previous attempt to generate a PyOPL model and data file failed due to syntax errors.\n"
                "Revise the model and data to fix the errors while retaining alignment with the original intent.\n"
                "Validate all syntax against the provided PyOPL grammar implementation reference only.\n"
                "Change only what is necessary to fix the errors.\n"
                "</task>\n\n"
                "<grammar_reference>\n"
                "--- BEGIN REFERENCE ---\n"
                f"{grammar_implementation}\n"
                "--- END REFERENCE ---\n"
                "</grammar_reference>\n\n"
                "<problem_description>\n"
                f"{prompt}\n"
                "</problem_description>\n\n"
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

    # Load latest version of the model and data files
    with open(model_file, "r") as f:
        model_code = f.read()
    with open(data_file, "r") as f:
        data_code = f.read()

    # Final assessment prompt
    syntax_errors_str = f"SYNTAX ERRORS:\n{syntax_errors}\n\n" if syntax_errors else ""
    assessment_prompt = (
        "<role>\n"
        "You are an expert in mathematical optimization and PyOPL.\n"
        "</role>\n\n"
        "<task>\n"
        "Assess how well the generated PyOPL model and data align with the original problem description.\n"
        "Be critical and specific about modeling choices, feasibility, and consistency.\n"
        "Reference only the provided PyOPL grammar implementation for syntax validity.\n"
        "</task>\n\n"
        "<grammar_reference>\n"
        "--- BEGIN REFERENCE ---\n"
        f"{grammar_implementation}\n"
        "--- END REFERENCE ---\n"
        "</grammar_reference>\n\n"
        "<inputs>\n"
        "<problem_description>\n"
        f"{prompt}\n"
        "</problem_description>\n\n"
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
    print("Final assessment of model and data alignment...")
    create_params = {
        "model": model_name,
        "input": assessment_prompt,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }
    if "gpt-5" in model_name:
        create_params["reasoning"] = {"effort": REASONING_EFFORT}
    assessment_response = client.responses.create(**create_params)
    assessment_text = _coalesce_response_text(assessment_response)
    if not assessment_text:
        raise RuntimeError(f"Empty assessment response. Full response: {assessment_response}")
    if return_statistics:
        return {
            "iterations": iteration + 1,
            "assessment": assessment_text.strip(),
            "syntax_errors": syntax_errors,
        }
    else:
        return assessment_text.strip()


# Use GPT-5 model
# model_name = "gpt-5"
# model_name = "gpt-5-mini"
# model_name = "gpt-5-nano"
# model_name = "gpt-4.1"
def generative_feedback(prompt, model_file, data_file, model_name="gpt-5", mode=Grammar.CODE):
    # Use API key from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")
    client = OpenAI(api_key=api_key)

    if mode == Grammar.NONE:
        grammar_implementation = ""
    elif mode == Grammar.BNF:
        grammar_implementation = _read_pyopl_grammar()
    elif mode == Grammar.CODE:
        grammar_implementation = _read_pyopl_code()
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Read files first (avoid inline open().read())
    with open(model_file, "r") as fh:
        model_code = fh.read()
    with open(data_file, "r") as fh:
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

    create_params = {
        "model": model_name,
        "input": user_prompt,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }
    if "gpt-5" in model_name:
        create_params["reasoning"] = {"effort": REASONING_EFFORT}
    response = client.responses.create(**create_params)
    content = _coalesce_response_text(response)
    if not content:
        raise RuntimeError(f"Empty model response. Full response: {response}")
    try:
        result = json.loads(extract_json_from_markdown(content))
        return result
    except Exception as e:
        raise RuntimeError(f"Failed to parse GPT-5 response as JSON: {e}\nResponse: {content}")
