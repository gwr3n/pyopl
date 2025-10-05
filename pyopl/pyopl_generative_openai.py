import json
import os
import re

# import google.generativeai as genai
from openai import OpenAI

from .pyopl_core import OPLCompiler, SemanticError

MAX_ITERATIONS = 5
MAX_OUTPUT_TOKENS = 4096 * 2


def _read_pyopl_grammar():
    grammar_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "PyOPL grammar.md")
    print(grammar_path)
    with open(grammar_path, "r") as f:
        return f.read()


def _read_pyopl_code():
    code_path = os.path.join(os.path.dirname(__file__), "pyopl_core.py")
    print(code_path)
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
        return ""

# Use GPT-5 model
# model_name = "gpt-5"
# model_name = "gpt-5-mini"
# model_name = "gpt-5-nano"
# model_name = "gpt-4.1"
def generative_solve(prompt, model_file, data_file, model_name = "gpt-5", iterations=MAX_ITERATIONS, return_statistics=False):
    """
    Generate a PyOPL model and data file from a prompt using OpenAI GPT-5, validate with pyopl, iterate on errors, and assess alignment.
    Args:
            prompt (str): Textual description of the optimization problem.
            model_file (str): Path to save the generated model file.
            data_file (str): Path to save the generated data file.
    Returns:
            str: GPT-5's assessment of alignment between model/data and prompt.
    """

    # grammar_implementation = _read_pyopl_grammar()
    grammar_implementation = _read_pyopl_code()
    # grammar_implementation = ""
    
    # Use API key from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")
    client = OpenAI(api_key=api_key)

    user_prompt = (
        "You are an expert in mathematical optimization and PyOPL. "
        "Given the following prompt, generate a PyOPL model (.mod) and a matching data file (.dat). "
        "If the prompt does not specify data, create a plausible mock instance. "
        "Use the following PyOPL syntax implementation as a reference for valid PyOPL syntax:\n\n"
        "--- PyOPL syntax implementation ---\n"
        f"{grammar_implementation}\n"
        "--- END OF PyOPL syntax implementation ---\n\n"
        f"PROMPT:\n{prompt}\n"
        "Return ONLY a strict JSON object with two fields: "
        "'model' (the PyOPL model as a single JSON string) and 'data' (the data file as a single JSON string). "
        "Do not include Markdown or code fences. Escape all double quotes and backslashes inside the strings."
    )

    for iteration in range(iterations):
        print(f"Iteration {iteration + 1}/{iterations}")
        response = client.responses.create(model=model_name, input=user_prompt, max_output_tokens=MAX_OUTPUT_TOKENS)
        content = _coalesce_response_text(response)
        if not content:
            raise RuntimeError(f"Empty model response. Full response: {response}")
        try:
            result = json.loads(extract_json_from_markdown(content))
            model_code = result["model"]
            data_code = result["data"]
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
            break
        else:
            # Feedback errors to GPT-5 and retry
            user_prompt = (
                "The following attempt to generate a PyOPL model and data file for the prompt failed due to syntax errors. "
                "Use the following PyOPL syntax implementation as a reference for valid PyOPL syntax:\n\n"
                "--- PyOPL syntax implementation ---\n"
                f"{grammar_implementation}\n"
                "--- END OF PyOPL syntax implementation ---\n\n"
                f"PROMPT:\n{prompt}\n\n"
                f"PREVIOUS MODEL:\n{model_code}\n\n"
                f"PREVIOUS DATA:\n{data_code}\n\n"
                f"SYNTAX ERRORS:\n{syntax_errors}\n\n"
                "Please revise the model and data to fix the errors while retaining alignment with the original intent. Return only a JSON object with 'model' and 'data'."
                "Return ONLY a strict JSON object with two fields: "
                "'model' (the PyOPL model as a single JSON string) and 'data' (the data file as a single JSON string). "
                "Do not include Markdown or code fences. Escape all double quotes and backslashes inside the strings."
            )

    # Load latest version of the model and data files
    with open(model_file, "r") as f:
        model_code = f.read()
    with open(data_file, "r") as f:
        data_code = f.read()

    # Final assessment prompt
    syntax_errors_str = f"SYNTAX ERRORS:\n{syntax_errors}\n\n" if syntax_errors else ""
    assessment_prompt = (
        "Given the following prompt and the generated PyOPL model and data, assess how well the model and data align with the original intent. "
        "Be critical and specific. Use the following PyOPL syntax implementation as a reference for valid PyOPL syntax:\n\n"
        "--- PyOPL syntax implementation ---\n"
        f"{grammar_implementation}\n"
        "--- END OF PyOPL syntax implementation ---\n\n"
        f"PROMPT:\n{prompt}\n\n"
        f"MODEL:\n{model_code}\n\n"
        f"DATA:\n{data_code}\n\n"
        f"{syntax_errors_str}"
        "Provide your assessment as a short texual paragraph."
    )
    assessment_response = client.responses.create(
        model=model_name, input=assessment_prompt, max_output_tokens=MAX_OUTPUT_TOKENS
    )
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


def generative_feedback(prompt, model_file, data_file):
    grammar_implementation = _read_pyopl_code()
    # Use API key from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")
    client = OpenAI(api_key=api_key)
    # Use GPT-5 model
    # model_name = "gpt-5"
    # model_name = "gpt-5-mini"
    # model_name = "gpt-5-nano"
    model_name = "gpt-4.1"

    # Read files first (avoid inline open().read())
    with open(model_file, "r", encoding="utf-8") as fh:
        model_code = fh.read()
    with open(data_file, "r", encoding="utf-8") as fh:
        data_code = fh.read()

    user_prompt = (
        "You are an expert in mathematical optimization and PyOPL. "
        "Answer the following question about the given PyOPL model and data file."
        "Use the following PyOPL syntax implementation as a reference for valid PyOPL syntax:\n\n"
        "--- PyOPL syntax implementation ---\n"
        f"{grammar_implementation}\n"
        "--- END OF PyOPL syntax implementation ---\n\n"
        "<prompt>\n"
        f"{prompt}\n"
        "</prompt>\n\n"
        "<model>\n"
        f"{model_code}\n"
        "</model>\n\n"
        "<data>\n"
        f"{data_code}\n"
        "</data>\n\n"
        "Return ONLY a strict JSON object with three fields, of which two are optional: "
        "'feedback' (your textual feedback), "
        "'revised_model' (the revised PyOPL model as a single JSON string), "
        "'revised_data' (the revised data file as a single JSON string). "
        "Feedback is mandatory, while revised_model and revised_data are optional and can be omitted if no changes are needed. "
        "Do not include Markdown or code fences. Escape all double quotes and backslashes inside the strings."
    )

    response = client.responses.create(model=model_name, input=user_prompt, max_output_tokens=MAX_OUTPUT_TOKENS)
    content = _coalesce_response_text(response)
    if not content:
        raise RuntimeError(f"Empty model response. Full response: {response}")
    try:
        result = json.loads(extract_json_from_markdown(content))
        return result
    except Exception as e:
        raise RuntimeError(f"Failed to parse GPT-5 response as JSON: {e}\nResponse: {content}")
