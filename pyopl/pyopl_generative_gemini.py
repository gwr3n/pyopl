import json
import os
import re

import google.generativeai as genai

from .pyopl_core import OPLCompiler, SemanticError

MAX_ITERATIONS = 5


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


def _coalesce_gemini_text(resp) -> str:
    # Prefer SDK convenience if present
    if getattr(resp, "text", None):
        return resp.text or ""
    # Fallback: walk candidates -> content.parts
    try:
        candidates = getattr(resp, "candidates", None) or []
        chunks = []
        for cand in candidates:
            # Handle blocked/safety finishes explicitly
            finish = getattr(cand, "finish_reason", None) or getattr(cand, "finishReason", None)
            if finish in {"SAFETY", "BLOCKLIST", "PROHIBITED"}:
                raise RuntimeError(f"Response blocked by safety: finish_reason={finish}")
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None
            if parts:
                for p in parts:
                    # Parts can be text or dict-like
                    if hasattr(p, "text"):
                        chunks.append(getattr(p, "text") or "")
                    elif isinstance(p, dict) and isinstance(p.get("text"), str):
                        chunks.append(p["text"])
        return "".join(chunks)
    except Exception:
        return ""

# Use Gemini 2.5 Flash model
# model_name = "gemini-2.5-flash"
def generative_solve(prompt, model_file, data_file, model_name = "gemini-2.5-flash", iterations=MAX_ITERATIONS):
    """
    Generate a PyOPL model and data file from a prompt using Gemini, validate with pyopl, iterate on errors, and assess alignment.
    Args:
            prompt (str): Textual description of the optimization problem.
            model_file (str): Path to save the generated model file.
            data_file (str): Path to save the generated data file.
    Returns:
            str: Gemini's assessment of alignment between model/data and prompt.
    """

    # grammar = _read_pyopl_grammar()
    grammar_implementation = _read_pyopl_code()
    # Use API key from environment variable
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel(
        model_name,
        generation_config={
            "temperature": 0.2,  # low temperature for reliability
            "response_mime_type": "application/json",  # ask for JSON
        },
    )

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
        response = model.generate_content(user_prompt)
        try:
            # When response_mime_type is application/json, response.text is JSON
            content = _coalesce_gemini_text(response) or getattr(response, "text", "") or ""
            if not content:
                raise RuntimeError(f"Empty model response. Full response: {response}")
            # Extract JSON from Gemini's response
            # json_str = extract_json_from_markdown(content)
            result = json.loads(content)
            model_code = result["model"]
            data_code = result["data"]
        except Exception as e:
            raise RuntimeError(
                f"Failed to parse Gemini response as JSON: {e}\nResponse: {getattr(response, 'text', str(response))}"
            )

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
            # Feedback errors to Gemini and retry
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
    assessment_prompt = (
        "Given the following prompt and the generated PyOPL model and data, assess how well the model and data align with the original intent. "
        "Be critical and specific. Use the following PyOPL syntax implementation as a reference for valid PyOPL syntax:\n\n"
        "--- PyOPL syntax implementation ---\n"
        f"{grammar_implementation}\n"
        "--- END OF PyOPL syntax implementation ---\n\n"
        f"PROMPT:\n{prompt}\n\n"
        f"MODEL:\n{model_code}\n\n"
        f"DATA:\n{data_code}\n\n"
        "Provide your assessment as a short textual paragraph."
    )
    assessment_response = model.generate_content(assessment_prompt)
    assessment_text = _coalesce_gemini_text(assessment_response) or getattr(assessment_response, "text", "") or ""
    if not assessment_text:
        raise RuntimeError(f"Empty assessment response. Full response: {assessment_response}")
    return assessment_text.strip()


def generative_feedback(prompt, model_file, data_file):
    grammar_implementation = _read_pyopl_code()
    # Use API key from environment variable
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    # Use Gemini 2.5 Flash model
    model_name = "gemini-2.5-flash"
    model = genai.GenerativeModel(
        model_name,
        generation_config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
        },
    )

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

    response = model.generate_content(user_prompt)
    try:
        content = _coalesce_gemini_text(response) or (response.text or "")
        if not content:
            raise RuntimeError(f"Empty model response. Full response: {response}")
        # Extract JSON from Gemini's response
        # json_str = extract_json_from_markdown(content)
        result = json.loads(content)
        return result
    except Exception as e:
        raise RuntimeError(
            f"Failed to parse Gemini response as JSON: {e}\nResponse: {getattr(response, 'text', str(response))}"
        )
