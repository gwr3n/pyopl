import json
import os
import re

from ollama import generate

from .pyopl_core import OPLCompiler, SemanticError

MAX_ITERATIONS = 5
MAX_OUTPUT_TOKENS = 4096 * 2  # used as num_predict for Ollama


def _read_pyopl_grammar():
    grammar_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "PyOPL grammar.md")
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
    model_name="gpt-oss:20b",
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
        iterations (int): Maximum number of refinement iterations.
        return_statistics (bool): If True, return a dict with stats and assessment.
    Returns:
        str | dict: Assessment text or dict with iterations, assessment, and syntax_errors.
    """
    grammar_implementation = _read_pyopl_code()

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

        # Feedback errors and retry
        user_prompt = (
            "You are an expert in mathematical optimization and PyOPL. "
            "The following attempt to generate a PyOPL model and data file for the prompt failed due to syntax errors. "
            "Use the following PyOPL syntax implementation as a reference for valid PyOPL syntax:\n\n"
            "--- PyOPL syntax implementation ---\n"
            f"{grammar_implementation}\n"
            "--- END OF PyOPL syntax implementation ---\n\n"
            f"PROMPT:\n{prompt}\n\n"
            f"PREVIOUS MODEL:\n{model_code}\n\n"
            f"PREVIOUS DATA:\n{data_code}\n\n"
            f"SYNTAX ERRORS:\n{syntax_errors}\n\n"
            "Please revise the model and data to fix the errors while retaining alignment with the original intent. "
            "Return ONLY a strict JSON object with two fields: "
            "'model' (the PyOPL model as a single JSON string) and 'data' (the data file as a single JSON string). "
            "Do not include Markdown or code fences. Escape all double quotes and backslashes inside the strings."
        )

    # Load latest version of the model and data files (ensure we assess what's written)
    with open(model_file, "r", encoding="utf-8") as f:
        model_code = f.read()
    with open(data_file, "r", encoding="utf-8") as f:
        data_code = f.read()

    # Final assessment
    syntax_errors_str = f"SYNTAX ERRORS:\n{syntax_errors}\n\n" if syntax_errors else ""
    assessment_prompt = (
        "You are an expert in mathematical optimization and PyOPL. "
        "Given the following prompt and the generated PyOPL model and data, assess how well the model and data align with the original intent. "
        "Be critical and specific. Use the following PyOPL syntax implementation as a reference for valid PyOPL syntax:\n\n"
        "--- PyOPL syntax implementation ---\n"
        f"{grammar_implementation}\n"
        "--- END OF PyOPL syntax implementation ---\n\n"
        f"PROMPT:\n{prompt}\n\n"
        f"MODEL:\n{model_code}\n\n"
        f"DATA:\n{data_code}\n\n"
        f"{syntax_errors_str}"
        "Provide your assessment as a short textual paragraph."
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
def generative_feedback(prompt, model_file, data_file, model_name="gpt-oss:20b"):
    """
    Ask questions or request revisions about a given PyOPL model and data using Ollama.
    Returns a JSON object with:
      - 'feedback' (str, mandatory)
      - 'revised_model' (str, optional)
      - 'revised_data' (str, optional)
    """
    grammar_implementation = _read_pyopl_code()

    with open(model_file, "r", encoding="utf-8") as fh:
        model_code = fh.read()
    with open(data_file, "r", encoding="utf-8") as fh:
        data_code = fh.read()

    user_prompt = (
        "You are an expert in mathematical optimization and PyOPL. "
        "Answer the following question about the given PyOPL model and data file. "
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

    content = _ollama_generate_text(model_name, user_prompt, num_predict=MAX_OUTPUT_TOKENS)
    if not content:
        raise RuntimeError("Empty model response from Ollama.")
    try:
        result = json.loads(extract_json_from_markdown(content))
        return result
    except Exception as e:
        raise RuntimeError(f"Failed to parse Ollama response as JSON: {e}\nResponse: {content}")
