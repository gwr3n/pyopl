from __future__ import annotations

import inspect
import json
import logging
import os
import re
from dataclasses import dataclass
from enum import Enum, auto
from importlib.resources import files
from pathlib import Path
from time import sleep
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

from ..pyopl_core import OPLCompiler, SemanticError
from .genai_pricing import _extract_gemini_usage, _extract_openai_usage
from .genai_pricing import estimate_costs as _estimate_costs
from .rag_helper import rank_problem_descriptions as rag_rank


class LLMProvider(Enum):
    OPENAI = "openai"  # Default
    GOOGLE = "google"
    OLLAMA = "ollama"


class Grammar(Enum):
    NONE = auto()
    BNF = auto()
    CODE = auto()


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, usage: Dict[str, int]) -> None:
        self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)

    def as_dict(self) -> Dict[str, int]:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens}


class GenAIStrategyBase:
    """Shared LLM + compile + RAG plumbing for all genai strategies.

    Strategy modules should subclass this and keep only strategy-specific prompt building
    and control-flow. Public module APIs can remain as thin wrappers.
    """

    # Defaults (subclasses/modules may override)
    MAX_OUTPUT_TOKENS: Optional[int] = None
    FEW_SHOT_TOP_K: int = 3
    FEW_SHOT_MAX_CHARS: int = 2**31 - 1

    def __init__(
        self,
        *,
        logger: logging.Logger,
        max_output_tokens: Optional[int] = None,
        few_shot_top_k: Optional[int] = None,
        few_shot_max_chars: Optional[int] = None,
    ) -> None:
        self._logger = logger
        if max_output_tokens is not None:
            self.MAX_OUTPUT_TOKENS = max_output_tokens
        if few_shot_top_k is not None:
            self.FEW_SHOT_TOP_K = int(few_shot_top_k)
        if few_shot_max_chars is not None:
            self.FEW_SHOT_MAX_CHARS = int(few_shot_max_chars)

    # -------- Progress / logging --------

    def notify(self, progress: Optional[Callable[[str], None]], msg: str) -> None:
        try:
            if progress:
                progress(str(msg))
            else:
                self._logger.debug(str(msg))
        except Exception:
            # Never let UI callback failures break the run
            pass

    # -------- Grammar / package resources --------

    @staticmethod
    def read_file(path: str) -> str:
        with open(path, "r") as f:
            return f.read()

    @staticmethod
    def read_pyopl_GBNF() -> str:
        return (files("pyopl") / "grammars" / "PyOPL_GBNF").read_text(encoding="utf-8")

    @staticmethod
    def read_pyopl_grammar() -> str:
        return (files("pyopl") / "grammars" / "PyOPL grammar.md").read_text(encoding="utf-8")

    @staticmethod
    def read_pyopl_code() -> str:
        code_path = os.path.join(os.path.dirname(__file__), "..", "pyopl_core.py")
        return GenAIStrategyBase.read_file(os.path.normpath(code_path))

    @classmethod
    def get_grammar_implementation(cls, mode: Grammar) -> str:
        if mode == Grammar.NONE:
            return ""
        if mode == Grammar.BNF:
            return cls.read_pyopl_grammar()
        if mode == Grammar.CODE:
            return cls.read_pyopl_code()
        raise ValueError(f"Invalid mode: {mode}")

    # -------- RAG few-shot helpers --------

    def safe_read_text(self, path: Path, max_chars: Optional[int] = None) -> str:
        cap = self.FEW_SHOT_MAX_CHARS if max_chars is None else max_chars
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if len(text) > cap:
                text = text[:cap]
            return text.strip()
        except Exception:
            return ""

    @staticmethod
    def find_pair_in_folder(desc_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
        """Locate associated .mod and .dat near a description file."""
        folder = desc_path.parent
        stem = desc_path.stem

        mod: Optional[Path] = folder / f"{stem}.mod"
        dat: Optional[Path] = folder / f"{stem}.dat"

        if not (mod and mod.exists() and mod.is_file()):
            mods = sorted(folder.glob("*.mod"))
            mod = mods[0] if mods else None

        if not (dat and dat.exists() and dat.is_file()):
            dats = sorted(folder.glob("*.dat"))
            dat = dats[0] if dats else None

        return (mod if mod and mod.exists() else None, dat if dat and dat.exists() else None)

    def gather_few_shots(
        self,
        problem_description: str,
        *,
        k: Optional[int] = None,
        models_dir: Optional[str | Path] = None,
        progress: Optional[Callable[[str], None]] = None,
    ) -> List[Dict[str, str]]:
        top_k = self.FEW_SHOT_TOP_K if k is None else int(k)

        # Resolve default models_dir from package data with a concrete Path
        if models_dir is None:
            try:
                pkg_dir = files("pyopl") / "opl_models"
                base_dir = Path(str(pkg_dir))
                if not base_dir.exists():
                    base_dir = Path(__file__).parent / "opl_models"
            except Exception:
                base_dir = Path(__file__).parent / "opl_models"
        else:
            base_dir = Path(models_dir)

        examples: List[Dict[str, str]] = []
        try:
            self.notify(progress, f"Retrieving few-shot examples (k={top_k})")
            hits = rag_rank(query=problem_description, models_dir=str(base_dir), top_k=top_k)
            self.notify(progress, f"Found {len(hits)} few-shot candidates: {[Path(hit['path']).name for hit in hits]}")
        except Exception as e:
            self._logger.debug(f"Few-shot retrieval skipped: {e}")
            self.notify(progress, "Few-shot retrieval failed; continuing without examples")
            return examples

        for hit in hits:
            try:
                desc_path = Path(hit["path"])
                desc_text = self.safe_read_text(desc_path)
                mod_path, dat_path = self.find_pair_in_folder(desc_path)
                if not desc_text or not mod_path or not dat_path:
                    continue
                mod_text = self.safe_read_text(mod_path)
                dat_text = self.safe_read_text(dat_path)
                if not mod_text or not dat_text:
                    continue
                examples.append(
                    {
                        "description": desc_text,
                        "model": mod_text,
                        "data": dat_text,
                        "desc_path": str(desc_path),
                        "model_path": str(mod_path),
                        "data_path": str(dat_path),
                    }
                )
                if len(examples) >= top_k:
                    break
            except Exception as e:
                self._logger.debug(f"Skipping example due to error: {e}")
                continue
        return examples

    @staticmethod
    def render_few_shots_section(few_shots: Optional[List[Dict[str, str]]]) -> str:
        if not few_shots:
            return ""
        blocks: List[str] = []
        for i, ex in enumerate(few_shots, 1):
            desc_hdr = f'<description path="{ex.get("desc_path", "")}">'
            mod_hdr = f'<model_file path="{ex.get("model_path", "")}">'
            dat_hdr = f'<data_file path="{ex.get("data_path", "")}">'
            blocks.append(
                f'<example index="{i}">\n'
                f"{desc_hdr}\n{ex.get('description','')}\n</description>\n\n"
                f"{mod_hdr}\n{ex.get('model','')}\n</model_file>\n\n"
                f"{dat_hdr}\n{ex.get('data','')}\n</data_file>\n"
                f"</example>\n"
            )
        return (
            "<few_shot_examples>\n"
            "Use these exemplars for structure and syntax only. Tailor names/indices to this problem.\n"
            + "".join(blocks)
            + "</few_shot_examples>\n\n"
        )

    # -------- JSON extraction/parsing --------

    @staticmethod
    def extract_json_from_markdown(text: str) -> str:
        """Default: extract a JSON object from fenced block or first balanced {...}."""
        match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)

        start = text.find("{")
        if start == -1:
            return text
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return text

    @classmethod
    def json_loads_relaxed(cls, text: str) -> Any:
        try:
            return json.loads(text)
        except Exception:
            return json.loads(cls.extract_json_from_markdown(text))

    # -------- LLM clients / calls --------

    @staticmethod
    def _coalesce_response_text(resp: Any) -> str:
        if getattr(resp, "output_text", None):
            return resp.output_text or ""

        try:
            chunks: list[str] = []
            for item in getattr(resp, "output", []) or []:
                content_blocks = getattr(item, "content", None)
                if content_blocks is None:
                    if hasattr(item, "text"):
                        chunks.append(getattr(item, "text") or "")
                    continue
                for block in content_blocks:
                    if hasattr(block, "text"):
                        chunks.append(getattr(block, "text") or "")
                    elif isinstance(block, dict) and isinstance(block.get("text"), str):
                        chunks.append(block["text"])
            if chunks:
                return "".join(chunks)
        except Exception:
            pass

        try:
            first = getattr(resp, "output", [])[0]
            first_content = getattr(first, "content", [])[0]
            if hasattr(first_content, "text"):
                return first_content.text or ""
        except Exception:
            pass

        return ""

    @staticmethod
    def _openai_client():
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("openai is not installed. pip install openai") from e
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set.")
        return OpenAI(api_key=api_key)

    @staticmethod
    def _google_client():
        try:
            import google.generativeai as genai
        except Exception as e:
            raise RuntimeError("google.generativeai is not installed. pip install google-generativeai") from e
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)
        return genai

    @staticmethod
    def _ollama_generate_text(
        *,
        model_name: str,
        prompt: str,
        num_predict: Optional[int],
        return_usage: bool,
        enforce_json: bool,
    ) -> Union[str, Tuple[str, Dict[str, int]]]:
        try:
            from ollama import generate as ollama_generate
        except Exception as e:
            raise RuntimeError("ollama package is not installed. pip install ollama") from e
        options: Dict[str, Any] = {}
        if num_predict is not None:
            options["num_predict"] = num_predict
        if enforce_json:
            options["format"] = "json"
        resp = ollama_generate(model=model_name, prompt=prompt, options=options)
        try:
            text = resp.get("response", "") or ""
        except (TypeError, KeyError) as e:
            raise RuntimeError(f"Failed to retrieve response text from Ollama response: {e}")
        if not return_usage:
            return text
        prompt_tokens = resp.get("prompt_eval_count")
        completion_tokens = resp.get("eval_count")
        usage = {
            "prompt_tokens": int(prompt_tokens or 0),
            "completion_tokens": int(completion_tokens or 0),
        }
        return text, usage

    @staticmethod
    def infer_provider(llm_provider: Optional[str], model_name: str) -> LLMProvider:
        if llm_provider:
            lp = llm_provider.strip().lower()
            if lp in ("openai", "oai"):
                return LLMProvider.OPENAI
            if lp in ("google", "genai", "gemini", "google.generativeai"):
                return LLMProvider.GOOGLE
            if lp in ("ollama",):
                return LLMProvider.OLLAMA
        if model_name.startswith("gemini"):
            return LLMProvider.GOOGLE
        if "gpt-oss" in model_name or model_name.startswith(("llama", "qwen", "mistral")):
            return LLMProvider.OLLAMA
        return LLMProvider.OPENAI

    @staticmethod
    def _build_openai_create_params(
        *,
        model_name: str,
        input_text: str,
        max_tokens: Optional[int],
        temperature: Optional[float],
        stop: Optional[list[str]],
        expected_json: bool,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "model": model_name,
            "input": input_text,
        }
        if expected_json:
            params["response_format"] = {"type": "json"}
        if max_tokens is not None:
            params["max_output_tokens"] = max_tokens
        if temperature is not None:
            params["temperature"] = temperature
        if stop:
            params["stop"] = stop
        return params

    def _call_openai_with_retry(
        self,
        client: Any,
        create_params: Dict[str, Any],
        *,
        retries: int = 3,
        backoff_sec: float = 1.5,
        progress: Optional[Callable[[str], None]] = None,
    ) -> Any:
        last_err: Optional[Exception] = None
        fallback_keys = ["response_format", "stop", "reasoning", "temperature"]
        params = dict(create_params)

        # Prune unsupported params for older SDKs/servers
        try:
            create_callable = client.responses.create
            sig = inspect.signature(create_callable)
            if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                supported = set(sig.parameters.keys())
                supported.discard("self")
                for k in list(params.keys()):
                    if k not in supported:
                        params.pop(k, None)
        except Exception:
            pass

        def _strip_param_from_error_message(msg: str) -> bool:
            removed = False
            low = msg.lower()
            for key in list(fallback_keys):
                if key in params and (
                    f"unexpected keyword argument '{key}'" in low or f"unsupported parameter: '{key}'" in low or key in low
                ):
                    params.pop(key, None)
                    removed = True
            m = re.search(r"unsupported parameter:\s*'([^']+)'", msg, re.IGNORECASE)
            if m and m.group(1) in params:
                params.pop(m.group(1), None)
                removed = True
            m2 = re.search(r"parameter\s*'([^']+)'\s*is not supported", msg, re.IGNORECASE)
            if m2 and m2.group(1) in params:
                params.pop(m2.group(1), None)
                removed = True
            return removed

        for attempt in range(retries):
            try:
                return client.responses.create(**params)
            except Exception as e:
                last_err = e
                msg = str(e) if e else "unknown error"
                self.notify(progress, f"[LLM] OpenAI: {msg}")
                if _strip_param_from_error_message(msg):
                    self.notify(progress, "[LLM] OpenAI: retrying without unsupported parameters")
                    continue
                self.notify(progress, f"[LLM] OpenAI: retry {attempt + 1}/{retries} after error: {msg}")
                sleep(backoff_sec * (2**attempt))
        self.notify(progress, f"[LLM] OpenAI: failed after {retries} attempts")
        raise RuntimeError(f"OpenAI request failed after {retries} attempts: {last_err}")

    def llm_generate_text(
        self,
        *,
        provider: LLMProvider,
        model_name: str,
        input_text: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[list[str]] = None,
        progress: Optional[Callable[[str], None]] = None,
        capture_usage: bool = False,
        expected_json: bool = False,
    ) -> Union[str, Tuple[str, Dict[str, int]]]:
        mt = self.MAX_OUTPUT_TOKENS if max_tokens is None else max_tokens

        if provider == LLMProvider.OPENAI:
            client = self._openai_client()
            create_params = self._build_openai_create_params(
                model_name=model_name,
                input_text=input_text,
                max_tokens=mt,
                temperature=temperature,
                stop=stop,
                expected_json=expected_json,
            )
            self.notify(progress, f"[LLM] OpenAI • {model_name}: sending request")
            response = self._call_openai_with_retry(client, create_params, progress=progress)
            self.notify(progress, "[LLM] OpenAI: response received")
            response_text = self._coalesce_response_text(response)
            if not response_text:
                raise RuntimeError(f"Empty OpenAI response: {response}.")
            if not capture_usage:
                return response_text
            usage = _extract_openai_usage(response, input_text, response_text, model_name)
            return response_text, usage

        if provider == LLMProvider.GOOGLE:
            genai = self._google_client()
            model = genai.GenerativeModel(model_name)
            generation_config: Dict[str, Any] = {}
            if mt is not None:
                generation_config["max_output_tokens"] = mt
            if temperature is not None:
                generation_config["temperature"] = temperature
            if expected_json:
                generation_config["response_mime_type"] = "application/json"
            self.notify(progress, f"[LLM] Gemini • {model_name}: sending request")
            resp = model.generate_content(input_text, generation_config=generation_config)
            self.notify(progress, "[LLM] Gemini: response received")
            text = getattr(resp, "text", None)
            if not text and getattr(resp, "candidates", None):
                parts: list[str] = []
                for c in resp.candidates:
                    content = getattr(c, "content", None)
                    if content and hasattr(content, "parts"):
                        for p in content.parts:
                            if hasattr(p, "text"):
                                parts.append(p.text or "")
                text = "".join(parts)
            text = text or ""
            if not capture_usage:
                return text
            usage = _extract_gemini_usage(resp, input_text, text)
            return text, usage

        if provider == LLMProvider.OLLAMA:
            self.notify(progress, f"[LLM] Ollama • {model_name}: generating")
            if not capture_usage:
                result = self._ollama_generate_text(
                    model_name=model_name,
                    prompt=input_text,
                    num_predict=mt,
                    return_usage=False,
                    enforce_json=expected_json,
                )
                self.notify(progress, "[LLM] Ollama: response received")
                return result
            result = self._ollama_generate_text(
                model_name=model_name,
                prompt=input_text,
                num_predict=mt,
                return_usage=True,
                enforce_json=expected_json,
            )
            result_text, usage = cast(Tuple[str, Dict[str, int]], result)
            self.notify(progress, "[LLM] Ollama: response received")
            return result_text, usage

        raise ValueError(f"Unsupported LLM provider: {provider}")

    # -------- Compile + filesystem helpers --------

    @staticmethod
    def compile_model_data(model_code: str, data_code: str) -> List[str]:
        compiler = OPLCompiler()
        errors: List[str] = []
        try:
            compiler.compile_model(model_code, data_code)
        except SemanticError as e:
            errors.append(str(e))
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")
        return errors

    @staticmethod
    def write_model_data_files(model_file: str, data_file: str, model_code: str, data_code: str) -> None:
        model_dir = os.path.dirname(model_file)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        data_dir = os.path.dirname(data_file)
        if data_dir:
            os.makedirs(data_dir, exist_ok=True)
        with open(model_file, "w") as f:
            f.write(model_code)
        with open(data_file, "w") as f:
            f.write(data_code)

    # -------- Pricing --------

    @staticmethod
    def estimate_cost(model_name: str, usage: Usage) -> Dict[str, Any]:
        try:
            from types import SimpleNamespace
        except Exception:
            SimpleNamespace = None  # type: ignore

        usage_summary = usage.as_dict()
        estimated_costs: Dict[str, Any] = {}
        if SimpleNamespace is not None:
            try:
                args = SimpleNamespace(model=model_name)
                estimated_costs = _estimate_costs(args, usage_summary) or {}
            except Exception:
                estimated_costs = {}
        return {
            "model": model_name,
            "usage": usage_summary,
            "estimated_costs": estimated_costs,
        }


# -------- Model discovery helpers (shared) --------


def list_openai_models(*, prefix: Optional[str] = "gpt") -> list[str]:
    client = GenAIStrategyBase._openai_client()
    try:
        resp = client.models.list()
    except Exception as e:
        raise RuntimeError(f"Failed to list OpenAI models: {e}")

    names: list[str] = []
    data = getattr(resp, "data", None)
    items = data if isinstance(data, list) else (list(resp) if resp is not None else [])
    for m in items:
        mid = getattr(m, "id", None) or (m.get("id") if isinstance(m, dict) else None)
        if isinstance(mid, str):
            names.append(mid)
    if prefix:
        names = [n for n in names if n.startswith(prefix)]
    return sorted(set(names))


def list_gemini_models(*, prefix: Optional[str] = "gemini") -> list[str]:
    genai = GenAIStrategyBase._google_client()
    try:
        models = genai.list_models()
    except Exception as e:
        raise RuntimeError(f"Failed to list Gemini models: {e}")

    names: list[str] = []
    for m in models or []:
        if "generateContent" in m.supported_generation_methods:
            name = m.name
            if isinstance(name, str) and name.startswith("models/"):
                name = name[len("models/") :]
                if prefix and name.startswith(prefix):
                    names.append(name)

    return sorted(set(names))


def list_ollama_models(*, prefix: Optional[str] = None) -> list[str]:
    try:
        from ollama import list as ollama_list
    except Exception as e:
        raise RuntimeError("ollama package is not installed. pip install ollama") from e

    models: list[str] = []
    try:
        resp = ollama_list()
        items = resp.get("models", []) if isinstance(resp, dict) else getattr(resp, "models", [])
        for m in items:
            name = (m.get("model") if isinstance(m, dict) else getattr(m, "model", None)) or (
                m.get("name") if isinstance(m, dict) else getattr(m, "name", None)
            )
            if isinstance(name, str):
                models.append(name)
    except Exception as e:
        raise RuntimeError(f"Failed to list Ollama models: {e}")

    if prefix:
        models = [n for n in models if n.startswith(prefix)]
    return sorted(set(models))


def list_models(*, llm_provider: Optional[str] = None, model_name: str) -> list[str]:
    provider = GenAIStrategyBase.infer_provider(llm_provider, model_name)
    if provider == LLMProvider.OPENAI:
        return list_openai_models()
    if provider == LLMProvider.GOOGLE:
        return list_gemini_models()
    if provider == LLMProvider.OLLAMA:
        return list_ollama_models()
    raise ValueError(f"Unsupported LLM provider: {provider}")
