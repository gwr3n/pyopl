from __future__ import annotations

import base64
import inspect
import json
import logging
import os
import re
import mimetypes
from dataclasses import dataclass
from enum import Enum, auto
from importlib.resources import files
from pathlib import Path
from time import sleep
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, cast, TypedDict

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


class ImageInput(TypedDict, total=False):
    """
    A single image reference for multimodal prompts.

    Supported forms:
      - {"path": "/local/path.png", "mime_type": "image/png"}   (mime_type optional)
      - {"url": "https://.../image.png", "mime_type": "image/png"} (mime_type optional; provider-dependent)
      - {"data_base64": "...", "mime_type": "image/png"}       (mime_type recommended)
    """
    path: str
    url: str
    data_base64: str
    mime_type: str


class PromptWithImages(TypedDict, total=False):
    """
    Multimodal prompt:
      - {"text": "...", "images": [<image inputs>]}
      - {"text": "...", "image": <single image input>}  (convenience)
    """
    text: str
    images: List[Any]
    image: Any


PromptInput = Union[str, PromptWithImages]


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, usage: Dict[str, int]) -> None:
        self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)

    def as_dict(self) -> Dict[str, int]:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens}


@dataclass(frozen=True)
class GoogleClient:
    kind: Literal["new", "legacy"]
    client: Any


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

    # -------- Multimodal prompt normalization --------

    @staticmethod
    def _normalize_images(images: Any) -> List[ImageInput]:
        if not images:
            return []

        out: List[ImageInput] = []
        if not isinstance(images, list):
            images = [images]

        for img in images:
            if isinstance(img, Path):
                out.append({"path": str(img)})
                continue
            if isinstance(img, str):
                out.append({"path": img})
                continue
            if isinstance(img, dict):
                # Accept already-normalized dicts, but only keep known keys
                entry: ImageInput = {}
                if isinstance(img.get("path"), str):
                    entry["path"] = img["path"]
                if isinstance(img.get("url"), str):
                    entry["url"] = img["url"]
                if isinstance(img.get("data_base64"), str):
                    entry["data_base64"] = img["data_base64"]
                if isinstance(img.get("mime_type"), str):
                    entry["mime_type"] = img["mime_type"]
                if entry:
                    out.append(entry)
                continue

        return out

    @classmethod
    def normalize_prompt_input(cls, prompt: PromptInput) -> Tuple[str, List[ImageInput]]:
        """
        Returns (text, images).

        Backwards compatible:
          - prompt: str -> (prompt, [])
          - prompt: {"text": "...", "images":[...]} or {"text":"...", "image": ...}
        """
        if isinstance(prompt, str):
            return prompt, []

        if isinstance(prompt, dict):
            text = prompt.get("text", "")
            if not isinstance(text, str):
                text = str(text)

            images_raw: Any = None
            if "images" in prompt:
                images_raw = prompt.get("images")
            elif "image" in prompt:
                images_raw = prompt.get("image")

            images = cls._normalize_images(images_raw)
            return text, images

        # last resort: stringify
        return str(prompt), []

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
                f"{desc_hdr}\n{ex.get('description', '')}\n</description>\n\n"
                f"{mod_hdr}\n{ex.get('model', '')}\n</model_file>\n\n"
                f"{dat_hdr}\n{ex.get('data', '')}\n</data_file>\n"
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
    def _google_client() -> GoogleClient:
        """
        Prefer the new `google.genai` SDK (google-genai). Fall back to the deprecated
        `google.generativeai` SDK only if the new one isn't available.
        """
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set.")

        # New SDK: google-genai (module: google.genai)
        try:
            import google.genai as genai  # type: ignore

            return GoogleClient(kind="new", client=genai.Client(api_key=api_key))
        except Exception:
            # Legacy SDK fallback (deprecated)
            try:
                import google.generativeai as genai_legacy  # type: ignore
            except Exception as e:
                raise RuntimeError("google-genai is not installed. pip install google-genai") from e

            genai_legacy.configure(api_key=api_key)
            return GoogleClient(kind="legacy", client=genai_legacy)

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

    # ---- Multimodal helpers (provider-specific payloads) ----

    @staticmethod
    def _guess_mime_type(path: str, fallback: str = "application/octet-stream") -> str:
        mt, _ = mimetypes.guess_type(path)
        return mt or fallback

    @classmethod
    def _image_to_openai_image_url(cls, img: ImageInput) -> str:
        """
        OpenAI Responses API expects image_url as a URL or a data URL.
        """
        if isinstance(img.get("url"), str) and img["url"]:
            return img["url"]

        if isinstance(img.get("data_base64"), str) and img["data_base64"]:
            mime_type = img.get("mime_type") or "application/octet-stream"
            data = img["data_base64"]
            if data.startswith("data:"):
                return data
            return f"data:{mime_type};base64,{data}"

        path = img.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError("Invalid image input for OpenAI: missing 'path', 'url', or 'data_base64'")

        mime_type = img.get("mime_type") or cls._guess_mime_type(path, fallback="image/png")
        raw = Path(path).read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{mime_type};base64,{b64}"

    @classmethod
    def _build_openai_input(cls, *, input_text: str, images: Optional[List[ImageInput]]) -> Any:
        if not images:
            return input_text

        content: List[Dict[str, Any]] = [{"type": "input_text", "text": input_text}]
        for img in images:
            content.append({"type": "input_image", "image_url": cls._image_to_openai_image_url(img)})

        return [{"role": "user", "content": content}]

    @classmethod
    def _image_to_gemini_part(cls, *, img: ImageInput, genai_types: Any) -> Any:
        """
        For google.genai new SDK.
        Prefer Part.from_uri when url is provided; otherwise Part.from_bytes.
        """
        url = img.get("url")
        if isinstance(url, str) and url:
            if hasattr(genai_types.Part, "from_uri"):
                mime_type = img.get("mime_type") or "image/png"
                return genai_types.Part.from_uri(file_uri=url, mime_type=mime_type)
            raise RuntimeError("Gemini URL images require google.genai types.Part.from_uri support")

        data_b64 = img.get("data_base64")
        if isinstance(data_b64, str) and data_b64:
            mime_type = img.get("mime_type") or "image/png"
            if data_b64.startswith("data:"):
                # data URL: data:<mime>;base64,<payload>
                m = re.match(r"data:([^;]+);base64,(.*)$", data_b64, re.DOTALL)
                if m:
                    mime_type = m.group(1) or mime_type
                    data_b64 = m.group(2)
            raw = base64.b64decode(data_b64)
            return genai_types.Part.from_bytes(data=raw, mime_type=mime_type)

        path = img.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError("Invalid image input for Gemini: missing 'path', 'url', or 'data_base64'")

        mime_type = img.get("mime_type") or cls._guess_mime_type(path, fallback="image/png")
        raw = Path(path).read_bytes()
        return genai_types.Part.from_bytes(data=raw, mime_type=mime_type)

    @staticmethod
    def _build_openai_create_params(
        *,
        model_name: str,
        input_content: Any,
        max_tokens: Optional[int],
        temperature: Optional[float],
        stop: Optional[list[str]],
        expected_json: bool,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "model": model_name,
            "input": input_content,
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

    def _generate_openai(
        self,
        *,
        model_name: str,
        input_text: str,
        images: Optional[List[ImageInput]],
        mt: Optional[int],
        temperature: Optional[float],
        stop: Optional[list[str]],
        progress: Optional[Callable[[str], None]],
        capture_usage: bool,
        expected_json: bool,
    ) -> Tuple[str, Optional[Dict[str, int]]]:
        client = self._openai_client()
        input_content = self._build_openai_input(input_text=input_text, images=images)
        create_params = self._build_openai_create_params(
            model_name=model_name,
            input_content=input_content,
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
            return response_text, None
        # Note: usage estimation falls back to tokenizing input_text only (images excluded).
        usage = _extract_openai_usage(response, input_text, response_text, model_name)
        return response_text, usage

    @staticmethod
    def _build_gemini_config(
        *,
        mt: Optional[int],
        temperature: Optional[float],
        expected_json: bool,
    ) -> Dict[str, Any]:
        config: Dict[str, Any] = {}
        if mt is not None:
            config["max_output_tokens"] = mt
        if temperature is not None:
            config["temperature"] = temperature
        if expected_json:
            config["response_mime_type"] = "application/json"
        return config

    def _generate_gemini_newsdk(
        self,
        g: Any,
        *,
        model_name: str,
        input_text: str,
        images: Optional[List[ImageInput]],
        mt: Optional[int],
        temperature: Optional[float],
        progress: Optional[Callable[[str], None]],
        capture_usage: bool,
        expected_json: bool,
    ) -> Tuple[str, Optional[Dict[str, int]]]:
        config = self._build_gemini_config(mt=mt, temperature=temperature, expected_json=expected_json)

        self.notify(progress, f"[LLM] Gemini • {model_name}: sending request (google.genai)")
        try:
            from google.genai import types as genai_types  # type: ignore

            if images:
                parts: List[Any] = [genai_types.Part.from_text(input_text)]
                for img in images:
                    parts.append(self._image_to_gemini_part(img=img, genai_types=genai_types))
                contents: Any = [genai_types.Content(role="user", parts=parts)]
            else:
                contents = input_text

            resp = g.models.generate_content(
                model=model_name,
                contents=contents,
                config=genai_types.GenerateContentConfig(**config),
            )
        except Exception:
            # Fallback: pass dict config; if images exist and typed parts aren't available, try inline_data shape.
            if images:
                inline_parts: List[Dict[str, Any]] = [{"text": input_text}]
                for img in images:
                    # Only support bytes/base64/path in fallback
                    if isinstance(img.get("url"), str) and img["url"]:
                        raise RuntimeError("Gemini image URLs require google.genai typed parts (install/upgrade google-genai)")
                    mime_type = img.get("mime_type") or "image/png"
                    data_b64 = img.get("data_base64")
                    if not (isinstance(data_b64, str) and data_b64):
                        path = img.get("path")
                        if not isinstance(path, str) or not path:
                            raise ValueError("Invalid image input for Gemini: missing 'path' or 'data_base64'")
                        mime_type = img.get("mime_type") or self._guess_mime_type(path, fallback="image/png")
                        data_b64 = base64.b64encode(Path(path).read_bytes()).decode("ascii")
                    inline_parts.append({"inline_data": {"mime_type": mime_type, "data": data_b64}})
                contents = [{"role": "user", "parts": inline_parts}]
            else:
                contents = input_text

            resp = g.models.generate_content(model=model_name, contents=contents, config=config)

        self.notify(progress, "[LLM] Gemini: response received")

        text = getattr(resp, "text", None)
        if not isinstance(text, str) or not text:
            try:
                parts2: list[str] = []
                for c in getattr(resp, "candidates", []) or []:
                    content = getattr(c, "content", None)
                    for p in getattr(content, "parts", []) or []:
                        t = getattr(p, "text", None)
                        if isinstance(t, str):
                            parts2.append(t)
                text = "".join(parts2)
            except Exception:
                text = ""
        text = text or ""

        if not capture_usage:
            return text, None
        # Note: usage estimation falls back to tokenizing input_text only (images excluded).
        usage = _extract_gemini_usage(resp, input_text, text)
        return text, usage

    def _generate_gemini_legacy(
        self,
        genai: Any,
        *,
        model_name: str,
        input_text: str,
        images: Optional[List[ImageInput]],
        mt: Optional[int],
        temperature: Optional[float],
        progress: Optional[Callable[[str], None]],
        capture_usage: bool,
        expected_json: bool,
    ) -> Tuple[str, Optional[Dict[str, int]]]:
        model = genai.GenerativeModel(model_name)
        generation_config = self._build_gemini_config(mt=mt, temperature=temperature, expected_json=expected_json)

        self.notify(progress, f"[LLM] Gemini • {model_name}: sending request (google.generativeai legacy)")

        if images:
            # Legacy SDK prefers PIL.Image. Support local paths (and base64) only.
            try:
                from PIL import Image  # type: ignore
                from io import BytesIO
            except Exception as e:
                raise RuntimeError(
                    "Gemini legacy SDK image prompts require Pillow. Install with: pip install pillow "
                    "or use the new google-genai SDK."
                ) from e

            parts: List[Any] = [input_text]
            for img in images:
                if isinstance(img.get("url"), str) and img["url"]:
                    raise RuntimeError("Gemini legacy SDK does not support URL images in this implementation")
                data_b64 = img.get("data_base64")
                if isinstance(data_b64, str) and data_b64:
                    if data_b64.startswith("data:"):
                        m = re.match(r"data:([^;]+);base64,(.*)$", data_b64, re.DOTALL)
                        if m:
                            data_b64 = m.group(2)
                    raw = base64.b64decode(data_b64)
                    parts.append(Image.open(BytesIO(raw)))
                    continue
                path = img.get("path")
                if not isinstance(path, str) or not path:
                    raise ValueError("Invalid image input for Gemini legacy: missing 'path' or 'data_base64'")
                parts.append(Image.open(path))

            resp = model.generate_content(parts, generation_config=generation_config)
        else:
            resp = model.generate_content(input_text, generation_config=generation_config)

        self.notify(progress, "[LLM] Gemini: response received")

        text = getattr(resp, "text", None)
        if not text and getattr(resp, "candidates", None):
            parts3: list[str] = []
            for c in resp.candidates:
                content = getattr(c, "content", None)
                if content and hasattr(content, "parts"):
                    for p in content.parts:
                        if hasattr(p, "text"):
                            parts3.append(p.text or "")
            text = "".join(parts3)
        text = text or ""

        if not capture_usage:
            return text, None
        usage = _extract_gemini_usage(resp, input_text, text)
        return text, usage

    def _generate_ollama(
        self,
        *,
        model_name: str,
        input_text: str,
        images: Optional[List[ImageInput]],
        mt: Optional[int],
        progress: Optional[Callable[[str], None]],
        capture_usage: bool,
        expected_json: bool,
    ) -> Tuple[str, Optional[Dict[str, int]]]:
        if images:
            raise RuntimeError("Ollama image prompts are not supported by this strategy implementation.")
        self.notify(progress, f"[LLM] Ollama • {model_name}: generating")
        if not capture_usage:
            text = cast(
                str,
                self._ollama_generate_text(
                    model_name=model_name,
                    prompt=input_text,
                    num_predict=mt,
                    return_usage=False,
                    enforce_json=expected_json,
                ),
            )
            self.notify(progress, "[LLM] Ollama: response received")
            return text, None

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

    def llm_generate_text(
        self,
        *,
        provider: LLMProvider,
        model_name: str,
        input_text: str,
        images: Optional[List[ImageInput]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[list[str]] = None,
        progress: Optional[Callable[[str], None]] = None,
        capture_usage: bool = False,
        expected_json: bool = False,
    ) -> Union[str, Tuple[str, Dict[str, int]]]:
        mt = self.MAX_OUTPUT_TOKENS if max_tokens is None else max_tokens

        if provider == LLMProvider.OPENAI:
            text, usage = self._generate_openai(
                model_name=model_name,
                input_text=input_text,
                images=images,
                mt=mt,
                temperature=temperature,
                stop=stop,
                progress=progress,
                capture_usage=capture_usage,
                expected_json=expected_json,
            )
            if not capture_usage:
                return text
            return text, cast(Dict[str, int], usage)

        if provider == LLMProvider.GOOGLE:
            g = self._google_client()

            if g.kind == "new":
                text, usage = self._generate_gemini_newsdk(
                    g.client,
                    model_name=model_name,
                    input_text=input_text,
                    images=images,
                    mt=mt,
                    temperature=temperature,
                    progress=progress,
                    capture_usage=capture_usage,
                    expected_json=expected_json,
                )
            else:
                text, usage = self._generate_gemini_legacy(
                    g.client,
                    model_name=model_name,
                    input_text=input_text,
                    images=images,
                    mt=mt,
                    temperature=temperature,
                    progress=progress,
                    capture_usage=capture_usage,
                    expected_json=expected_json,
                )

            if not capture_usage:
                return text
            return text, cast(Dict[str, int], usage)

        if provider == LLMProvider.OLLAMA:
            text, usage = self._generate_ollama(
                model_name=model_name,
                input_text=input_text,
                images=images,
                mt=mt,
                progress=progress,
                capture_usage=capture_usage,
                expected_json=expected_json,
            )
            if not capture_usage:
                return text
            return text, cast(Dict[str, int], usage)

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


def _list_gemini_models_newsdk(g: Any, *, prefix: Optional[str]) -> list[str]:
    names: list[str] = []
    try:
        for m in g.models.list():
            name = getattr(m, "name", None) or (m.get("name") if isinstance(m, dict) else None)
            if isinstance(name, str) and name.startswith("models/"):
                name = name[len("models/") :]
            if isinstance(name, str):
                if not prefix or name.startswith(prefix):
                    names.append(name)
    except Exception as e:
        raise RuntimeError(f"Failed to list Gemini models (google.genai): {e}")
    return sorted(set(names))


def _list_gemini_models_legacy(genai: Any, *, prefix: Optional[str]) -> list[str]:
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
                # Preserve existing behavior: only include when prefix is truthy
                if prefix and name.startswith(prefix):
                    names.append(name)

    return sorted(set(names))


def list_gemini_models(*, prefix: Optional[str] = "gemini") -> list[str]:
    g = GenAIStrategyBase._google_client()
    if g.kind == "new":
        return _list_gemini_models_newsdk(g.client, prefix=prefix)
    return _list_gemini_models_legacy(g.client, prefix=prefix)


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
