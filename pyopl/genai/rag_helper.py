from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- Logging Setup ---
# Use module-level logger, and set DEBUG level for development
logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _load_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
    except Exception as e:
        logging.warning("RAG disabled (embedding model load failed): %s", e)
        return None


def _iter_description_files(models_dir: Path) -> List[Path]:
    """
    Find all .txt files recursively under models_dir.
    """
    return sorted([p for p in models_dir.rglob("*.txt") if p.is_file()])


def _read_text(path: Path, max_chars: int = 100_000) -> str:
    """
    Read text content safely with UTF-8, truncating very large files.
    """
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            txt = f.read(max_chars + 1)
            if len(txt) > max_chars:
                txt = txt[:max_chars]
            return txt.strip()
    except Exception:
        return ""


def rank_problem_descriptions(
    query: str,
    models_dir: Optional[Path | str] = None,
    top_k: int = 10,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> List[Dict[str, Any]]:
    """
    Rank problem description .txt files under 'opl_models' by semantic similarity to the query.

    Args:
        query: The problem description to search with.
        models_dir: Root folder containing subfolders with .txt descriptions.
                    Defaults to '<repo_root>/opl_models'.
        top_k: Number of top matches to return.
        model_name: Sentence embedding model to use.

    Returns:
        A list of dicts with: path, score, preview.
    """
    repo_root = Path(__file__).resolve().parent
    models_dir = Path(models_dir) if models_dir else (repo_root / "opl_models")

    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    files_t = _iter_description_files(models_dir)
    if not files_t:
        return []

    texts_t = [_read_text(p) for p in files_t]
    # Filter out empty files while maintaining mapping
    nonempty = [(p, t) for p, t in zip(files_t, texts_t) if t]
    if not nonempty:
        return []

    files, texts = zip(*nonempty)

    model = _load_model(model_name)

    # Encode with normalized embeddings for cosine similarity via dot product
    # Using convert_to_tensor True for efficient cosine similarity
    doc_embs = model.encode(
        list(texts),
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_emb = model.encode(
        [query],
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    # Cosine similarity since embeddings are normalized
    # Equivalent to (doc_embs @ query_emb)
    import torch

    sims = torch.matmul(doc_embs, query_emb)
    scores = sims.cpu().tolist()

    # Rank by descending score
    ranked = sorted(
        (
            {
                "path": str(p),
                "score": float(s),
                "preview": texts[i][:300].replace("\n", " "),
            }
            for i, (p, s) in enumerate(zip(files, scores))
        ),
        key=lambda x: x["score"],
        reverse=True,
    )

    return ranked[: top_k if top_k > 0 else len(ranked)]


if __name__ == "__main__":
    # CLI usage:
    #   python rag.py "your problem description here"
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        print('Usage: python rag.py "your problem description here"')
        sys.exit(1)

    results = rank_problem_descriptions(query=query, top_k=10)
    if not results:
        print("No descriptions found.")
        sys.exit(0)

    for i, r in enumerate(results, 1):
        print(f"{i:2d}. {r['path']}  score={r['score']:.4f}")
        # print(f"    {r['preview']}")
