# GenAI RAG Working-Directory Models Roadmap

## Goal

Extend default GenAI few-shot retrieval so it searches both the packaged
`pyopl/opl_models` examples and an optional `opl_models` directory beside the
working-directory `.pyopl_session` file. The additional directory must be
scanned recursively and use the existing description/model/data triplet rules.

## Compatibility Rules

- Keep packaged examples available during default retrieval.
- Treat `<working directory>/opl_models` as optional; its absence is not an
  error and does not change packaged retrieval.
- Preserve an explicitly supplied `models_dir` as a single-root override so
  existing callers and focused tests remain deterministic.
- Preserve the existing triplet convention: rank non-empty `.txt`
  descriptions recursively, then pair each description with same-stem `.mod`
  and `.dat` files or the first sorted model and data files in that description's
  folder.
- Deduplicate normalized search roots and description paths so an overlapping
  working directory cannot produce duplicate candidates.
- Rank candidates from all active roots together before applying `top_k`.

## Implementation Steps

1. Generalize the RAG ranking helper to accept one or more model roots, ignore
   missing optional roots, recursively collect descriptions, and deduplicate
   candidates.
2. Add a shared strategy-base resolver for the packaged models root and the
   optional `Path.cwd() / "opl_models"` root used by default retrieval.
3. Keep explicit `models_dir` calls isolated to the supplied root and route all
   default GenAI strategies through the combined resolver.
4. Add focused `unittest` coverage for recursive working-directory discovery,
   missing optional directories, root/path deduplication, combined ranking, and
   unchanged explicit-root behavior.
5. Record the user-visible retrieval expansion in `CHANGELOG.md` and run the
   narrow GenAI helper tests followed by the broader affected suite if needed.

## Acceptance Criteria

- A complete triplet in `./opl_models` or any nested subfolder can be selected
  by every GenAI strategy that uses shared few-shot retrieval.
- Packaged and working-directory examples compete in one semantic ranking.
- Incomplete triplets are skipped during few-shot assembly exactly as before.
- Starting PyOPL where `./opl_models` does not exist continues without an error.
- Passing `models_dir` continues to search only that directory.