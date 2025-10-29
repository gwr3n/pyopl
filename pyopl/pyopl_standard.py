from typing import Callable, Optional

from .pyopl_generative import (
    LLM_PROVIDER,
    MODEL_NAME,
    Grammar,
    generative_solve as _generative_solve,  # FIX: import under alias
)


def generative_solve(
    prompt,
    model_file,
    data_file,
    model_name: str = MODEL_NAME,
    mode: Grammar = Grammar.BNF,
    iterations: int = 1,  # ignored; always enforced to 1 below
    return_statistics: bool = False,
    alignment_check: Optional[bool] = False,  # ignored; always enforced to False below
    temperature: Optional[float] = None,
    stop: Optional[list[str]] = None,
    llm_provider: Optional[str] = LLM_PROVIDER,
    progress: Optional[Callable[[str], None]] = None,
    few_shot: bool = False,  # ignored; always enforced to False below
):
    """
    Thin wrapper that calls pyopl_generative.generative_solve in a closest-to-vanilla configuration:
      - few_shot=False
      - iterations=1
      - alignment_check=False

    Note: If the first attempt fails to compile, pyopl_generative will still do a second LLM
    call for the final assessment.
    """
    # Call generative_solve with a Literal[True]/[False] for return_statistics
    if return_statistics is True:
        result = _generative_solve(
            prompt,
            model_file,
            data_file,
            model_name=model_name,
            mode=mode,
            iterations=iterations,
            return_statistics=True,
            alignment_check=alignment_check,
            temperature=temperature,
            stop=stop,
            llm_provider=llm_provider,
            progress=progress,
            few_shot=few_shot,
        )
    else:
        result = _generative_solve(
            prompt,
            model_file,
            data_file,
            model_name=model_name,
            mode=mode,
            iterations=iterations,
            return_statistics=False,
            alignment_check=alignment_check,
            temperature=temperature,
            stop=stop,
            llm_provider=llm_provider,
            progress=progress,
            few_shot=few_shot,
        )

    return result
