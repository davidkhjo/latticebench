"""A local open-model client, so the LLM baseline needs no API key.

Uses mlx-lm on Apple Silicon (fast, native). The returned object is a plain
``str -> str`` callable, so it drops straight into ``LLMModel``.
"""

from __future__ import annotations

from latticebench.dataset import render_prompt
from latticebench.schema import Puzzle

DEFAULT_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"


def solve_template(puzzle: Puzzle) -> str:
    """Prompt the model to reason, then emit only the answer JSON on the last line."""
    return (
        render_prompt(puzzle)
        + "\n\nReason step by step. On the final line, after 'ANSWER:', output only "
        "the JSON object and nothing else."
    )


class MLXClient:
    def __init__(
        self, model: str = DEFAULT_MODEL, *, max_tokens: int = 1600, temp: float = 0.0
    ) -> None:
        from mlx_lm import generate, load

        self._generate = generate
        loaded = load(model)
        self.model, self.tokenizer = loaded[0], loaded[1]
        self.model_id = model
        self.max_tokens = max_tokens
        self.temp = temp

    def _sampler(self):
        from mlx_lm.sample_utils import make_sampler

        return make_sampler(temp=self.temp)

    def __call__(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception:
            text = prompt
        return self._generate(
            self.model,
            self.tokenizer,
            text,
            max_tokens=self.max_tokens,
            sampler=self._sampler(),
            verbose=False,
        )


def short_name(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1]
