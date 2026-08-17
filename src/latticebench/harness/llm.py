"""The language-model adapter.

An ``LLMModel`` wraps any callable ``str -> str`` (a completion function). The
prompt is the puzzle's natural-language statement, and the parser reads back the
JSON grid the prompt asks for. Thin Anthropic and OpenAI clients are provided
behind the ``[llm]`` extra, but any callable works.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

from latticebench.dataset import render_prompt
from latticebench.harness.base import Prediction
from latticebench.schema import Assignment, Domain, Puzzle

Client = Callable[[str], str]

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(text: str) -> Any:
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    match = _JSON_RE.search(text)
    if match:
        try:
            return json.loads(match.group(0))
        except ValueError:
            return None
    return None


def parse_grid_response(text: str, domain: Domain) -> Assignment | None:
    """Parse a ``{attribute: [value_per_house]}`` JSON object into an assignment."""
    obj = _extract_json_object(text)
    if not isinstance(obj, dict):
        return None
    pos: Assignment = {}
    for a in domain.attributes:
        values = obj.get(a.name)
        if not isinstance(values, list) or len(values) != domain.n:
            continue
        row: dict[str, int] = {}
        for house, value in enumerate(values):
            if isinstance(value, str) and value in a.values:
                row[value] = house
        if len(row) == domain.n:
            pos[a.name] = row
    return pos or None


class LLMModel:
    """Present a puzzle to a completion callable and parse its answer."""

    def __init__(
        self,
        client: Client,
        name: str,
        *,
        template: Callable[[Puzzle], str] = render_prompt,
        parser: Callable[[str, Domain], Assignment | None] = parse_grid_response,
    ) -> None:
        self.client = client
        self.name = name
        self.template = template
        self.parser = parser

    def predict(self, puzzle: Puzzle) -> Prediction:
        prompt = self.template(puzzle)
        t0 = perf_counter()
        text = self.client(prompt)
        elapsed = perf_counter() - t0
        assignment = self.parser(text, puzzle.domain)
        return Prediction(assignment=assignment, solve_time=elapsed, raw=text)


def export_lm_eval_task(
    records: list[Any], out_dir: str | Path, *, task_name: str = "latticebench"
) -> Path:
    """Write a minimal lm-evaluation-harness task (a JSONL of prompts and a YAML)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_path = out / f"{task_name}.jsonl"
    with data_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(
                json.dumps({"id": rec.id, "prompt": rec.nl_prompt, "solution": rec.solution}) + "\n"
            )
    yaml_path = out / f"{task_name}.yaml"
    yaml_path.write_text(
        f"task: {task_name}\n"
        f"dataset_path: json\n"
        f"dataset_kwargs:\n  data_files: {task_name}.jsonl\n"
        f"output_type: generate_until\n"
        f"doc_to_text: '{{{{prompt}}}}'\n"
        f"doc_to_target: '{{{{solution}}}}'\n"
        f"generation_kwargs:\n  until: []\n",
        encoding="utf-8",
    )
    return yaml_path


class AnthropicChat:
    """A minimal Anthropic completion client (requires ``anthropic``)."""

    def __init__(self, model: str, *, max_tokens: int = 2048, **kwargs: Any) -> None:
        import anthropic

        self._client = anthropic.Anthropic(**kwargs)
        self.model = model
        self.max_tokens = max_tokens

    def __call__(self, prompt: str) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in msg.content if getattr(block, "type", None) == "text"
        )


class OpenAIChat:
    """A minimal OpenAI completion client (requires ``openai``)."""

    def __init__(self, model: str, **kwargs: Any) -> None:
        import openai

        self._client = openai.OpenAI(**kwargs)
        self.model = model

    def __call__(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""
