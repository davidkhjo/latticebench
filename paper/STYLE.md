# Writing style checklist

Re-read this before starting any draft of the paper or blog. arXiv readers are
quick to dismiss text that reads as machine-generated, so the bar is: does this
sound like a person who knows the material wrote it under a deadline.

## Banned words and phrases

Do not use any of these:

- delve
- leverage (use "use")
- underscore / underscores (as in "this underscores")
- tapestry
- "it's worth noting"
- "in the realm of"
- "a testament to"
- seamless / seamlessly
- "furthermore" and "moreover" used to chain sentences
- "significantly" as a filler intensifier (only for statistical significance)

## Rules

- Prose over bullet lists in the paper body. Bullets are allowed in exactly two
  places: the contributions list and the enumerated protocol steps. Nowhere else.
- At most one em-dash per paragraph.
- Concrete numbers, not vague claims. "roughly 80 percent solved" beats "largely
  solved" when a number exists.
- First-person plural, active voice. "We generate" not "instances are generated"
  where a subject is natural.
- Past tense for work that was done; present tense for what the artifact does.
- One idea per paragraph, topic sentence first.
- Every claim about a competitor benchmark carries a citation and a specific,
  checkable difference. No unsupported "unlike prior work" claims.

## Factual guardrails

- No AI-assistance fingerprints anywhere: no "generated with", no co-author
  trailers, no mention of tooling. The author is David alone.
- Tars appears at most once, only in Related Work, described precisely as an
  EBM-as-ranker inside an LLM-orchestration pipeline (recursive decomposition,
  parallel Lean, premise selection). Its 622/672 PutnamBench figure is that
  pipeline's score, not a pure EBM prover score. No Tars number ever appears in a
  LatticeBench results table, and no LatticeBench result is implied to come from
  Tars.
- Mark every placeholder experiment number as TODO. Do not invent results.
