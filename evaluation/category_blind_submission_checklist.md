# Category-Blind Submission Checklist

## Goal

Turn the current `category-aware` evaluation path into a `category-blind`
submission path.

`qa.category` should only be used for:

- evaluation grouping
- logging
- reporting metrics

`qa.category` should not be used for:

- retrieval routing
- prompt selection
- abstain gating
- answer postprocessing

## Why This Is Needed

The current pipeline still reads the benchmark category before generating the
answer, so it is not a fully blind evaluation setup.

Current generation-time dependencies on `category`:

- Retrieval routing: [runsingle.py](E:/Benchmark/Mem-R-benchmark/runsingle.py#L195)
- Prompt selection: [runsingle.py](E:/Benchmark/Mem-R-benchmark/runsingle.py#L1028)
- Postprocessing: [runsingle.py](E:/Benchmark/Mem-R-benchmark/runsingle.py#L880)
- Evaluation loop passes `qa.category` into generation:
  [runsingle.py](E:/Benchmark/Mem-R-benchmark/runsingle.py#L1467)

## Minimal Change List

### 1. Stop passing `qa.category` into answer generation

Change:

- `answer_question(self, question: str, category: int)`

To something like:

- `answer_question(self, question: str)`

Then update the call site in the evaluation loop so generation only sees the
question text.

Current call site:

- [runsingle.py](E:/Benchmark/Mem-R-benchmark/runsingle.py#L1467)

Keep `qa.category` only after prediction for metric bucketing and reporting.

### 2. Make routing fully question-driven

Change:

- `_classify_question_mode(self, question: str, category: int)`

To:

- `_classify_question_mode(self, question: str)`

Current routing function:

- [runsingle.py](E:/Benchmark/Mem-R-benchmark/runsingle.py#L195)

Current direct category branches to remove from submission mode:

- `if category == 3`
- `if category == 5`
- `if category == 2 or llm_task == "temporal"`
- `if category == 4 or llm_task == "open_domain"`
- `if category == 1 or llm_task in (...)`

Recommended submission rule:

- route only by `attrs` and `llm_task`
- do not use benchmark category as a hard switch

### 3. Remove category-based fallback from LLM routing

Current call:

- `_llm_route_question(question, category)`

If keeping a fallback path, the fallback should come from question heuristics,
not from the benchmark label.

Safer options:

- fallback to heuristic attributes only
- fallback to `"factoid"` as a neutral default
- fallback to a heuristic mapping derived from the question text

Avoid:

- using `fallback_category` from the dataset during submission

### 4. Move prompt selection from `category` to inferred answer style

Current prompt selection still uses:

- `if category == 5`
- `elif category == 2`
- `elif category == 3`
- `elif category == 1`

Current prompt block:

- [runsingle.py](E:/Benchmark/Mem-R-benchmark/runsingle.py#L1074)

Submission version should select prompts from inferred route output, for example:

- `answer_style = temporal`
- `answer_style = reasoning`
- `answer_style = extractive_fact`
- `answer_style = adversarial_abstain`

Practical approach:

- let `_classify_question_mode()` return an extra field such as
  `answer_style`
- in `answer_question()`, branch on `answer_style` instead of `category`

### 5. Make Cat5 abstain gate route-driven, not label-driven

Current gate:

- [runsingle.py](E:/Benchmark/Mem-R-benchmark/runsingle.py#L1051)

For submission mode, keep the abstain gate only when the inferred route says the
question is unsupported, for example:

- `enable_abstain_gate = True`

Do not tie the gate to benchmark category.

### 6. Make postprocessing category-blind

Current postprocessing still uses `category`:

- [runsingle.py](E:/Benchmark/Mem-R-benchmark/runsingle.py#L880)
- [runsingle.py](E:/Benchmark/Mem-R-benchmark/runsingle.py#L905)

Submission version should use:

- question text
- inferred route
- answer style

Instead of:

- benchmark category

Practical change:

- `postprocess_answer(question, answer_text, raw_context="", answer_style="...")`

### 7. Keep category only for evaluation and analysis

These uses are acceptable:

- metric aggregation by category
- per-category logs
- result reporting

Examples:

- [runsingle.py](E:/Benchmark/Mem-R-benchmark/runsingle.py#L1462)
- [runsingle.py](E:/Benchmark/Mem-R-benchmark/runsingle.py#L1520)
- [runsingle.py](E:/Benchmark/Mem-R-benchmark/runsingle.py#L1529)

## Recommended Submission Split

Keep two modes:

- `category-aware`
  Diagnostic mode for upper-bound analysis and ablations.
- `category-blind`
  Real submission mode. Only the question is visible at inference time.

Best practice:

- default to `category-blind`
- keep `category-aware` behind an explicit debug flag

## Fast Acceptance Checklist

Before submission, verify all of the following are true:

- `answer_question()` no longer receives `qa.category`
- `_classify_question_mode()` no longer receives `category`
- `_llm_route_question()` no longer uses dataset category as fallback
- prompt selection no longer branches on benchmark category
- abstain gate no longer checks `category == 5`
- postprocessing no longer branches on benchmark category
- `qa.category` is only used after prediction for metrics and reporting

## Practical Standard

If the model can only see `question` at answer time, the setup is category-blind.

If the model sees benchmark `category` before retrieval, prompting, or
postprocessing, the setup is still category-aware.
