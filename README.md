# AdaptIQ — the learning layer for loop engineering

[![PyPI](https://img.shields.io/pypi/v/adaptiq.svg)](https://pypi.org/project/adaptiq)
[![Python](https://img.shields.io/badge/python-3.11%2B-1f425f.svg)](https://www.python.org/)
[![Tests](https://github.com/adaptiq-ai/adaptiq/actions/workflows/tests.yml/badge.svg)](https://github.com/adaptiq-ai/adaptiq/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Benchmark DOI](https://img.shields.io/badge/benchmark-10.5281%2Fzenodo.16876743-blue)](https://doi.org/10.5281/zenodo.16876743)

**AdaptIQ learns the decisions inside your agent's loop — which action to take next, when to stop, when to escalate — from execution traces, and keeps the resulting policy readable, versioned and reversible.**

No gradient updates, no fine-tuning. The policy is a Q-table over (state, action) that you can read, diff and roll back. It is learned offline, with a human validation step, and promoted through paired evaluation.

Built for **bounded, repetitive agent workflows** — document processing, back-office automation, support flows — where predictability and auditability matter more than raw capability. Not for open-ended coding agents.

> **Read. Bound. Correct.** Read what your agent did against what it was meant to do. Bound how far it can go. Correct its policy in cycles you approve.

*Loop engineering* is the practice of deliberately designing how an agent iteratively acts, checks and stops ([IBM, 2026](https://www.ibm.com/think/topics/loop-engineering)). Every definition of it so far describes loops designed by hand. AdaptIQ makes the loop learnable — and keeps it legible.

---

## What it does today (v0.12)

- **Learns a tabular policy from agent traces.** State = current subtask, last action, last outcome, key context. Actions = a discrete menu of strategic actions. Reward = plan adherence + execution success, plus optional external feedback. Offline Q-learning, tabular updates, no model weights touched.
- **Reconciles intention with execution.** The plan extracted from your prompts is aligned with the actual execution logs, and optional human feedback is folded into the post-run analysis. This is where ground truth — and rewards — come from. A full human validation step, where states and rewards are reviewed and corrected one by one, is a roadmap item, not something the CLI does today.
- **Turns the learned policy into better task descriptions** for the next run, and applies it at runtime through a decision engine (the setup used in the benchmark).
- **Measures every run**: tokens, cost, latency. Reports are written locally by default; they are uploaded only if you put an e-mail address in the config.

**Supported today:** CrewAI as the agent framework (pinned range, see below) · OpenAI `gpt-4.1` and `gpt-4.1-mini` · Python 3.11+ · verified on Windows; Linux and macOS are expected to work and Linux is covered by CI from this release on.

## Measured results

Paired benchmark — same agent, same targets, pinned model IDs, with and without AdaptIQ. Full method, data and code: [adaptiq-benchmark](https://github.com/adaptiq-ai/adaptiq-benchmark) (DOI [10.5281/zenodo.16876743](https://doi.org/10.5281/zenodo.16876743)).

| Metric | Baseline | AdaptIQ | Δ | p-value |
|---|---|---|---|---|
| Token dispersion (std. dev.) | ≈ 1 278 | ≈ 457 | **÷ 2.8** | — |
| Latency (s) | 13.94 | 11.85 | −15.0 % | < 0.001 |
| Cost (USD / task) | 0.0099 | 0.0086 | −13.6 % | < 0.001 |
| Tokens | 8 347 | 7 459 | −10.6 % | 0.366 (n.s.) |
| Quality (CLIP) | 91.18 | 91.01 | −0.17 (target ≥ 0) | — |

The headline is the first line, not the cost line: the same agent becomes **2.8× more predictable** at near-equal quality. The quality delta missed its ≥ 0 target by 0.17 points; it is reported as measured.

## Quick start

```bash
pip install adaptiq            # or: uv pip install adaptiq

# 1. Scaffold a project (CrewAI is the only supported template today)
#    This creates ./my_project/src/my_project/
adaptiq init --name my_project --template crew-ai --path ./my_project

# 2. Edit the generated config (project name, model, embeddings, framework adapter,
#    and the e-mail address that opts you in to uploaded reports — empty means local only)
#    ./my_project/src/my_project/config/adaptiq_config.yml

# 3. Validate the configuration
adaptiq validate --config_path ./my_project/src/my_project/config/adaptiq_config.yml --template crew-ai

# 4. Run your CrewAI agent as usual — AdaptIQ instruments the run, learns, and writes the reports
```

Development install:

```bash
git clone https://github.com/adaptiq-ai/adaptiq.git && cd adaptiq
python -m venv .venv && source .venv/bin/activate
pip install -e . && pip install pytest
pytest tests
```

## How it works

1. **Trace** — the agent's execution is logged step by step (thoughts, actions, tool results, outcomes).
2. **Reconcile** — the intended plan is aligned with the trace; a human validates states, actions and rewards.
3. **Learn** — a Q-table over (state, action) is updated from the validated trace. Tabular, gradient-free.
4. **Apply** — the policy shapes the next run: better task descriptions offline, decisions at runtime.
5. **Evaluate** — paired runs, pinned models, statistical significance, before any policy is promoted.

Details: [ARCHITECTURE.md](ARCHITECTURE.md).

## Roadmap — loop engineering, learned

One engine, four verbs. Each verb opens only when the previous one has passed its exit benchmark. Full plan with gates and kill criteria: [ROADMAP.md](ROADMAP.md).

| Verb | What it learns | Status |
|---|---|---|
| `audit` | Where execution diverged from intention, and why — from standard traces (OpenTelemetry GenAI, LangSmith exports, LiteLLM logs, DeepSeek Harness session logs) | planned · Q4 2026 |
| `route` / `context` | Which model to call, whether to call at all, when to compact context — as a LiteLLM plugin with `observe` / `suggest` / `act` modes | planned · Q4 2026 – Q1 2027 |
| `loop` | Step budgets, stop and escalate-to-human decisions — as harness plugins (DeepSeek Harness first, then LangGraph, CrewAI) | planned · 2027 |
| governance | Versioned policies, diff and rollback, promotion gates, audit pack | planned · 2027 |

Principles that will not change: the policy stays a readable table; learning stays offline with a human in the loop; an action is promoted only if its quality delta is ≥ 0 at p < 0.05; the policy is learned over abstract actions, so it survives a change of model or provider.

## Limits

- Tabular learning fits bounded, repetitive workflows. It does not fit open-ended agents.
- The state abstraction is hand-designed; generalisation across unrelated tasks is limited.
- Today's integration is CrewAI-only. Framework-agnostic trace ingestion is the first roadmap item.
- Dependencies are pinned to the ranges the code was validated against: `crewai < 0.178`, because later releases drop an API the CrewAI logger relies on, and `langchain < 1.0`. Supporting the current lines of both is the first item after the roadmap's `audit` verb.

## Contributing

Issues and pull requests are welcome — bug reports, benchmark reproductions and trace samples from real workflows are the most useful contributions right now. See [CONTRIBUTING.md](CONTRIBUTING.md). Updates: [@adaptiq_ai](https://x.com/adaptiq_ai).

## Citation

First released in July 2025; cite the version you used.

```bibtex
@software{amri_adaptiq_2026,
  author  = {Amri, Wassim},
  title   = {AdaptIQ: a learned control layer for AI agents},
  year    = {2026},
  version = {0.12.9},
  url     = {https://github.com/adaptiq-ai/adaptiq},
  note    = {Benchmark: doi:10.5281/zenodo.16876743}
}
```

## License

Code: [Apache 2.0](LICENSE). Policies (Q-tables) learned from your own traces belong to you.
