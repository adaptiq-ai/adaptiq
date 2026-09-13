# AdaptIQ roadmap — loop engineering, learned

AdaptIQ is a learning layer for agent loops: it learns, from execution traces, the decisions an agent takes inside its loop — verify, stop, escalate, compact, route — and keeps the policy readable, versioned and reversible.

**Scope, stated first:** bounded, repetitive agent workflows. Not open-ended agents.

**One engine, four verbs.** The engine (Q-table, intention/execution reconciliation, paired evaluation) is built once. Verbs are added one at a time. A stage opens only when the previous stage has passed its exit benchmark. Every stage ships a public artifact — proof, never promise.

| Stage | Window | Deliverable | Exit benchmark | Kill criterion |
|---|---|---|---|---|
| **0 — Credibility** | Sept. 2026 | Repositioned README, cleaned repo, pinned dependencies, Linux CI, named authorship | A newcomer clones and runs the demo on Linux in under 10 minutes | None — prerequisite |
| **1 — `audit`** (traceable) | Q4 2026 | Offline CLI: standard traces (OpenTelemetry GenAI, LangSmith exports, LiteLLM logs, DeepSeek Harness session logs) → plan-adherence score per run, divergence list, human validation in the CLI, counterfactual report | On 200 runs of a public workload: precision and recall ≥ 0.8 against human labels; first report under 5 minutes from clone | If 3 out of 5 users do not say "I didn't know my agent was doing that", the report presentation is wrong — not the mechanism |
| **2 — `route` / `context`** (learning) | Q4 2026 – Q1 2027 | LiteLLM plugin with `observe` / `suggest` / `act` modes: model choice, call-or-skip, learned context compaction | 500 public tasks of mixed difficulty, 3 models including a local one, 2 baselines (single large model; static rule router): lower variance than rules, cost ≤ rules, quality Δ ≥ 0 at p < 0.05. On 50-turn sessions: context −30 % at quality ≥ baseline | If static rules are not beaten on variance, learning adds nothing at the proxy — move to stage 3 |
| **3 — `loop`** (bounded) | 2027 | Harness plugins (DeepSeek Harness first, then LangGraph, CrewAI): learned step budgets, stop, escalate-to-human | Public agentic benchmark where runaway loops are observable: success ≥ baseline, steps −20 %, over-budget runs −50 %, escalations approved by humans ≥ 80 %. Published with a DOI | If success drops as steps drop, the policy learns to stop too early — the reward is miscalibrated |
| **4 — governance** (governable) | 2027 | Versioned policies, diff and rollback, promotion gates, audit pack generated from the table and the evaluations | Policy learned on model A, applied after switching to model B: ≥ 80 % of performance retained without relearning; rollback under one minute; an external reviewer explains 10 random decisions from the table alone | If an external reviewer cannot explain the decisions, "readable" is false — the whole positioning falls |
| **Horizon — governed memory** | after stage 4 | Episodic, semantic, negative and calibration memory planes as a fifth verb | Temporal validity of facts, staleness reduction | Does not open before the four verbs hold |

## Principles that do not change

- The policy is a table: read it, diff it, roll it back. Never weights.
- Learning is offline, with a human validation step. An action is promoted only if its quality delta is ≥ 0 at p < 0.05, measured on paired runs with pinned models.
- The policy is learned over abstract actions, not over a model: it survives a change of model or provider.
- Cost is a reward term, never the pitch. Predictability and auditability are.
- AdaptIQ complements enforcement platforms and gateways; it does not replace them. It reads their traces and plugs into their hooks.

## What stays, what changes, what goes (from v0.12)

- **Stays:** the Q-table engine, the reconciliation with human validation, the paired benchmark method, Apache-2.0, the PyPI package.
- **Changes:** FinOps becomes a reward term; the image-generation workload gives way to relatable workloads (support, QA, documents); CrewAI-only ingestion gives way to standard traces; authorship is named.
- **Goes:** the "prompt optimisation" positioning, the SaaS sections, working files at the repository root.
