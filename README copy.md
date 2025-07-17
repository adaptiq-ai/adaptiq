# AdaptIQ — RL-Powered Benchmark to Optimize Your AI Agents

[![CI](https://img.shields.io/github/actions/workflow/status/adaptiq-ai/adaptiq/ci.yml?label=CI)](https://github.com/adaptiq-ai/adaptiq/actions)
[![PyPI](https://img.shields.io/pypi/v/adaptiq.svg)](https://pypi.org/project/adaptiq)
[![cost saving](https://img.shields.io/badge/cost%20saving-30%25-brightgreen)](#benchmarks--methodology)
[![CO₂ aware](https://img.shields.io/badge/CO%E2%82%82%20aware-yes-1abc9c)](#benchmarks--methodology)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![AdaptiQ Score](https://img.shields.io/badge/AdaptIQ-100%25-00f0ff.svg?style=flat-square)](https://benchmyagent.com) [![AdaptiQ Score](https://img.shields.io/badge/AdaptIQ%20Eval-100%25-00f0ff.svg?style=flat-square)](https://www.google.com)





**Build better AI agents, spend less, emit zero surprises — powered by Reinforcement Learning.**

---

## Table of Contents
1. [Why AdaptiQ ?](#why-adaptiq-)
2. [Quick Start (wizard)](#quick-start-wizard)
3. [Features](#features)
4. [How It Works (RL + Q-table)](#how-it-works-rl-qtable)
5. [CLI Reference](#cli-reference)
6. [GitHub Action](#github-action)
7. [API Embed](#api-embed)
8. [Leaderboard (agents)](#leaderboard-agents)
9. [Bench my agent (LHI)](#bench-my-agent)
10. [Upgrade Path → AdaptiQ FinOps Cloud](#upgrade-path--adaptiq-finops-cloud)
11. [Roadmap](#roadmap)
12. [Community & Contributing](#community--contributing)
13. [Telemetry & Privacy](#telemetry--privacy)
14. [License](#license)

---

## Why AdaptiQ ?

| Pain point | Traditional workaround | **AdaptiQ advantage** |
|------------|-----------------------|-----------------------|
| Prompt/agent errors discovered **after** expensive runs | Manual review, trial‑and‑error | Detects & fixes issues **before** execution |
| GPU/LLM cost spikes | Spreadsheet audits | Predicts € & CO₂ inline |
| No common prompt style | Word/PDF guidelines | Enforced by lint rules, autofixable |
| Dev ↔ FinOps gap | Slack + e‑mails | Same CLI / dashboard for both teams |

---

## Quick Start (wizard)

**Install**

```bash
# SPDX-License-Identifier: Apache-2.0
pip install adaptiq
```

**Discover capabilities**

```bash
wizard info
```

**Initialise a project**

```bash
wizard init                 # interactive wizard -> generates adaptiq.yml
```

**Validate your configuration**

```bash
wizard validate config ./adaptiq.yml
```

**Optimise & run**

```bash
wizard start                # dry‑run optimisation suggestions, then execute
```

---

## Features

| Category | Free | Cloud (SaaS) |
|----------|------|--------------|
| YAML wizard & validation | ✅ | ✅ |
| Prompt & agent lint rules | ✅ | ✅ |
| **Pre‑run cost** | ✅ | ✅ |
| RL‑powered optimisation suggestions | ✅ | ✅ |
| Automatic optimisation at scale | — | ✅ |
| GPU‑spot arbitrage, ESG ledger | — | ✅ |
| Multi‑tenant FinOps dashboard | — | ✅ |

---

## How It Works (RL + Qtable)

1. **State** = key attributes of a pending run  
   `(model, context_len, temperature, batch_size, …)`

2. **Action** = change one or more attributes  
   e.g. `swap_model`, `shrink_context`, `enable_cache`.

3. **Reward** = cost savings (€) weighted by quality delta (BLEU/ROUGE).

Lightweight **Q‑table** sample :

| State (hash) | Action | Q‑value (expected € saved) |
|--------------|--------|---------------------------|
| `0x4fa9c3` | shrink_context | **0.14** |
| `0x4fa9c3` | swap_to_gpt35 | 0.11 |
| `0x2a8b7e` | prune_prompts | **0.18** |


---

## CLI Reference

| Command | Description |
|---------|-------------|
| `wizard info` | Show version and current providers/models support |
| `wizard init` | Interactive setup → generates `adaptiq.yml` |
| `wizard validate config <path>` | Validate a YAML config locally |
| `wizard start` | Optimise (dry‑run) then execute a run in prod |

---

## GitHub Action

```yaml
name: AdaptiQ Validate & Optimise
on:  [push, pull_request]
jobs:
  finops:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install AdaptiQ
        run: pip install adaptiq
      - name: Validate config and enforce budget
        run: wizard validate config adaptiq.yml --max-cost 0.50
```

---

## API Embed

```python
# SPDX-License-Identifier: Apache-2.0
from adaptiq import lint_file, apply_suggestions

report = lint_file("welcome_email.prompt", max_cost=0.25)
print(report.issues)
fixed = apply_suggestions(report)
```

---

## Leaderboard (agents)

Benchmark your agent in **90 s** and show off the score.

### Quick Eval (local)

```bash
pip install adaptiq-eval
export ADAPTIQ_API_KEY=<your_key>
adaptiq-eval quick               # 10 % prompts, returns run.zip + score
```
The CLI uploads a run archive to BenchMyAgent and returns:

```bash
✔ Score 92.4 %   ✔ LHI 76   ✔ Qual. 90   ✔ Cost 0.21 €/k
Markdown badge copied to clipboard!
```

[![AdaptiQ Score – Gold](https://img.shields.io/badge/AdaptiQ-90.1%25-00f0ff.svg?style=flat-square&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI%2BPHBhdGggZmlsbD0iI0ZGRDcwMCIgZD0iTTQgM2gxNnYyaC0xdjJhNyA3IDAgMDEtNiA2LjkzVjE2aDR2Mkg3di0yaDR2LTIuMDdBNyA3IDAgMDE1IDdWNUg0VjN6bTIgMnYyYTUgNSAwIDAwNCA0LjlWNUg2em04IDB2Ni45YTUgNSAwIDAwNC00LjlWNWgtNHoiLz48L3N2Zz4%3D)](https://www.google.com) [![AdaptiQ Score – Silver](https://img.shields.io/badge/AdaptiQ-80.1%25-00f0ff.svg?style=flat-square&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI%2BPHBhdGggZmlsbD0iI0MwQzBDMCIgZD0iTTQgM2gxNnYyaC0xdjJhNyA3IDAgMDEtNiA2LjkzVjE2aDR2Mkg3di0yaDR2LTIuMDdBNyA3IDAgMDE1IDdWNUg0VjN6bTIgMnYyYTUgNSAwIDAwNCA0LjlWNUg2em04IDB2Ni45YTUgNSAwIDAwNC00LjlWNWgtNHoiLz48L3N2Zz4%3D)](https://www.google.com)
[![AdaptiQ Score – Bronze](https://img.shields.io/badge/AdaptiQ-70.1%25-00f0ff.svg?style=flat-square&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI%2BPHBhdGggZmlsbD0iI0NEN0YzMiIgZD0iTTQgM2gxNnYyaC0xdjJhNyA3IDAgMDEtNiA2LjkzVjE2aDR2Mkg3di0yaDR2LTIuMDdBNyA3IDAgMDE1IDdWNUg0VjN6bTIgMnYyYTUgNSAwIDAwNCA0LjlWNUg2em04IDB2Ni45YTUgNSAwIDAwNC00LjlWNWgtNHoiLz48L3N2Zz4%3D)](https://www.google.com)


## Bench my agent

**Build better AI agents. Use AdaptiQ and see your Agent Learning Health Index**

| ⚙️ | Benefit | |  |  |  |
|-------|---------------:|-----------:|----:|------------------:|------------:|
| 🏅 Social proof | Public badge increases repo trust.   
| 💰 FinOps insight  | 	Cost €/k-token & CO₂/tkn surfaced instantly. 
| 🔒 Security gate | 	Evaluator flags jailbreaks & PII leaks before prod. 
| ♻️ Continuous learning | LHI tracks the agent’s health across versions.

## See the leaderboard in action 🚀

<p align="center">
  <!-- remplace XYZ par l’URL publique Loom si tu veux le plein écran HD -->
  <a href="https://www.loom.com/share/XYZ" target="_blank">
    <img src="./docs/assets/leaderboard.gif" width="720"
         alt="Live demo : carrousel, live-feed et tri du leaderboard" />
  </a>
</p>


---

## Upgrade Path → AdaptiQ FinOps Cloud

Need hands‑free optimisation across hundreds of projects ?  
**AdaptiQ FinOps Cloud** adds:

* Auto‑tuning RL in production  
* GPU‑spot arbitrage  
* ESG & carbon ledger  
* Role‑based dashboards (Dev / FinOps / C‑suite)

30‑day free trial — migrate in **one CLI command**.

---

## Roadmap

| Quarter | Milestone |
|---------|-----------|
| **Q3 2025** | OpenTelemetry exporter, multi‑language prompts |
| **Q4 2025** | Edge SDK (quantised Q‑table <16 MB) |
| **2026** | GPU‑Spot optimiser & carbon‑aware scheduler |

Vote or propose features in [`discussions/`](https://github.com/adaptiq-ai/adaptiq/discussions).

---

## Community & Contributing

We ❤️ PRs : bug fixes, lint rules, language support.  
See [`CONTRIBUTING.md`](./CONTRIBUTING.md).

* Discord : **#adaptiq** (roadmap call 1st Tuesday)  
* X/Twitter : [@adaptiq_ai](https://x.com/adaptiq_ai)

---

## Telemetry & Privacy

*Opt‑in only.*  
Data : anonymised project hash, rule IDs, optional cost/CO₂ deltas.  
Disable via `--no-telemetry` or env `ADAPTIQ_OPTOUT=1`.

Vectors older than 30 days are pruned unless cloud sync under DPA.

---

## License

* Code : **Apache‑2.0**  
* RL weights & FinOps Cloud components : proprietary.

© 2025 AdaptiQ AI. All trademarks belong to their respective owners.
