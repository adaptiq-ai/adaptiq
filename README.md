# AdaptIQ — Adaptive Optimization Framework for AI Agents

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/adaptiq.svg)](https://pypi.org/project/adaptiq)
[![Cost Saving](https://img.shields.io/badge/cost%20saving-30%25-brightgreen)](#benchmarks--methodology)
[![CO₂ Aware](https://img.shields.io/badge/CO%E2%82%82%20aware-yes-1abc9c)](#benchmarks--methodology)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![AdaptiQ Score](https://img.shields.io/badge/AdaptIQ-100%25-00f0ff.svg?style=flat-square)](https://benchmyagent.com)

**AdaptIQ — Adaptive Optimization Framework for AI Agents – Optimize behaviors, reduce resource usage, and accelerate learning with low-cognitive reinforcement tuning.**

---

## 🚀 Quick Overview

AdaptIQ uses reinforcement learning to automatically optimize your AI agents. Point it at your agent's logs, and it learns which actions work best in different situations, reducing costs by 30% while improving performance.

**Key Benefits:** Lower costs, better performance, data-driven optimization  
**Current Support:** CrewAI + OpenAI (more coming soon)

---

## 📋 Table of Contents
1. [🤔 Why AdaptiQ?](#-why-adaptiq)
2. [⚡ Quick Start](#-quick-start)
3. [✨ Features](#-features)
4. [🧠 How It Works (RL + Q-table)](#-how-it-works-rl--q-table)
5. [🏗️ Architecture](#️-architecture)
6. [📊 Reporting Mode](#-reporting-mode)
7. [🔮 What's Next](#-whats-next)
8. [🏆 Leaderboard (agents)](#-leaderboard-agents)
9. [🎯 Bench my agent](#-bench-my-agent)
10. [☁️ Upgrade Path → AdaptiQ FinOps Cloud](#️-upgrade-path--adaptiq-finops-cloud)
11. [🗺️ Roadmap](#️-roadmap)
12. [🤝 Community & Contributing](#-community--contributing)
13. [📄 License](#-license)

---

## 🤔 Why AdaptiQ?

AdaptIQ addresses the critical challenge of optimizing AI agent performance through intelligent, data-driven approaches. Our framework transforms the traditionally manual and error-prone process of agent tuning into a systematic, reinforcement learning-powered optimization workflow that learns from execution patterns and continuously improves agent behavior while reducing costs and resource consumption.

| Pain point | Traditional workaround | **AdaptiQ advantage** |
|------------|-----------------------|-----------------------|
| Prompt/agent errors discovered **after** expensive runs | Manual review, trial‑and‑error | Detects & fixes issues **before** execution |
| GPU/LLM cost spikes | Spreadsheet audits | Predicts € & CO₂ inline |
| No common prompt style | Word/PDF guidelines | Enforced by lint rules, autofixable |
| Dev ↔ FinOps gap | Slack + e‑mails | Same CLI / dashboard for both teams |

---

## 🎬 Demo Video

[![Demo Video](https://img.youtube.com/vi/ymNvLe73EhI/maxresdefault.jpg)](https://www.youtube.com/watch?v=ymNvLe73EhI)

*Click the image above to watch the demo video*

---

## ⚡ Quick Start

### 📋 Prerequisites

Before installing AdaptIQ, ensure you have:

- **Python 3.12+** - Required for AdaptIQ framework
- **CrewAI framework** - Set up and configured for your agents
- **OpenAI API key** - For LLM provider access
- **Windows OS** - Linux and Mac support not tested yet

### 📦 Installation

First, install UV package manager:

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

> ⚠️ **Note**: Linux and Mac support is not tested yet. We recommend using Windows for now.

Then activate your virtual environment and install AdaptIQ:

```bash
uv pip install adaptiq
```

For development mode:
```bash
uv pip install -e .
```

### 🪄 Quick Setup

**Non-interactive mode (recommended for first-time users):**

```bash
adaptiq wizard-headless --llm_provider openai --api_key your_api_key --prompt "wizard init <name_of_project>"
```

> 📝 **Note**: Only OpenAI provider is supported for the wizard assistant currently.

This will initialize a project with `adaptiq_config.yml` that you should configure.

### 🎮 Running AdaptIQ

**Interactive mode (DEV Environment):**
```bash
wizard validate config <path_of_config>
wizard start  # For first optimization
```

**Non-interactive mode (PROD Environment):**
```bash
adaptiq wizard-headless --llm_provider openai --api_key your_api_key --prompt "wizard execute <path_config>"
```

> 📝 **Important**: AdaptIQ currently supports only **CrewAI** as the agentic framework, **OpenAI** as the provider, and **GPT-4.1-mini** as the LLM for the workflow. Other models and frameworks have not been tested yet.

---

## ✨ Features

| Category | Free | Cloud (SaaS) |
|----------|------|--------------|
| 🧙 YAML wizard & validation | ✅ | ✅ |
| 🔍 Prompt & agent lint rules | ✅ | ✅ |
| 💰 **Pre‑run cost** | ✅ | ✅ |
| 🤖 RL‑powered optimisation suggestions | ✅ | ✅ |
| 🏭 Automatic optimisation at scale | — | ✅ |
| 💚 GPU‑spot arbitrage, ESG ledger | — | ✅ |
| 📊 Multi‑tenant FinOps dashboard | — | ✅ |

---

## 🧠 How It Works (RL + Q-table)

### 🎯 ADAPTIQ - Agent Development & Prompt Tuning Iteratively with Q-Learning

ADAPTIQ is a framework designed for the iterative improvement of AI agent performance through offline Reinforcement Learning (RL). Its primary goal is to systematically enhance an agent's guiding Configuration, focusing mainly on its Task Description (Prompt), by learning from the agent's past execution behaviors and incorporating user validation through an interactive "Wizard" process. It provides a structured, data-driven alternative to purely manual prompt engineering.

### 🚀 Vision and Goal

Adaptiq's mission is to optimize agent behavior by refining its core instructions (prompts/task descriptions). It achieves this by analyzing what an agent intended to do (from its prompt), what it actually did (from execution logs), and how effective those actions were (via a multi-faceted reward system). It is especially suited for agents using frameworks like CrewAI, LangChain, etc., where direct, real-time RL control is often impractical.

### 🔧 Key Concepts in Adaptiq

#### 🧩 State (s)
Represents the agent's situation at a specific step, defined by features like:

- **Current_SubTask**: The immediate objective (validated via Wizard)
- **Last_Action_Taken**: The previous validated ARIC strategic action
- **Last_Outcome**: The validated result of the previous action
- **Key_Context**: Accumulated relevant information (validated flags/data)

States are transformed into consistent, hashable representations for Q-table storage, potentially using generalization techniques.

#### 🎯 Action (a)
A selection from a predefined menu of discrete, strategic actions (e.g., Use_Tool_X, Action_Write_Content). Adaptiq maps observed log events to these predefined actions.

#### 📊 Q-Table
The core knowledge base: `Q(state_representation, action) → value`. It stores the learned long-term value of taking an action in a specific state, refined through the Adaptiq loop.

#### 🏆 Reward (R)
Calculated offline during/after trace reconciliation, guided by the Wizard and predefined rules. It incorporates:

- **Plan Adherence**: How well the actual execution matched the intended plan from prompt parsing
- **Execution Success (R_execution/internal)**: Based on tool outcomes, task progress, constraint adherence, and output quality from the logs
- **External Feedback (R_external - Optional)**: Real-world impact metrics (e.g., email open rates, conversions). To be implemented soon (now as external feedback only human feedback of user's evaluation of the agent after adaptiq optimization)

### 🛠️ Trace Analysis & Reconciliation Strategy

Adaptiq employs a multi-stage approach:

1. **Prompt Parsing (default-run)**: An LLM-powered module analyzes the agent's task description to extract the intended sequence of sub-tasks and actions

2. **Hypothetical State Generation (default-run)**: Uses the prompt parser's output to define idealized states and actions for heuristic Q-table initialization

3. **Log Parsing (first step of the run)**: Module parses raw execution logs to identify actual agent thoughts, tool calls, and outcomes

4. **Reconciliation (second step of the run)**: A central facilitates the alignment of the intended plan with actual execution. It allows the user to:
   - Validate/correct inferred states and actions
   - Confirm/override calculated rewards
   - Refine the understanding of the agent's behavior
   
   This produces the mapping data.

**Lightweight Q‑table examples:**

| State | Action | Q‑value |
|-------|--------|---------|
| `('InformationRetrieval_Company', 'None', 'None', 'company info')` | FileReadTool | **0.6** |
| `('InformationRetrieval_Lead', 'FileReadTool', 'Success_DataFound', 'company info lead name')` | LeadNameTool | **0.7** |
| `('ActionExecution_SendEmail', 'Write_Email_Body', 'Success_ActionCompleted', 'email sent lead')` | SendEmailTool | **0.7** |
| `('ResultFinalization', 'SendEmailTool', 'Success_ActionCompleted', 'email content final answer')` | Formulate_Final_Answer | **0.8** |

---

## 🏗️ Architecture

![AdaptIQ Architecture](./docs/assets/architecture.png)

---

## 📊 Reporting Mode

AdaptIQ offers flexible reporting options:

### 💾 Local Reporting
- Save optimization reports locally as Markdown
- Detailed performance metrics and recommendations
- Offline analysis capabilities

### 📧 Email Reports
- Send comprehensive reports to your email
- URL-based report sharing
- Real-time optimization insights (multiple)

> 📝 **Privacy Note**: When you provide your email in the YAML config, you acknowledge that we can process your data according to our privacy policy.

![UI Screenshot](./docs/assets/ui_screenshot.png)

---

## 🔮 What's Next

### 🎯 Upcoming Features

- **🔄 Support for More Models and Providers**: Expanding compatibility beyond OpenAI to include other LLM providers and models
- **🔄 Context Engineering Optimization**: Advanced prompt and context management through Q-learning
  - **📝 Prompt Optimization Workflow**: Implementing external rewards data type and tool tracking and evaluation
  - **📚 Q-Table Strategy for RAG Systems**: Learn which effective chunks reduce cost and increase speed
  - **🧠 Memory Layer Integration**: Q-table learns optimal context retention patterns - storing frequently accessed information states and reducing redundant retrievals through intelligent caching strategies
  - **📊 Knowledge Graph Integration**: Dynamic relationship mapping between entities and concepts for contextually-aware agent decisions
  - **🔌 External Context Integration APIs**: Seamless integration with CRM, databases, and third-party tools for enriched contextual understanding
  - **🛡️ Governance & Constraints**: 
    - **🚧 Guardrails**: Q-learning enforced safety boundaries and compliance rules
    - **🔐 Access Control**: Context-aware permission management
    - **📋 Policy Enforcement**: Automated adherence to organizational guidelines and industry standards
- **📱 Q-Table for Edge Devices**: Optimizing AI models performance to work better on resource-constrained devices

---

## 🏆 Leaderboard (agents) - Coming Soon

A comprehensive evaluation system to benchmark your agents based on specific KPIs (Health Learning Index HLI). Agents working on the same tasks can anonymously compare their performance, fostering continuous improvement and healthy competition in the AI agent community. This system helps maintain agent quality in production environments through continuous monitoring and benchmarking.

---

## 🎯 Bench my agent

**🚀 Build better AI agents. Use AdaptiQ and see your Agent Learning Health Index**

| ⚙️ | Benefit | Description |
|-------|---------|-------------|
| 🏅 **Social proof** | Public badge increases repo trust |
| 💰 **FinOps insight** | Cost €/k-token & CO₂/tkn surfaced instantly |
| 🔒 **Security gate** | Evaluator flags jailbreaks & PII leaks before prod |
| ♻️ **Continuous learning** | LHI tracks the agent's health across versions |

### 🎬 See the leaderboard in action

![Live demo: carrousel, live-feed et tri du leaderboard](./docs/assets/leaderboard.gif)

---

## ☁️ Upgrade Path → AdaptiQ FinOps Cloud

Need hands‑free optimisation across hundreds of projects? 🏢  
**AdaptiQ FinOps Cloud** adds:

* 🤖 Auto‑tuning RL in production  
* 💎 GPU‑spot arbitrage  
* 🌱 ESG & carbon ledger  
* 👥 Role‑based dashboards (Dev / FinOps / C‑suite)

**🆓 30‑day free trial** — migrate in **one CLI command**.

**Contact us for more information via email**

---

## 🗺️ Roadmap

| Quarter | Milestone |
|---------|-----------|
| **Q3 2025** | 🔄 Support for More Models and Providers & Cost optimization via LLM routing |
| **Q4 2025** | 🔄 Context Engineering Optimization: Memory Layer, Knowledge Graphs, External API Integration |
| **2026** | 📱 Edge SDK (quantised Q‑table <16 MB), 🛡️ Governance & Constraints framework, GPU‑Spot optimiser |

Vote or propose features in [`discussions/`](https://github.com/adaptiq-ai/adaptiq/discussions). 🗳️

---

## 🤝 Community & Contributing

We ❤️ PRs: bug fixes, lint rules, language support.  
See [`CONTRIBUTING.md`](./CONTRIBUTING.md).

* 💬 **Discord**: [**#adaptiq**](https://discord.com/invite/tZZUvcSY) (roadmap call 1st Tuesday) 
* 🐦 **X/Twitter**: [@adaptiq_ai](https://x.com/adaptiq_ai)

---

## 🧪 Beta Version Notice

AdaptIQ is currently in **beta version**. We welcome any issues, bug reports, or contributions to improve the framework! Your feedback helps us build a better tool for the AI agent community. 🙏

Please feel free to:
- 🐛 Report bugs via GitHub Issues
- 💡 Suggest new features
- 🤝 Contribute code improvements
- 📝 Improve documentation

Together, we can make AdaptIQ the best optimization framework for AI agents! 🚀

## 📚 Citation

If you use AdaptIQ in your research, project, or commercial application, please cite us:

### 📖 BibTeX

```bibtex
@software{adaptiq2025,
  title={AdaptIQ: Adaptive Optimization Framework for AI Agents},
  author={AdaptIQ AI Team},
  year={2025},
  url={https://github.com/adaptiq-ai/adaptiq},
  note={Adaptive Optimization Framework for AI Agents with Reinforcement Learning}
}
```
### 🔗 MLA Format

AdaptIQ AI Team. "AdaptIQ: Adaptive Optimization Framework for AI Agents." GitHub, 2025, https://github.com/adaptiq-ai/adaptiq.

### 📊 Research Papers

If you publish research using AdaptIQ, we'd love to hear about it! Please:
- 📧 Email us at research@getadaptiq.io
- 🐦 Tag us on Twitter [@adaptiq_ai](https://x.com/adaptiq_ai)
- 💬 Share in our Discord **#research** channel

---

## 📄 License

* **Code**: Apache 2.0 License 🆓
* **RL weights & FinOps Cloud components**: proprietary

© 2025 AdaptiQ AI. All trademarks belong to their respective owners.