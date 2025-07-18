# Contributing to **AdaptiQ**

First – **thank you** for taking the time to contribute!\
This guide explains the ground rules for participating in the project, the coding standards we follow, and the process for submitting pull requests.

---

## 1 – Ground Rules

| Rule                                                                       | Why it matters                              |
| -------------------------------------------------------------------------- | ------------------------------------------- |
| **Be respectful.** Follow our [Code of Conduct](CODE_OF_CONDUCT.md).       | Healthy discussion and inclusive community. |
| **Search first.** Check open issues/PRs before filing a new one.           | Avoids duplicates and wasted effort.        |
| **Small is beautiful.** Prefer focused PRs over large “catch-all” changes. | Easier reviews and faster merges.           |

---

## 2 – How to Contribute

### 2.1  Reporting Bugs or Requesting Features

1. Open a **GitHub Issue**.
2. Use the **“Bug”** or **“Feature”** template and fill in *all* required sections.
3. Provide reproducible steps or a minimal code sample whenever possible.

### 2.2  Submitting Code (Pull Requests)

1. **Fork** the repo and create your branch from `main`:
   ```bash
   git checkout -b feat/my-awesome-feature
   ```
2. **Write tests** for any new behaviour (`pytest` is required).
3. **Run the linter & formatter** locally:
   ```bash
   pre-commit run --all-files
   ```
4. Commit using conventional commits (see §3.2).
5. Push and open a **Draft PR** early; mark “Ready for review” when complete.
6. A core maintainer will review, request changes, or approve.

---

## 3 – Project Standards

### 3.1  Code Style

- Python 3.11+.
- **Black** for formatting, **Ruff** for linting. CI will fail if not compliant.
- Type hints are mandatory for all new public functions/classes.
- Keep functions under \~50 LOC; refactor otherwise.

### 3.2  Commit Message Convention

We use **Conventional Commits**:

```
<type>(<scope>): <subject>

<body>  # optional
```

Types: `feat`, `fix`, `docs`, `test`, `chore`, `refactor`, `perf`, `ci`.

Example:

```
feat(agents): add reward shaping for PPO optimizer
```

### 3.3  Branch Naming

`<type>/<short-description>` → e.g. `fix/resource-leak`.

### 3.4  Tests

- Place tests under `tests/` mirroring package structure.
- Aim for **≥60 %** coverage on new modules.
- Use **pytest fixtures** for shared setup; avoid network calls in unit tests.

---

## 4 – Continuous Integration

Every PR triggers:

1. **Lint / Format** (`ruff`, `black --check`)
2. **Tests** (`pytest -q`)
3. **Build** (wheel + optional Docker image)

All checks **must pass** before merge.

---

## 5 – Licensing & Developer Certificate of Origin (DCO)

AdaptiQ is licensed under **Apache License 2.0**.\
By submitting a contribution you confirm that:

1. You have the right to license your contribution under Apache 2.0.
2. You sign off your commits (`git commit -s`) to certify compliance with the [DCO](https://developercertificate.org/).

---

## 6 – Need Help?

- **Quick questions** → GitHub Discussions tab
- **Security issues** → [security@adaptiq.ai](mailto\:support@kosmostechnologies.io) (do **not** open a public issue)
- **Roadmap / design feedback** → open a “Proposal” issue with `[RFC]` prefix.

Happy coding! 🚀

