# Changelog

## [v0.12.9] - 2026-09-13

### Fixed
- Pin `crewai` (< 0.178) and `langchain` (< 1.0) to the validated ranges: fresh installs had been failing since both shipped releases that removed APIs the package depends on.
- Declare `pydantic`, `requests`, `tiktoken` and `langchain-core`, which the code imports directly but only received transitively.
- Correct the license classifier, which declared MIT while LICENSE is Apache-2.0, and drop the Python 3.8-3.10 classifiers contradicted by `requires-python`.
- Require the e-mail opt-in before uploading a run report. The failure path used to POST results to the AdaptIQ API with no check at all, while the success path already required a configured e-mail address; both now honour the same opt-in, and with `email` left empty nothing leaves the machine.
- Always write the local run report. Building and saving it sat inside the same branch as the upload, so disabling one disabled the other.
- Remove the duplicated `__version__` in `core/__init__.py`, which had drifted to 0.12.2.

### Added
- `tests.yml` workflow: Linux CI on Python 3.11 and 3.12, with a weekly scheduled run to catch dependency drift.
- `CITATION.cff` and `ROADMAP.md`.

### Changed
- README repositioned: AdaptIQ as the learning layer for loop engineering; measured results first; honest support matrix; roadmap in four verbs.
- Root-level scripts moved to `scripts/`, working notes to `docs/notes/`.
- Removed the SaaS sections and the claim that Linux was untested.
- Dropped the unused `scikit-learn` dependency.

## [v0.12.8] - 2025-09-08

### Added
- Refactor/post run optimization

### Changed
- Updated from PR #43

## [v0.12.7] - 2025-08-22

### Added
- Fix/template include on deploy

### Changed
- Updated from PR #40


## [v0.12.6] - 2025-08-22

### Added
- Fix/template include on deploy

### Changed
- Updated from PR #39


## [v0.12.5] - 2025-08-22

### Added
- Fix/template include on deploy

### Changed
- Updated from PR #38

## [v0.12.4] - 2025-08-22

### Added
- Fix/template include on deploy

### Changed
- Updated from PR #36


## [v0.12.3] - 2025-08-22

### Added
- Refactor/scale adaptiq

### Changed
- Updated from PR #32


## [v0.12.2] - 2025-07-21

### Added
- fix: ignore file system and editor files

### Changed
- Updated from PR #30


## [v0.12.1] - 2025-07-21

### Added
- fix: add ignore logs

### Changed
- Updated from PR #28

