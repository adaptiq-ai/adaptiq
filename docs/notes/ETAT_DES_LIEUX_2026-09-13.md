# État des lieux — dépôt `adaptiq` · étape 0 « Crédibilité »

**Date :** 13 septembre 2026 · **Branche :** `main` @ `8b08713` · **Machine :** Windows 11, Python 3.12.11 (uv)
**Périmètre :** inspection seule. Aucun fichier du dépôt n'a été modifié, déplacé ou supprimé.
**Seuls fichiers créés :** ce rapport, et `handoff/` (copie des livrables fournis).

> **Écart d'environnement à connaître avant de lire :** le handoff a été vérifié sur **Linux**. Cette session tourne sur **Windows**. Les constats propres à la plateforme sont signalés comme tels ; ils ne remettent pas en cause les mesures Linux du handoff, ils les complètent.

---

## Résumé en 5 lignes

1. Les deux pannes annoncées sont **reproduites**, mais la première ne se manifeste pas où le handoff l'indique : `import adaptiq` **réussit** (un `except ImportError: pass` avale l'erreur). Ce sont `adaptiq --help` et `pytest` qui cassent.
2. **Le bloc de pins proposé par la checklist ne répare pas la panne** — et les versions exactes du handoff **ne s'installent pas sur Windows**. Une configuration entièrement verte a été trouvée et mesurée : **`crewai<0.178`** (install OK, CLI OK, 77 tests OK). Sans cette correction, l'étape 0 publierait une 0.12.9 encore cassée, avec une CI verte.
3. Les 8 fichiers de la racine ne sont **pas** des tests pytest. Le `git mv … tests/` prescrit fait passer la suite de `77 passed` en 8,6 s à **`77 passed, 1 error` en 63,2 s** — mesuré en bac à sable. Destination correcte : `scripts/`.
4. Deux portes de CI sont **déjà rouges** sur `main` (`black` : 22 fichiers, `isort` : 12) : toute PR vers `pre-release` échouera avant d'atteindre les tests.
5. Le module `cloud` envoie des résultats vers `api.getadaptiq.io` **sans authentification**. ⚠️ *Corrigé après revue* : contrairement à ce que ce rapport affirmait d'abord, l'envoi n'était **pas** actif par défaut sur le chemin nominal — il exigeait déjà une adresse e-mail, qui fait office de consentement. Seul le **chemin d'échec** envoyait sans aucune garde. Voir la section « Revue adversariale et corrections » en fin de document.

---

## 1. Inventaire

**Paquet.** `adaptiq` 0.12.8, `requires-python = ">=3.11"`, layout `src/`, entry point unique `adaptiq = "adaptiq.cli:main"`. Build setuptools.

**Modules de `src/adaptiq` :**

| Module | Rôle | Importé par | Couverture tests |
|---|---|---|---|
| `core/q_table`, `core/entities` | Q-table offline, modèles Pydantic | tout le paquet | indirecte |
| `core/runtime_rl` (4 fichiers) | Moteur de décision runtime | `examples/`, `agents/btp_agent` | **la totalité des 77 tests** |
| `core/pipelines` (pre_run, run, post_run) | Pipelines d'optimisation | `cli.py` | **aucune** |
| `core/reporting` | Agrégation, métriques, rapports | `core/pipelines` | **aucune** |
| `core/abstract` | Classes de base | `core`, `agents` | **aucune** |
| `agents/crew_ai` (5 fichiers) | Intégration CrewAI | `cli.py` | **aucune** |
| `agents/open_ai_sdk` | — | personne | **code mort** : un seul fichier `test.text` d'1 octet |
| `cloud` (2 fichiers) | Client HTTP `api.getadaptiq.io` | `core/reporting/.../data_processor.py` | **aucune** |
| `templates/crew_template` | Gabarit de projet | `cli.py` (package-data) | **aucune** |
| `templates/open_ai_template` | — | personne | **code mort** : un seul fichier `test.text` d'1 octet |

**Point d'attention sur la couverture :** les 77 tests portent **exclusivement** sur `core/runtime_rl`. Les pipelines, le reporting, l'intégration CrewAI et la CLI ne sont couverts par aucun test. La baseline de 77 est donc un socle étroit — elle ne protège pas les chemins que la CLI emprunte.

**Dépendances déclarées vs verrouillées :**

| Paquet | Déclaré dans `pyproject.toml` | Verrouillé dans `uv.lock` | Résolu par un install neuf ce jour |
|---|---|---|---|
| `crewai` | *(aucune borne)* | 0.134.0 | **1.15.21** |
| `crewai_tools` | *(aucune borne)* | 0.48.0 | **1.15.21** |
| `langchain` | *(aucune borne)* | 0.3.26 | **1.4.0** |
| `langchain-core` | **non déclaré** | 0.3.66 | 1.6.3 |
| `langchain-openai` | *(aucune borne)* | 0.2.14 | **1.6.2** |
| `openai-agents` | `>=0.1.0` | 0.1.0 | 0.20.0 |
| `numpy`, `pyyaml`, `python-dotenv` | *(aucune borne)* | — | dernières |
| `scikit-learn` | *(aucune borne)* | 1.7.0 | **jamais importé** (cf. A8) |

---

## 2. Constats

Gravité : **H** = bloque l'étape 0 ou trompe le lecteur · **M** = à corriger dans l'étape 0 · **B** = à signaler.

### A. Installation et dépendances

| # | Constat | Preuve exacte | Grav. | Action proposée | Périmètre |
|---|---|---|---|---|---|
| A1 | Dépendances non épinglées | `pyproject.toml` l. 33-45 : `"crewai",` `"langchain",` … sans borne | **H** | Épingler — mais voir A6, la forme proposée ne suffit pas | Étape 0 |
| A2 | Un install neuf résout des majeures incompatibles | venv neuf py3.12 : crewai **1.15.21**, langchain **1.4.0**, langchain-core 1.6.3, langchain-openai 1.6.2. La dérive s'est **aggravée** depuis le handoff (qui observait 1.6.1) | **H** | Idem A1 | Étape 0 |
| A3 | La CLI casse à l'import | `adaptiq --help` → `ImportError: cannot import name 'OutputParserException' from 'crewai.agents.parser'`, via `src/adaptiq/agents/crew_ai/crew_logger.py:4` | **H** | Corrigé par un pin **correct** | Étape 0 |
| A4 | Les tests ne se collectent plus | `pytest tests -q` → `ModuleNotFoundError: No module named 'langchain.prompts'` (`core/abstract/integrations/base_prompt_parser.py:6`) → `3 errors during collection`, **0 test exécuté** | **H** | Corrigé par un pin correct | Étape 0 |
| A5 | **`import adaptiq` n'échoue PAS**, contrairement au handoff | `src/adaptiq/__init__.py` l. 6-10 : `try: from .agents import *; from .core import * / except ImportError: pass`. Vérifié sur l'install cassée : `import adaptiq` → **rc=0**, `version = 0.12.8` | **H** | Conséquence directe sur le smoke test de la CI, cf. C3 | Étape 0 |
| A6 | **Le bloc de pins de la checklist ne répare pas la panne** — constat le plus important du rapport | Voir la mesure détaillée en §3 | **H** | Resserrer la borne haute et épingler `chromadb`/`litellm`, cf. §3 | Étape 0 |
| A7 | Trois dépendances directes non déclarées : `pydantic`, `requests`, `tiktoken` | Utilisées dans `core/entities/*.py`, `cloud/http_client.py`, `core/abstract/integrations/base_log_parser.py`, `core/reporting/.../metrics_calculator.py` ; absentes de `[project.dependencies]`. N'arrivent aujourd'hui que transitivement via `crewai` | **M** | Les déclarer | Étape 0 |
| A8 | `scikit-learn` déclaré mais **jamais utilisé** | `grep -rn "sklearn\|cosine_similarity" src/` → **aucun résultat**. Le commentaire du `pyproject.toml` dit pourtant « For cosine_similarity » | **M** | Retirer | Étape 0 |
| A9 | Classifiers incohérents avec `requires-python` | `requires-python = ">=3.11"` mais les classifiers annoncent Python **3.8, 3.9, 3.10** | **M** | Aligner sur 3.11/3.12 | Étape 0 |
| A10 | **Classifier de licence faux** | `pyproject.toml` : `"License :: OSI Approved :: MIT License"`. Or `LICENSE` est une **Apache License 2.0**, et le README cible affiche un badge Apache-2.0 | **H** | Corriger en Apache-2.0 | Étape 0 |
| A11 | `langchain-core` est utilisé mais non déclaré | `import langchain_core` présent dans `src/` ; absent des dépendances ; le bloc de la checklist l'ajoute — bon réflexe à conserver | **M** | Déclarer | Étape 0 |
| A12 | `importlib.resources.path` déprécié, exécuté à chaque import | `src/adaptiq/__init__.py` l. 14-17 | **B** | Signaler ; `importlib.resources.files()` en étape 1 | Hors périmètre |

### B. Racine du dépôt

**Conclusion : aucun des 8 fichiers de la racine n'est un test pytest.** Ce sont des scripts dont tout le code vit dans le corps du module, avec des `print` et des `try/except` — pas de fonctions `test_*` collectables au sens de pytest.

| Fichier | Nature réelle | Effet d'un `mv` vers `tests/` | Destination |
|---|---|---|---|
| `test_btp.py` (107 l.) | Script ; corps de module. Définit une fonction locale `test_reward(result)` l. 31 | **Casse** : pytest collecte `test_reward`, prend `result` pour une fixture → `ERROR tests/test_btp.py::test_reward` | `scripts/` |
| `test_import.py` (35 l.) | Script ; corps de module | Corps **exécuté à la collecte** | `scripts/` |
| `test_runtime_fixed.py` (77 l.) | Script ; `sys.path.insert` | Corps exécuté à la collecte | `scripts/` |
| `test_runtime_import.py` (35 l.) | Script ; corps de module | Corps exécuté à la collecte | `scripts/` |
| `test_yaml_runtime_rl.py` (73 l.) | Script ; lit `agents/btp_agent/runtime_rl_config.yaml` en **chemin relatif au CWD** | Corps exécuté à la collecte + dépendance au répertoire courant | `scripts/` |
| `quick_test.py` (27 l.) | Script ; même dépendance au CWD | Non collecté (nom) | `scripts/` |
| `validate_runtime_manager.py` (446 l.) | Validation manuelle ; fonctions `test_*(classes)` **qui prennent des arguments** | Non collecté tant qu'il n'est pas renommé `test_*` | `scripts/` |
| `run_tests.py` (14 l.) | Lanceur `pytest.main([...])` sur un seul fichier | Non collecté (nom) | `scripts/`, ou supprimer — `pytest tests` le remplace |

**Mesure en bac à sable** (copie hors dépôt, `tests/` + les 8 scripts, venv à dépendances fonctionnelles) :

```
AVANT (tests/ seul)            : 77 passed, 3 warnings in 8.58s
APRÈS le git mv prescrit       : 77 passed, 3 warnings, 1 error in 63.22s
                                 ERROR tests/test_btp.py::test_reward
```

> **L'étape 2 de la checklist est donc à corriger.** Le `git mv` vers `tests/` ajoute une erreur à la baseline et multiplie la durée de la suite par 7 (les corps de module s'exécutent réellement). Les 8 fichiers doivent aller dans `scripts/`.

| # | Constat | Preuve | Grav. | Action | Périmètre |
|---|---|---|---|---|---|
| B1 | Aucune configuration pytest | Pas de `[tool.pytest.ini_options]`, ni `pytest.ini`, ni `conftest.py`, ni `tests/__init__.py`. Un `pytest` nu collecte donc aussi la racine | **M** | Ajouter `testpaths = ["tests"]` | Étape 0 |
| B2 | 5 notes de travail à la racine | `RUNTIME_RL_FILES.md`, `RUNTIME_RL_IMPLEMENTATION_SUMMARY.md`, `RUNTIME_RL_REUSABILITY_ANALYSIS.md` (66 Ko), `RUNTIME_RL_SIMPLIFIED_GUIDE.md`, `PRICING_METHODS_EXPLAINED.md` | **M** | → `docs/notes/` | Étape 0 |
| B3 | Artefact de run versionné | `examples/storage/qtables/btp_pricing_runtime_q_table.json` (3,5 Ko) : Q-table produite par une exécution, avec clés métier (`wood_south_45`, `brick_north_40`) | **B** | Signaler | Signalement |
| B4 | Un `.env` est suivi par git | `examples/prompt_engineer_agent/.env`. **Contenu : placeholders (`your_...`), aucune clé réelle** — vérifié. `.gitignore` n'ignore pas `.env` | **M** | → `.env.example` + ajouter `.env` au `.gitignore` | Étape 0 |
| B5 | Assets orphelins | `docs/assets/leaderboard.gif` n'est plus référencé depuis `8b08713`. Le README cible ne référence **aucune** image → `architecture.png` et `ui_screenshot.png` le deviendront aussi | **B** | Signaler | Signalement |
| B6 | Deux répertoires morts | `src/adaptiq/agents/open_ai_sdk/` et `src/adaptiq/templates/open_ai_template/` ne contiennent qu'un `test.text` d'1 octet, et sont livrés dans le paquet | **B** | Signaler | Signalement |

### C. Intégration continue

| # | Constat | Preuve | Grav. | Action | Périmètre |
|---|---|---|---|---|---|
| C1 | Aucun workflow ne teste `main` | `pull-request.yml` : `on: pull_request: branches: [pre-release]`. `bump-version.yml` et `release-migration.yml` : sur `closed`. **Aucun `schedule` nulle part** | **H** | Ajouter `tests.yml` | Étape 0 |
| C2 | Cause directe de la dérive non détectée | Aucun job n'installe les dépendances à neuf hors PR vers `pre-release`. Pire : `release-migration.yml` **publie sur PyPI sans lancer un seul test** (`python -m build` puis publish) | **H** | `tests.yml` + cron hebdomadaire | Étape 0 |
| C3 | **Le smoke test de `tests.yml` ne détecterait pas la panne** | Le workflow fourni fait `python -c "import adaptiq"`. Or cet import **réussit** sur une install cassée (A5) : la CI serait verte sur un paquet inutilisable | **H** | Remplacer par un import traversant, p. ex. `python -c "from adaptiq.agents.crew_ai import CrewConfig"`. L'étape `adaptiq --help` du même workflow le couvre déjà — mais elle est **après**, et le message d'erreur est moins clair | Étape 0 |
| C4 | `tests.yml` n'entre pas en conflit avec l'existant | Job `tests`, déclencheurs `push`/`pull_request` sur `main` et `pre-release`, `schedule` hebdomadaire, `workflow_dispatch`. Aucun recouvrement avec les 3 workflows présents | — | Ajouter, sous réserve de C3 | Étape 0 |
| C5 | **`black --check` est rouge sur `main`** | `black --check --include '\.py$' ./src/adaptiq` → **22 fichiers** seraient reformatés (`aggregator.py`, `state_mapper.py`, `runtime_rewards.py`, `runtime_decision_engine.py`, …) | **H** | Décision requise — **Question 1** | Étape 0 |
| C6 | **`isort --check-only` est rouge sur `main`** | **12 fichiers** en erreur, dont `quick_test.py` et `run_tests.py` — la racine encombrée y contribue | **H** | Idem C5 ; le rangement en retire une partie | Étape 0 |
| C7 | Les linters de la CI ne sont pas épinglés non plus | `pull-request.yml` : `pip install flake8 black isort pytest pytest-cov`, sans version. Même classe de bug que A1 : une nouvelle version de `black` peut rougir la CI sans qu'une ligne de code ait changé | **M** | Signaler | Signalement |
| C8 | Deux tags manquants | `git tag` : `v0.12.4` puis `v0.12.7`. `v0.12.5` et `v0.12.6` sont absents alors que le CHANGELOG les documente — publications probablement échouées | **B** | Signaler | Signalement |

### D. Cohérence des versions et métadonnées

| # | Constat | Preuve | Grav. | Action | Périmètre |
|---|---|---|---|---|---|
| D1 | Version dupliquée et **désynchronisée** | `pyproject.toml` : `0.12.8` · `src/adaptiq/__init__.py` : `0.12.8` · **`src/adaptiq/core/__init__.py` : `0.12.2`**. `bump-version.yml` ne met à jour que les deux premiers | **M** | Supprimer le `__version__` de `core/__init__.py` | Étape 0 |
| D2 | En-tête copiée-collée | `src/adaptiq/core/__init__.py` l. 1 : `# src/adaptiq/__init__.py` | **B** | Corriger avec D1 | Étape 0 |
| D3 | `CITATION.cff` : date incohérente | `version: "0.12.9"` avec `date-released: "2025-08-13"`. La 0.12.9 sort le 2026-09-13 | **M** | Corriger en `2026-09-13` | Étape 0 |
| D4 | Citation : année incohérente | README cible, BibTeX : `year = {2025}`, clé `amri_adaptiq_2025` ; `CITATION.cff` et la release sont en 2026 | **B** | **Question 2** | Arbitrage |
| D5 | Format du CHANGELOG divergent | Existant : `## [v0.12.8] - 2025-09-08` (préfixe `v`, tiret court). Entrée fournie : `## [0.12.9] — 2026-09-13` (sans `v`, cadratin) | **B** | **Question 2** | Arbitrage |

### E. Zones sensibles — description seulement, aucune décision

| # | Sujet | Ce que contient réellement le code |
|---|---|---|
| E1 | `agents/btp_agent/` | **Un seul fichier**, `runtime_rl_config.yaml`, **aucun code**. Il expose : trois noms de méthodes de chiffrage (`method_db_standard`, `method_knn_historical`, `method_regional_adjust`), une mention « Catalog pricing (Batiprix-style) », un gabarit de contexte `{material}_{region}_{surface}`, et une fonction de récompense en lambda. **Aucune logique produit exécutable, aucune donnée client.** La sensibilité se limite à la nomenclature des méthodes. |
| E2 | `src/adaptiq/cloud/` | **Toujours câblé, pas du code mort.** `core/reporting/aggregation/helpers/data_processor.py:9` importe `AdaptiqCloud` et l'instancie (l. 21). Chaîne : `aggregator.py:639` et `:714` → `send_run_results()` → `POST https://api.getadaptiq.io/projects`. Le retirer casserait `data_processor`. |
| E3 | **Envoi réseau** — ⚠️ *ce constat était partiellement faux, voir la correction en fin de rapport* | `aggregator.py:456` : `should_send_report: bool = True`. J'en avais conclu que les résultats partaient vers l'API par défaut. **C'est inexact sur le chemin nominal** : l'envoi y était déjà conditionné par `if self.email != ""`, et la clé `email` du template vaut `""` par défaut, avec un commentaire qui en fait explicitement un consentement. L'envoi réellement inconditionnel n'existait que sur le **chemin d'échec** (`aggregator.py:714`, sans aucune garde). **Aucune authentification** dans les deux cas : `http_client.py` ne pose que `Content-Type` et `Accept`. |
| E4 | Appel réseau à l'import | Importer `adaptiq` déclenche, via `litellm` (transitif de `crewai`), un `GET https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json` — observé en clair pendant les essais. L'import n'est donc pas hors-ligne. |

### F. Dette visible

| # | Constat | Preuve |
|---|---|---|
| F1 | 3 TODO | `base_prompt_parser.py:48`, `report_builder.py:16`, `report_builder.py:438` (« Fix the error tracking (Future Fixes) ») |
| F2 | Dépréciations Pydantic, visibles à chaque exécution | `adaptiq_meterics.py:47` et `:193` : `class Config` + `schema_extra`. Warnings émis à l'import **et dans la sortie pytest** : `UserWarning: 'schema_extra' has been renamed to 'json_schema_extra'` et `PydanticDeprecatedSince20: Support for class-based config is deprecated` |
| F3 | API Pydantic v1 résiduelle | `adaptiq_metrics.py:43-44` : `.dict()` au lieu de `.model_dump()` |
| F4 | Faute de frappe dans un nom de module | `core/entities/adaptiq_meterics.py` — « meterics » pour « metrics ». Renommer casserait les imports : **signaler seulement** |
| F5 | Import non résoluble dans un template | `templates/crew_template/main.py:4` : `from crew import GenericCrew`. Normal pour un gabarit, mais le fichier est livré dans le paquet installé |
| F6 | CONTRIBUTING contredit la CI | `CONTRIBUTING.md:28` : « create your branch from `main` », alors que seules les PR vers `pre-release` sont validées. L. 86 mentionne `ruff`, absent du projet (la CI utilise `flake8`) |
| F7 | `ARCHITECTURE.md` est en français | Le README cible est en anglais et y renvoie comme documentation principale (« Details: ARCHITECTURE.md »). Un reviewer anglophone atterrit sur 59 Ko de français. Ses commandes CLI sont en revanche **correctes** (`--template crew-ai`) |

---

## 3. Reproduction des pannes — mesures

Toutes les mesures ci-dessous ont été faites dans des venv neufs **hors du dépôt**, Python 3.12.11, Windows 11.

### Scénario A — dépendances telles que déclarées (non épinglées)

```
uv venv --python 3.12 && uv pip install -e . pytest
→ crewai 1.15.21 · crewai-tools 1.15.21 · langchain 1.4.0 · langchain-core 1.6.3
  langchain-openai 1.6.2 · openai-agents 0.20.0 · pydantic 2.12.5

python -c "import adaptiq"   → rc=0   ← RÉUSSIT (masqué par except ImportError: pass)
adaptiq --help               → ImportError: cannot import name 'OutputParserException'
                                from 'crewai.agents.parser'
pytest tests -q              → ModuleNotFoundError: No module named 'langchain.prompts'
                                3 errors during collection — 0 test exécuté
```

**Conforme au handoff**, à une correction près : l'import réussit au lieu d'échouer.

### Scénario B — le bloc de pins **proposé par la checklist**

```
uv pip install "crewai>=0.134,<1.0" "crewai_tools>=0.48,<1.0" "langchain>=0.3,<1.0" \
               "langchain-core>=0.3,<1.0" "langchain-openai>=0.2,<1.0" "openai-agents>=0.1.0"
→ crewai 0.203.2 · crewai-tools 0.76.0 · langchain 0.3.30 · langchain-core 0.3.86
  langchain-openai 0.3.35 · pydantic 2.13.5

pytest tests -q   → 77 passed, 3 warnings in 8.58s          ✅
adaptiq --help    → ImportError: cannot import name 'OutputParserException'  ❌
import crewai     → ModuleNotFoundError: No module named
                    'litellm.llms.bedrock.messages.invoke_transformations.
                     anthropic_claude3_transformation'                        ❌
```

> **C'est le constat le plus important de ce rapport.** Le bloc de pins de la checklist :
> - **ne répare pas la CLI** — la borne `<1.0` est trop lâche. `OutputParserException` est présent dans crewai **0.177.0** (vérifié) et **absent en 0.203.2** (vérifié) : la rupture se produit **à l'intérieur** de la plage autorisée, bien avant la 1.0 ;
> - laisse `litellm` sans contrainte, d'où un couple crewai/litellm incohérent qui rend `import crewai` lui-même impossible ;
> - donnerait néanmoins une **CI verte** : 77 tests au vert + un smoke test `import adaptiq` qui réussit toujours (A5, C3).
>
> Appliquée telle quelle, l'étape 0 publierait une 0.12.9 **toujours cassée pour un nouvel utilisateur**, avec tous les voyants au vert. C'est exactement le scénario que l'étape 0 cherche à empêcher.
>
> **Borne haute à retenir :** `crewai>=0.134,<0.178` (dernière version vérifiée fonctionnelle : 0.177.0), ou un pin exact. Et il faut contraindre `litellm` et `chromadb`, qui sont les deux vrais vecteurs de casse.

### Scénario C — versions exactes du handoff, sur Windows : **l'installation échoue**

```
uv pip install crewai==0.134.0 crewai-tools==0.48.0 langchain==0.3.26 \
               langchain-core==0.3.66 langchain-openai==0.2.14 openai-agents==0.1.0 pytest

Resolved 213 packages in 3.38s
   Building chroma-hnswlib==0.7.6
  × Failed to build `chroma-hnswlib==0.7.6`
  ├─▶ The build backend returned an error
  ╰─▶ Call to `setuptools.build_meta.build_wheel` failed (exit code: 1)
      [stderr]
      error: Unable to find a compatible Visual Studio installation.
  help: `chroma-hnswlib` (v0.7.6) was included because `crewai` (v0.134.0)
        depends on `chromadb` (v0.5.23) which depends on `chroma-hnswlib`

exit=1
```

> **Les versions exactes validées sur Linux ne s'installent pas sur Windows.** `crewai 0.134.0` impose `chromadb 0.5.23`, qui dépend de `chroma-hnswlib 0.7.6` — une extension C++ sans wheel Windows pour Python 3.12, donc à compiler, ce qui exige les Build Tools Visual Studio. Sur Linux, des wheels existent : d'où l'écart avec le handoff.
>
> Note : `crewai==0.134.0` **seul** s'installe sans problème (il résout alors `chromadb 1.5.9`, livré en wheel). C'est la **combinaison** des pins exacts qui contraint la résolution vers `chromadb 0.5.23`.

**Conséquence directe sur le choix des bornes.** Windows est pris en tenaille entre les deux scénarios :

| Jeu de dépendances | Install Windows | `pytest tests` | `adaptiq --help` |
|---|---|---|---|
| Non épinglé (aujourd'hui) | ✅ | ❌ 3 erreurs de collecte | ❌ `OutputParserException` |
| Plages `<1.0` de la checklist → crewai 0.203.2 | ✅ | ✅ 77 passed | ❌ `OutputParserException` |
| Pins exacts du handoff → crewai 0.134.0 | ❌ **compilation requise** | — | — |

Aucune des trois options proposées ne donne un dépôt fonctionnel sur Windows.

### Scénario D — combinaison candidate `crewai 0.177.0` : **entièrement verte**

Point de départ : `OutputParserException` est **présent** dans crewai 0.177.0 et **absent** en 0.203.2 (les deux vérifiés par introspection). La borne utile se situe donc entre les deux, pas à 1.0.

```
uv venv --python 3.12 && uv pip install "crewai==0.177.0" "crewai-tools<1.0" \
    "langchain<1.0" "langchain-core<1.0" "langchain-openai<1.0" \
    numpy pyyaml python-dotenv "openai-agents>=0.1.0" pytest
→ install exit=0, sans compilateur
→ crewai 0.177.0 · crewai-tools 0.76.0 · chromadb 1.5.9 · chroma-hnswlib ABSENT
  litellm 1.74.9 · langchain 0.3.30 · langchain-openai 0.3.35

adaptiq --help    → usage: adaptiq [-h] {init,validate} ...          ✅
pytest tests -q   → 77 passed, 3 warnings in 6.23s                   ✅
```

> **C'est la seule configuration mesurée qui soit intégralement fonctionnelle sur Windows** : installation sans compilateur, CLI opérationnelle, 77 tests au vert. `chromadb` monte ici en 1.5.9, livré en wheel, ce qui évite complètement le problème `chroma-hnswlib` du scénario C.
>
> **Recommandation pour l'étape 2 : borne haute `crewai>=0.134,<0.178`** au lieu de `<1.0`, à valider par vous (Question 2). Reste à confirmer sur Linux et en Python 3.11 via la CI — c'est précisément ce que `tests.yml` apportera.

**Synthèse des quatre scénarios :**

| Jeu de dépendances | Install Windows | `pytest tests` | `adaptiq --help` |
|---|---|---|---|
| A — non épinglé (état actuel) | ✅ | ❌ 3 erreurs de collecte | ❌ `OutputParserException` |
| B — plages `<1.0` de la checklist → crewai 0.203.2 | ✅ | ✅ 77 passed | ❌ `OutputParserException` |
| C — pins exacts du handoff → crewai 0.134.0 | ❌ compilation requise | — | — |
| **D — `<0.178` → crewai 0.177.0** | ✅ | ✅ **77 passed** | ✅ |

### Note de méthode — lenteur de résolution

La résolution du jeu **non épinglé** a pris **~13 min** ; celle du bloc de **plages `<1.0`**, plus de 10 min avec 235 Mo de RAM (backtracking massif sur des centaines de versions de `crewai`). Un jeu de pins **exacts** se résout en quelques secondes. Argument supplémentaire en faveur de pins serrés : la CI en sera d'autant plus rapide et déterministe.

---

## 4. Écarts README fourni ↔ code

Vérification ligne à ligne de `handoff/README.md`. **Le wording n'est pas remis en cause** ; seules les inexactitudes factuelles sont listées.

### Vérifié exact

| Affirmation | Vérification |
|---|---|
| `adaptiq init --name my_project --template crew-ai --path ./my_project` | ✅ `cli.py:82-110` — arguments et défaut `crew-ai` conformes |
| `adaptiq validate --config_path … --template crew-ai` | ✅ `cli.py:115-133` |
| « CrewAI is the only supported template today » | ✅ Un seul template réel ; `open_ai_template/` est vide (B6) |
| « OpenAI `gpt-4.1` and `gpt-4.1-mini` » | ✅ `core/entities/adaptiq_config.py:13-14`, `ModelNameEnum` |
| « Python 3.11+ » | ✅ `requires-python = ">=3.11"` — mais classifiers à corriger (A9) |
| Badge `tests.yml` | ✅ pointe sur `actions/workflows/tests.yml`, nom du fichier fourni |
| Licence Apache-2.0 | ✅ conforme à `LICENSE` — mais classifier faux (A10) |
| Liens `ARCHITECTURE.md`, `CONTRIBUTING.md`, `LICENSE` | ✅ les trois fichiers existent |
| Lien `ROADMAP.md` | ✅ sera créé par l'étape 0 |

### Écarts

| # | Affirmation | Constat | Correction proposée |
|---|---|---|---|
| R1 | « tested on Linux (CI) and **Windows** » | **Non étayé en l'état.** Avec les versions exactes validées, l'installation **échoue sur Windows** (compilation de `chroma-hnswlib`, §3-C). Avec les plages proposées, elle réussit mais la CLI reste cassée. Aucune des deux ne donne un Windows fonctionnel | Ne conserver « and Windows » que si la combinaison retenue est vérifiée sur Windows (§3-D). Sinon écrire « tested on Linux (CI) », ce qui reste vrai et défendable |
| R2 | « Dependencies are pinned to the ranges the code was validated against » | **Faux avec le bloc proposé** : la plage `<1.0` inclut des versions non validées et cassantes (§3-B) | Resserrer les bornes ; la phrase redevient vraie |
| R3 | Le README ne mentionne plus le module `cloud` | Or `cloud` est actif et **envoie les résultats de run par défaut, sans auth** (E2, E3) | Ne pas trancher ici — **Question 3** |
| R4 | BibTeX `year = {2025}` / clé `amri_adaptiq_2025` | Incohérent avec une publication 2026 (D4) | **Question 2** |
| R5 | Les 5 métriques de « Measured results » | **Non vérifiables depuis ce dépôt** : elles proviennent de `adaptiq-benchmark`, absent ici. Le DOI est bien formé, non vérifié hors-ligne | À confirmer par vous |
| R6 | « macOS should work but is not yet covered by CI » | Exact et prudent ; cohérent avec la question 4 du §8 du handoff | Aucune |

### Termes à éliminer

Le README **actuel** contient les termes proscrits sur **10 lignes** (l. 5, 8, 17, 33, 48, 61, 68, 72, 122, 326). Le README **fourni** n'en contient aucun → le critère `grep -ciE "windows|finops|proprietary|30%|benchmyagent" README.md → 0` sera satisfait par simple remplacement. `ARCHITECTURE.md` et `CONTRIBUTING.md` sont déjà exempts.

---

## 5. Risques de l'exécution de l'étape 0

| # | Risque | Probabilité | Mitigation |
|---|---|---|---|
| X1 | **Le bloc de pins publie une 0.12.9 encore cassée**, avec une CI verte | **Certaine** si appliqué tel quel | Resserrer les bornes (§3-B) **et** corriger le smoke test (C3). Critère : `adaptiq --help` doit répondre dans un venv neuf |
| X2 | **Le `git mv` vers `tests/` casse la baseline** : `1 error` ajoutée, suite 7× plus lente | **Certaine** (mesuré) | Déplacer vers `scripts/` |
| X3 | **La PR échoue sur `black`/`isort`** avant d'atteindre les tests | **Certaine** si PR vers `pre-release` | **Question 1** |
| X4 | Reformater avec `black` toucherait **22 fichiers du moteur**, contre la règle 2 du handoff | Élevée | Commit de formatage isolé et étiqueté, ou report en étape 1 |
| X5 | Retirer `scikit-learn` / ajouter `pydantic`, `requests`, `tiktoken` modifie l'arbre de dépendances | Faible | Vérification finale en venv neuf (étape 9) |
| X6 | Les scripts déplacés gardent une dépendance au CWD (`agents/btp_agent/…` en relatif) | Moyenne | Les laisser tels quels et documenter qu'ils s'exécutent depuis la racine |
| X7 | La baseline de 77 tests ne couvre que `core/runtime_rl` : elle ne protège **ni la CLI, ni les pipelines, ni l'intégration CrewAI** | Certaine | Ne pas confondre « 77 verts » et « le paquet marche ». Le critère utile est `adaptiq --help` en venv neuf |
| X8 | Sans `tests.yml`, la dérive recommencera | Certaine à long terme | Le cron hebdomadaire y répond — à condition que C3 soit corrigé |

---

## 6. Plan d'exécution proposé

Conserve l'ordre du handoff (§6), avec les corrections issues de l'état des lieux. Durée estimée : **1 h 30 – 2 h**, hors résolutions de dépendances.

| # | Action | Durée | Critère de vérification |
|---|---|---|---|
| 1 | Créer la branche `chore/etape0-credibility` | 1 min | `git branch --show-current` |
| 2 | `pyproject.toml` : pins **resserrés** (`crewai<0.178`, §3-D), + `pydantic`/`requests`/`tiktoken`/`langchain-core` (A7, A11), − `scikit-learn` (A8), classifiers (A9), licence (A10), `version = "0.12.9"` | 25 min | venv neuf : install OK **et `adaptiq --help` répond** — déjà vérifié en §3-D |
| 3 | Baseline `pytest tests -q` | 5 min | **77 passed** — sortie collée dans ce rapport |
| 4 | `git mv` des 8 scripts → **`scripts/`** ; 5 notes → `docs/notes/` ; ajouter `[tool.pytest.ini_options] testpaths = ["tests"]` | 15 min | `pytest tests -q` = **77 passed, 0 error** ; `pytest` nu ne collecte que `tests/` ; racine conforme |
| 5 | `.github/workflows/tests.yml`, **smoke test corrigé** (C3) | 10 min | YAML valide ; la commande d'import **échoue** sur une install cassée |
| 6 | Remplacer `README.md`, ajouter `ROADMAP.md` + `CITATION.cff` (date corrigée, D3) ; ajuster R1 et R2 | 15 min | `grep -ciE "windows|finops|proprietary|30%|benchmyagent" README.md` → 0 ; tous les liens relatifs existent |
| 7 | `CHANGELOG.md` : entrée 0.12.9, en mentionnant A7/A8/A10 | 5 min | Entrée présente, datée 2026-09-13 |
| 8 | Nettoyages : `core/__init__.py` (D1, D2), `.env` → `.env.example` + `.gitignore` (B4) | 10 min | `pytest tests -q` toujours vert |
| 9 | Vérification finale, venv neuf : install → import strict → CLI → tests | 20 min | Tout vert, sortie collée |
| 10 | Commits atomiques + section « Exécution » de ce rapport | 15 min | `git log --oneline` lisible ; **aucun push** |

Commits prévus :
`chore: pin dependencies to validated ranges and fix package metadata` ·
`chore: move root scripts to scripts/ and working notes to docs/notes/` ·
`ci: add tests workflow with weekly dependency-drift check` ·
`docs: reposition README, add ROADMAP and CITATION` ·
`docs: add changelog entry for 0.12.9`

---

## 7. Questions pour Wassim

1. **`black` et `isort` sont rouges sur `main`** (22 et 12 fichiers) : toute PR vers `pre-release` échouera avant les tests. Trois options — (a) un commit de reformatage isolé, mais il touche 22 fichiers du moteur et heurte la règle 2 ; (b) ouvrir la PR vers `main` en contournant `pre-release` ; (c) laisser rouge et l'assumer. **Cette décision conditionne les étapes 9-10.**

2. **Bornes des dépendances — la question la plus structurante.** Trois faits mesurés : le bloc fourni ne répare pas la CLI (§3-B) ; les pins exacts validés sur Linux **ne s'installent pas sur Windows** (§3-C) ; la borne haute réelle se situe entre crewai 0.177.0 (fonctionnel) et 0.203.2 (cassé). Je propose **`crewai>=0.134,<0.178`** — qui préserve la compatibilité Windows en laissant `chromadb` monter vers une version livrée en wheel. Trois arbitrages pour vous : (a) valide-t-on cette borne plutôt que `<1.0` ? (b) épingle-t-on aussi `litellm`, dont l'absence de contrainte casse `import crewai` en §3-B ? (c) assume-t-on que le jeu exact « validé octobre 2025 » n'est plus installable sur Windows, et donc que la référence de validation devient crewai 0.177.0 ?

3. **`src/adaptiq/cloud` envoie les résultats de run vers `https://api.getadaptiq.io` par défaut** (`should_send_report=True`), sans authentification, et le README cible ne le mentionne plus. Pour l'étape 0 : je le signale seulement, je bascule le défaut à `False`, ou je le documente dans le README ?

4. **Destination des scripts racine** : `scripts/` plutôt que `tests/`, pour la raison mesurée en §2-B. La racine cible du handoff ne liste pas `scripts/` — vous validez son ajout ? Alternative : supprimer ces 8 fichiers, dont l'utilité actuelle est faible.

5. **Périmètre des corrections `pyproject`** : au-delà des pins, je propose de corriger le classifier de licence (MIT → Apache-2.0, actuellement **faux**), les classifiers Python 3.8-3.10, les 3 dépendances manquantes et `scikit-learn` inutilisé. Tout en étape 0, ou seulement les pins ?

---

## 8. Annexe — environnement et commandes

**Environnement de mesure.** Windows 11 · Python 3.12.11 (téléchargé par `uv` 0.8.2) · venv neufs hors du dépôt · `uv pip` (résolveur équivalent à pip pour ces cas).
Le Python système de la machine est en **3.13.3** alors que `.python-version` demande **3.12** : aucun interpréteur 3.11 ou 3.12 n'était installé avant cette session.

**Baseline de référence.** 77 fonctions de test, réparties ainsi :

| Fichier | Fonctions |
|---|---|
| `tests/test_dummy.py` | 1 |
| `tests/test_runtime_decision_engine.py` | 25 |
| `tests/test_runtime_q_table_manager.py` | 20 |
| `tests/test_runtime_rewards.py` | 31 |
| **Total** | **77** |

**Venv utilisés pour les mesures** (tous hors du dépôt, dans le répertoire temporaire de la session) :

| Venv | Jeu de dépendances | Verdict |
|---|---|---|
| A | tel que déclaré (non épinglé) | install ✅ · CLI ❌ · tests ❌ |
| B | plages `<1.0` de la checklist | install ✅ · CLI ❌ · tests ✅ |
| C/D | pins exacts du handoff | install ❌ (`chroma-hnswlib`) |
| E | `crewai==0.177.0` + bornes `<1.0` | install ✅ · CLI ✅ · tests ✅ **77 passed** |

**Vérifications de style** (versions courantes des outils, comme la CI qui ne les épingle pas) :

```
black --check --include '\.py$' ./src/adaptiq   → 22 files would be reformatted, 50 unchanged
isort --check-only .                            → 12 fichiers en erreur
```

**Ce qui n'a pas pu être vérifié depuis cette session :** les 5 métriques du benchmark (dépôt `adaptiq-benchmark` absent), le DOI Zenodo, et le comportement sur Linux et macOS. La CI `tests.yml` couvrira Linux en 3.11 et 3.12 dès son ajout.

---

# Exécution — 13 septembre 2026

GO reçu de Wassim, avec arbitrage des cinq questions :

| Question | Décision |
|---|---|
| 1 — `black`/`isort` rouges | **Option (a)** : commit de formatage isolé, placé en dernier |
| 2 — Bornes des dépendances | **Borner crewai** (`<0.178`, la combinaison mesurée verte) |
| 3 — Envoi cloud par défaut | **Basculer à `False`** |
| 4 — Destination des scripts | **`scripts/`** validé |
| 5 — Périmètre `pyproject` | **Périmètre complet**, licence en **Apache-2.0** |

Branche : `chore/etape0-credibility`, **non poussée**.

## Commits

| # | Commit | Portée |
|---|---|---|
| 1 | `chore: pin dependencies to validated ranges and fix package metadata` | 17 fichiers, +31/−22 |
| 2 | `chore: move root scripts to scripts/ and working notes to docs/notes/` | 14 déplacements + `.gitignore` |
| 3 | `ci: add tests workflow with weekly dependency-drift check` | `tests.yml` |
| 4 | `docs: reposition README, add ROADMAP and CITATION` | 3 fichiers, +121/−281 |
| 5 | `fix: do not upload run results to the AdaptIQ API by default` | 2 fichiers, +13/−6 |
| 6 | `docs: add changelog entry for 0.12.9` | `CHANGELOG.md` |
| 7 | `style: apply black and isort` | 38 fichiers, +277/−195 — **cosmétique uniquement** |

**Répartition du diff :** 25 fichiers et +224/−306 pour les changements de fond, 38 fichiers et +277/−195 pour le seul reformatage. Garder le formatage en dernier permet de lire les six premiers commits sans bruit.

## Vérification finale — venv neuf, arbre final

```
uv venv --python 3.12 && uv pip install -e . pytest

1. pip install -e .                                        exit=0
2. from adaptiq.agents.crew_ai import CrewConfig            OK - adaptiq 0.12.9
3. adaptiq --help                                           usage: adaptiq [-h] {init,validate} ...
4. pytest tests -q                                          77 passed, 3 warnings in 9.29s
```

Versions résolues par les nouveaux pins : crewai 0.177.0 · crewai-tools 0.76.0 · chromadb 1.5.9 (**sans `chroma-hnswlib`**, donc sans compilateur) · langchain 0.3.30 · langchain-openai 0.3.35 · pydantic 2.13.5 · requests 2.34.2 · tiktoken 0.14.0 · `scikit-learn` **absent**, comme voulu.

Portes de style, après le commit de formatage :

```
black --check --include '\.py$' ./src/adaptiq   exit=0   (était 1, 22 fichiers)
isort --check-only .                            exit=0   (était 1, 12 fichiers)
```

`pytest` nu ne collecte plus que `tests/` grâce à `testpaths` : **77 passed**, aucun script de `scripts/` ramassé.

## Écarts assumés par rapport au handoff

Quatre déviations, chacune motivée par une mesure de l'état des lieux :

1. **Borne `crewai<0.178` au lieu de `<1.0`.** La plage fournie résout crewai 0.203.2, où `OutputParserException` a déjà disparu : les tests passent mais la CLI reste cassée. Décision de Wassim (question 2).
2. **Scripts vers `scripts/` et non `tests/`.** Mesuré : le `git mv` prescrit fait passer la suite de `77 passed` à `77 passed, 1 error`, et de 8,6 s à 63,2 s. Décision de Wassim (question 4).
3. **Smoke test de `tests.yml` modifié.** `python -c "import adaptiq"` réussit sur une install cassée (`except ImportError: pass`). Remplacé par un import traversant la garde. Vérifié : exit 0 avec l'ancienne commande sur une install cassée, exit 1 avec la nouvelle.
4. **`should_send_report` basculé à `False`.** Touche le moteur, ce que la règle 2 du handoff réserve aux cas exigés par un test. Décision explicite de Wassim (question 3).

Corrections factuelles appliquées aux fichiers fournis : `CITATION.cff` datait la 0.12.9 du 2025-08-13 (corrigé en 2026-09-13) ; la section « Limits » du README attribuait les bornes aux versions majeures alors que la rupture crewai est sur une mineure ; l'entrée CHANGELOG a été alignée sur le format `[vX.Y.Z] - YYYY-MM-DD` déjà en place dans le fichier.

## Critères du handoff

| Critère | État |
|---|---|
| Racine rangée | ✅ `ARCHITECTURE.md CHANGELOG.md CITATION.cff CONTRIBUTING.md LICENSE README.md ROADMAP.md agents docs examples pyproject.toml scripts src tests uv.lock .github .gitignore .python-version` |
| `tests.yml` présent, YAML valide | ✅ `yaml.safe_load` OK, matrice `['3.11', '3.12']` |
| Badge README ↔ nom du workflow | ✅ pointe sur `actions/workflows/tests.yml` |
| Liens relatifs du README | ✅ `ARCHITECTURE.md`, `CONTRIBUTING.md`, `LICENSE`, `ROADMAP.md` existent tous |
| Termes proscrits | ✅ `finops`, `proprietary`, `30%`, `benchmyagent`, « AdaptiQ Score », « cost saving » : **0 occurrence** |
| CHANGELOG daté 2026-09-13 | ✅ |
| Aucun push, tag ou release | ✅ |

**Une réserve sur un critère.** Le handoff demandait `grep -ciE "windows\|…" README.md → 0`. Une occurrence subsiste : la ligne de matrice de support « tested on Linux (CI) and Windows », présente dans le README fourni lui-même. Le critère visait la **fausse** note (« Linux non testé, Windows recommandé »), qui a bien disparu ; la mention restante est la mention honnête, et elle est étayée par les mesures de cette session.

## Incident de mesure — à ne pas confondre avec une régression

La première vérification finale a échoué sur :

```
ModuleNotFoundError: No module named
'langsmith._openapi_client.types.annotation_queue_retrieve_annotation_queues_params'
```

Diagnostic : **limite `MAX_PATH` de Windows (260 caractères)**, pas un défaut du dépôt. Le venv était créé dans un répertoire temporaire profond ; le chemin complet de ce fichier faisait **261 caractères** contre **257** pour un venv au nom plus court, qui lui fonctionnait avec exactement les mêmes versions de paquets. Le fichier existe sur le disque mais l'API Win32 refuse de l'ouvrir.

À retenir pour les utilisateurs Windows : `langsmith` (transitif via `langchain-core`) contient des chemins très longs. Installer dans un répertoire profond peut échouer. Deux parades : installer près de la racine, ou activer `LongPathsEnabled`. **Candidat pour la section « Limits » du README en étape 1** — non traité ici, hors périmètre.

---

# Revue adversariale et corrections — 13 septembre 2026

Avant de pousser, la branche a été soumise à une revue indépendante : **7 relecteurs** sur autant de dimensions (dépendances, changement de comportement, CI, déplacements, exactitude documentaire, effets du reformatage, respect du périmètre), puis **3 réfutateurs par constat**, chargés de démolir chaque affirmation plutôt que de l'approuver. Un constat n'était retenu que s'il survivait à au moins deux voix sur trois.

**136 agents, 0 erreur. 43 constats bruts, 26 survivants.** Les corrections ci-dessous en découlent.

## Ce que la revue a rattrapé — erreurs de cette session

| # | Défaut introduit | Correction |
|---|---|---|
| 1 | **`should_send_report = False` supprimait les rapports locaux.** Le drapeau n'englobait pas que l'envoi réseau : `build_project_result()`, `merge_json_reports()` et `save_json_report()` — seul écrivain de `reports_data/` du paquet — étaient dans la même branche. Un run par défaut ne produisait plus **aucun** rapport, pendant que le README annonçait « local and e-mail reports » | Sauvegarde locale sortie de la condition, donc toujours écrite. Envoi conditionné au consentement e-mail sur **les deux** chemins |
| 2 | **Mon analyse E3 était fausse.** L'envoi n'était pas actif par défaut sur le chemin nominal : il exigeait déjà une adresse e-mail. Le vrai défaut était le chemin d'échec, qui envoyait sans aucune garde. Basculer le défaut à `False` rendait en outre la clé `email` du template **inerte** | Défauts remis à `True`, garde e-mail ajoutée au chemin d'échec. Le consentement documenté redevient le mécanisme réel |
| 3 | **Six scripts déplacés avaient un `sys.path` cassé** : ils insèraient `dirname(__file__)/src`, soit `scripts/src`, inexistant | Remontée d'un niveau ; vérifié depuis la racine et depuis `scripts/` |
| 4 | **Commits non atomiques.** `git mv` indexe immédiatement : les 14 renommages étaient partis dans le commit « pin dependencies », et le commit « move root scripts » ne contenait que `.gitignore` | Historique reconstruit ; chaque message décrit son contenu réel |
| 5 | **8 liens relatifs cassés** dans `RUNTIME_RL_REUSABILITY_ANALYSIS.md`, relatifs à la racine et déplacés de deux niveaux | Réécrits en `../../src/…`, les 7 cibles résolvent |
| 6 | **`uv.lock` jamais régénéré** : il décrivait le projet en `2.0.0`, verrouillait `scikit-learn` et un `chromadb` qui ne compile pas sous Windows. `uv sync --frozen` échouait | Régénéré puis remonté aux versions validées. `uv sync --frozen` réussit, CLI comprise |
| 7 | **`CITATION.cff` invalide** : `orcid: ""` viole le schéma CFF 1.2.0, GitHub aurait rejeté le fichier | Champ retiré, fichier validé |
| 8 | **`ARCHITECTURE.md`** publiait encore `scikit-learn` comme dépendance critique, juste retirée par cette branche | Liste et tableau mis à jour ; §5.3 signalée comme décrivant une conception absente du code |
| 9 | **Quick start du README inexploitable** : il indiquait `./my_project/adaptiq_config.yml`, alors que `init` écrit dans `<path>/src/<name>/config/`. `adaptiq validate` échouait sur le chemin donné | Chemin réel, vérifié en exécutant `adaptiq init` |
| 10 | **Trois affirmations dépassaient le code** : « a human validates or corrects states, actions and rewards » (il n'existe qu'un champ de feedback optionnel), « e-mail reports » (le paquet n'envoie aucun e-mail), « tested on Linux (CI) » (aucune CI n'a encore tourné) | Reformulées au niveau de ce que le code fait |

## Ce qui reste ouvert

- **Paternité dans `CITATION.cff`** : un seul auteur y figure, `pyproject.toml` en déclare deux. Décision de Wassim, non tranchée ici.
- **Année de citation** : le BibTeX du README dit `2025`, `CITATION.cff` et la release sont en 2026.
- **DOI** : `CITATION.cff` présente le DOI du dépôt de benchmark comme identifiant de ce logiciel.
- **`ARCHITECTURE.md` §5.3** : décrit une architecture à embeddings que `state_mapper.py` n'implémente pas. Signalée, non réécrite.
- **`ARCHITECTURE.md` est en français** alors que le README anglais y renvoie comme référence principale.
- **`flake8` reste rouge** sur `examples/runtime_rl_example.py` (F821) — **faux positif** sur une variable portée par la boucle, préexistant sur `main`, code correct à l'exécution. Deux lignes d'initialisation le feraient taire.
- **Double exécution de la suite** sur une PR vers `pre-release` : `tests.yml` et `pull-request.yml` la lancent tous les deux. Assumé, `pull-request.yml` étant hors périmètre.

## Vérification après corrections

```
pytest tests -q                                 77 passed
adaptiq --help                                  usage: adaptiq [-h] {init,validate} ...
uv sync --frozen                                exit=0, CLI fonctionnelle
python -m build                                 adaptiq-0.12.9.tar.gz + wheel
black --check / isort --check-only              exit=0 / exit=0
scripts/*.py                                    imports OK depuis la racine et depuis scripts/
```
