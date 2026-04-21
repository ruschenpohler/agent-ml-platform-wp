# An ML Platform for Trustworthy Human-Agent Research Production

## Abstract

Machine learning research in regulated environments requires data integrity, experiment provenance, and governance controls that are demonstrable, reproducible, and auditable. These requirements are difficult to maintain through convention alone, and they become harder when autonomous agents join the research workflow. This essay describes a platform that encodes these requirements as mechanistic guarantees: executable checks at load time, commit time, and merge time that block non-compliant work before it enters the record. The platform is organized into four complementary modules. Content-Addressed Storage provides cryptographic data integrity and immutability through SHA-256 hashing. Mechanistic Governance distributes validation across three enforcement layers (runtime, commit-time, and integration-time) with a single authoritative source for all governance rules. Provenance and Audit captures a tiered record of every experiment (data state, code state, environment, reasoning) supplemented by independent audit logs outside the experiment tracking system. Agentic Control extends these guarantees to autonomous contributors through mandatory hypothesis declaration, cost tracking, and mechanical enforcement of the governed entrypoint. The composition of these modules produces three properties that no single module provides: reconstructible research from a single run identifier, institutional memory that accumulates as a byproduct of ordinary work, and regulatory readiness (SR 11-7, BCBS 239, EBA guidelines, EU AI Act) as a structural property of the research process. Three case studies in GDP nowcasting, sectoral lending, and credit default prediction illustrate these properties in practice.


## I. Introduction

This platform is a mechanistically governed ML research environment for regulated settings. It enforces data integrity, experiment provenance, audit trails, and reproducibility not as guidelines or conventions but as structural guarantees. It uses code that runs at load time, commit time, and merge time, blocking non-compliant work before it can enter the record. The term *mechanistic governance* refers to the principle that every governance rule is encoded as an executable check, and no check can be bypassed without producing an audit record. The motivation is straightforward: Research governance in regulated environments (i.e., banking, insurance, public policy) degrades under team churn, time pressure, and loss of institutional knowledge. The degradation is gradual and invisible until a compliance failure surfaces it. Regulatory frameworks including SR 11-7 (model risk management), BCBS 239 (data aggregation and lineage), EBA guidelines on internal governance, and the EU AI Act for high-risk AI systems all require demonstrable controls over model development, data handling, and decision documentation. These controls are difficult to maintain through convention alone, and they become harder still when autonomous agents join the research workflow.

The platform addresses this through four complementary *modules*, each responsible for a distinct concern. Module 1, Content-Addressed Storage, guarantees data integrity through cryptographic hashing and immutability. Module 2, Mechanistic Governance, encodes validation and governance rules as executable checks at three layers of the development workflow. Module 3, Provenance and Audit, captures a complete, queryable record of what ran, why, and under what conditions, supplemented by independent audit logs outside the experiment record. Module 4, Agentic Control, extends the platform's guarantees to autonomous contributors by requiring structured hypothesis declaration, cost tracking, and mechanical enforcement of the single governed entrypoint. The modules are complementary in the precise sense that each addresses a concern the others do not: content-addressed storage says nothing about governance rules, and governance enforcement says nothing about data integrity. They are also mutually reinforcing: the integrity guarantees of Module 1 make the provenance records of Module 3 meaningful, and the governance enforcement of Module 2 makes the immutability promises of Module 1 trustworthy. Their composition produces properties that no single module provides alone, and Section VI discusses these compositional properties in detail.

Three case studies demonstrate the platform in practice: a GDP nowcasting pipeline at a policy organization, where temporal data, team handoffs, and six-month reconstruction requirements test the platform's provenance and institutional memory; a sectoral lending analysis at a development bank, where cross-border governance and multi-jurisdictional teams test mechanistic enforcement across organizational boundaries; and a credit default prediction model in regulated banking, where strict audit requirements, EU AI Act high-risk classification, and agent-assisted development test the full module stack. These cases are developed substantively in Section VI.


## II. Module 1: Content-Addressed Storage

Dataset mutability is a quiet failure mode in applied ML. It manifests as version-label inconsistency (the same filename pointing to different bytes on different machines), silent file replacement (an updated extract overwriting the original without notice), and directory overwrites that destroy the ability to reconstruct prior experiments. The fundamental gap is between knowing a filename and knowing the exact data: a file named `Q3_2024_final.csv` carries no guarantee about its contents. In regulated environments, where reconstruction months or years after the fact is a routine requirement, this gap is not an inconvenience but a compliance risk.

The platform closes this gap through content-addressed storage (CAS). Every dataset is stored in a directory named by the SHA-256 hash of its contents, following the pattern `data/shadow/{sha256_hash}/`. The hash is computed over the raw bytes of all CSV files in the directory, processed in sorted lexicographic filename order. For a single CSV, this reduces to the SHA-256 of that file's bytes. For multi-file datasets, the current implementation uses byte-concatenation, which is sufficient for the platform's typical use case of one CSV per dataset; length-prefixed hashing is a known upgrade path for cases where concatenation ambiguity matters.

Verification occurs at two points. At storage time, `compute_content_hash()` computes the canonical hash and the directory is named accordingly. At load time, `load_shadow_dataset()` recomputes the hash from the actual file bytes and compares it against the directory name. If the hashes do not match, the load fails immediately with an explicit error displaying both the expected and actual values. There is no silent fallback. A secondary integrity signal is provided by the schema hash, a SHA-256 of column names and their dtypes, logged alongside the content hash. This makes schema drift between runs detectable even when the underlying data bytes remain unchanged.

Immutability follows from the naming scheme: because the directory name is the content hash, any modification to the data produces a different hash and therefore requires a new directory. The old directory remains untouched, preserving the ability to reconstruct any historical experiment from its recorded dataset version. Boundary enforcement complements immutability. A `Path.relative_to()` check inside the data access layer ensures that all data loads resolve to paths within `data/shadow/`, preventing traversal to production data directories. The production-research boundary is therefore architectural rather than conventional.

SHA-256 was chosen for its collision resistance, its status as a regulatory standard in financial environments, and its compatibility with existing compliance infrastructure. The shadow naming convention makes the production-research boundary visible in the directory structure itself, so that the distinction between governed research data and production data does not depend on documentation or team discipline.

Content-addressed storage proves integrity, that these specific bytes are unchanged, but not provenance: where the data came from, who extracted it, and under what conditions. A SHA-256 hash cannot distinguish a synthetic test fixture from a real client dataset. To close this gap, each dataset directory contains a `data_manifest.yaml` alongside its CSV files. The manifest requires a `data_type` field (accepting `synthetic` or `client`), and includes `source` (free text describing origin and extraction context), `created_by` (contributor or team name), and `created_date`. The `data_type` field is enforced by a pre-commit hook (`check_data_manifest.py`) that scans every staged file under `data/shadow/` and blocks the commit if the corresponding manifest is missing or declares a `data_type` other than `synthetic`. The effect is that a contributor cannot commit a new dataset without making an active, typed declaration that it is synthetic. Omission is caught mechanically; misfiling real client data as `synthetic` is an explicit act, not an accident. At load time, `load_shadow_dataset()` reads the manifest fields and returns them alongside the dataframe and schema hash. The entrypoint logs all manifest fields as Tier 1 MLflow parameters, making data provenance, not just data integrity, part of every experiment record.


## III. Module 2: Mechanistic Governance

Governance by convention relies on code review discipline, documentation, and institutional memory. All three degrade under the ordinary pressures of applied research: time constraints, team churn, shifting priorities, and the accumulation of undocumented exceptions. The degradation is typically invisible until a regulatory examination or a failed reproduction attempt reveals that the documented process and the actual process have diverged. The platform addresses this by encoding governance as executable checks distributed across three layers of the development workflow. If a check fails, the workflow stops. If a governance rule is overridden, the override is recorded. There is no path from experiment configuration to merged code that does not pass through all three layers.

**Layer 1: Runtime Validation.** Pydantic models in `config_schema.py` validate both configuration files (`experiment.yaml` and `reasoning.yaml`) at load time, before any data is accessed or any model is trained. The allowed values for every configurable field are loaded from `project_standards.yaml`, a single YAML file that serves as the authoritative source for all governance rules. Cross-field validators catch incompatible combinations: for example, specifying `task_type=regression` with `metric=roc_auc` produces a clear error explaining the incompatibility and listing valid alternatives. Similarly, a `class_path` pointing to a classifier class when `task_type=regression` is declared is caught by interface inspection, which checks whether the class exposes `predict_proba` (indicating a classifier rather than a regressor). These checks ensure that configuration errors are caught at the earliest possible moment, before any compute-expensive operations begin.

**Layer 2: Commit-Time Enforcement.** Four pre-commit hooks run on every commit, each targeting a specific governance concern. The training guard (`check_training_guard.py`) performs an AST-based scan of all files under `src/` for training patterns such as `.fit(`, `.train(`, `.partial_fit(`, `mlflow.start_run`, and `mlflow.log_`, blocking any commit that contains training code outside the approved entrypoint `run_experiment.py`. The known limitation is that `exec()`-based code construction can bypass static analysis; the guard prevents accidental violations and raises the cost of intentional ones. The protected fields hook (`check_protected_fields.py`) blocks commits where experiment configuration files contain protected field values that differ from the reference values in `project_standards.yaml`. The schema validation hook (`check_config_schema.py`) runs full Pydantic validation at commit time, providing defense in depth against configurations that might have been edited after passing runtime validation. The data manifest hook (`check_data_manifest.py`) blocks commits that stage files under `data/shadow/` without a valid manifest declaring `data_type: synthetic`, closing a gap that none of the other hooks address: the accidental commitment of real client data to version control.

**Layer 3: Integration-Time Verification.** A pre-merge hook (`verify_pr_run_match.py`) performs cryptographic verification that the MLflow run ID cited in a pull request description matches an experiment actually executed at the branch head commit. This closes a specific gap in standard practice: that code was merged does not imply that an experiment was actually run on that code. The hook makes the gap detectable and therefore preventable at the point where code enters the protected branch.

No single layer is sufficient. Runtime validation catches configuration errors but cannot prevent a contributor from committing code that bypasses the entrypoint. Commit-time hooks catch code-level violations but cannot verify that an experiment was actually executed. Integration-time verification confirms execution but depends on the provenance records produced by the earlier layers. The three layers form a defense-in-depth architecture where each catches failure modes the others cannot.

### Protected Fields

Certain configuration fields are designated *protected* because changing them invalidates all historical metric comparisons within a project. The platform currently protects two fields: `metric_definition` (the primary evaluation metric) and `split_strategy` (the cross-validation scheme). The rationale is metric comparability: changing the primary metric makes all prior metric values non-comparable, and changing the split strategy alters fold composition so that even the same metric on the same data is not comparable across runs.

Changing a protected field requires the `--override-protected` flag on `run_experiment.py`. When used, the override is recorded in `logs/protected_overrides.log` as a JSONL entry containing the timestamp, field name, old value, new value, operator identity (resolved from git configuration or an environment variable), and the configuration file path. Protected fields are enforced at both Layer 1 (runtime) and Layer 2 (commit time). Both checks must pass for a change to proceed, and both reference the same `project_standards.yaml` values.

### The Single Source of Truth

All governance rules, including allowed metrics, split strategies, model types, hypothesis categories, expected effects, class path prefixes, and mandatory metrics, are centralized in `project_standards.yaml`. Every validator in the system reads from this file. There is no second location where allowed values are defined, no risk of divergence between what the runtime validator permits and what the commit-time hook enforces. Modifying `project_standards.yaml` requires explicit human approval. Once changed, the new rules propagate automatically to all validation layers.


## IV. Module 3: Provenance and Audit

When a regulator, an auditor, or a new team member asks how a particular model was developed, the answer must go beyond metrics and hyperparameters. The operative question is: exactly what code ran, on exactly what data, with exactly what dependencies, and why was this experiment conducted rather than another? Standard experiment tracking records some of these elements but typically leaves gaps in data provenance, platform code state, dependency versioning, and the reasoning behind experimental decisions. This module addresses two related concerns: what the platform records about every experiment (tiered provenance, inside MLflow), and what independent verification exists outside the experiment record (audit logs).

### Tiered Provenance

Every experiment run automatically logs parameters across three tiers, without manual intervention.

**Tier 1** captures immutable identifiers: the elements needed to reconstruct exactly what ran. These include the dataset content hash and schema hash (linking back to the CAS guarantees of Module 1), the git commit hash (exact code state), the platform hash (a SHA-256 over all `src/*.py` files, making any change to platform code visible without manual git inspection), the dependency lock hash (SHA-256 of `uv.lock`, capturing the full transitive dependency state), the feature pipeline hash (SHA-256 of `features.py`, tracking feature engineering changes independently), and the model configuration hash (SHA-256 of the model type, class path, and hyperparameters, tracking model architecture changes independently).

| Parameter | Source | What it captures |
|---|---|---|
| `dataset_version` | `experiment.yaml` | SHA-256 content hash of dataset directory |
| `dataset_schema_hash` | computed at load time | Column names and dtypes hash |
| `git_commit_hash` | `git rev-parse HEAD` | Exact code state |
| `platform_hash` | SHA-256 of all `src/*.py` | Platform code changes |
| `dependency_lock_hash` | SHA-256 of `uv.lock` | Full transitive dependency state |
| `feature_pipeline_hash` | SHA-256 of `features.py` | Feature engineering code changes |
| `model_config_hash` | SHA-256 of model config | Model architecture and hyperparameter changes |

**Tier 2** captures the execution environment: Python version, package versions for key dependencies (MLflow, scikit-learn, XGBoost, pandas, numpy), execution timestamp, runner type (`human` or `agent`), and Docker image digest when containerized.

**Tier 3** captures reasoning metadata. Each experiment requires a `reasoning.yaml` file specifying the hypothesis category (drawn from a controlled vocabulary of eleven categories such as `feature_addition`, `hyperparameter_tuning`, and `model_architecture`), a free-text change description, and an expected effect (`improve`, `reduce_variance`, or `explore`). All fields are validated against `project_standards.yaml`. The expected effect vocabulary is deliberately constrained: it forces the experimenter to declare, before execution, what they believe will happen, converting experimental intent from implicit assumption to queryable record.

### The Hyperparameter Audit Trail

A common failure in applied ML is the silent mismatch between configured and effective hyperparameters. A researcher specifies `max_depth=6` in the configuration, but if the model class does not accept that parameter, the value is quietly ignored and the researcher's mental model of what the model received diverges from reality. The platform addresses this with a three-way audit trail logged on every run.

The first record, `hp_declared_*`, captures what was configured in the YAML file. The second, `hp_effective_*`, captures what the model actually received, read from the fitted model's `get_params()` after instantiation. The third, `hp_dropped`, lists parameters that were declared in the configuration but not accepted by the model class. The filtering is performed by `_safe_instantiate()`, which inspects the model class's `__init__` signature to determine which parameters it accepts. For models with strict signatures (such as `LogisticRegression` or `RandomForestClassifier`), parameters not in the signature are filtered out. For models that accept `**kwargs` (such as `XGBClassifier` or `LGBMClassifier`), the platform detects the variable-keyword parameter and passes all declared hyperparameters through without filtering.

To make this concrete: if a researcher configures `max_depth=6` for a `LogisticRegression`, the platform inspects the class signature, finds no `max_depth` parameter, filters it out, instantiates the model with only the parameters it accepts, and logs `hp_dropped: max_depth`. The MLflow record shows truthfully what the model received, not what the researcher intended. Models that reject `random_state` at runtime (such as certain statsmodels wrappers) are handled gracefully: the platform catches the `TypeError`, retries without `random_state`, and logs `random_seed_applied=False` so that the non-determinism is visible in the experiment record rather than hidden.

### Independent Audit Logs

The tiered provenance described above lives inside the MLflow experiment record. Two additional logs exist outside MLflow, providing independent records that do not depend on the experiment tracking system.

The data access log (`logs/data_access.log`) is an append-only JSONL file recording every call to `load_shadow_dataset()`. Each entry contains a UTC timestamp, the dataset hash, the schema hash, the row count, and the caller identity. This log records every data load regardless of whether it occurs as part of an experiment run, providing an independent record of data access relevant to BCBS 239 requirements on data lineage. The protected overrides log (`logs/protected_overrides.log`) is an append-only JSONL file recording every use of `--override-protected`, with the same fields described in Module 2.

Both logs are currently append-only by convention, without cryptographic tamper detection. HMAC signing with keys stored outside the repository (for example, in HashiCorp Vault or AWS Secrets Manager) is a planned extension. Until that extension is in place, the logs serve as records of intent and are useful for audit, but they cannot provide cryptographic proof of non-tampering.

The platform hash, logged as a Tier 1 parameter in every run, serves as a third form of independent verification. Because it is computed as a SHA-256 over all `src/*.py` files, any change to the platform code between runs becomes visible in the experiment record. The platform hash and the git commit hash are complementary: the commit hash tells you the code changed, and the platform hash tells you whether the change affected platform infrastructure, feature engineering code, or both.

### Provenance, Experiment, and Pipeline Analytics

Because reasoning metadata is structured, validated, and stored in a queryable system, the platform supports three families of meta-analytics over experiment history.

Provenance analytics reconstruct the full development lineage for any experiment or set of experiments: what data was used, what code ran, what environment it ran in, and what governance events occurred. Experiment analytics track metric trajectories across runs: which hypothesis categories have the highest confirmation rate, how many experiments it took to reach a given performance level, and how metric deltas distribute across different modeling approaches. Pipeline analytics surface the relationship between configuration and execution: which hyperparameters were declared but dropped, which models required `random_state` fallback handling, and how feature pipeline hashes correlate with metric improvements.

Two scripts implement these meta-analytics. `query_reasoning.py` provides per-project experiment analytics (top runs by metric improvement, full run history) and cross-project experiment analytics (hypothesis category performance rates). `query_cross_project.py` provides institutional queries across all projects and clients: which modeling approaches have the highest confirmation rate, which feature engineering categories produce the largest metric gains, and how many experiments it takes to reach a target metric level by project.

Each run also automatically queries MLflow for the best prior metric value in the same project. The metric delta (current value minus prior best) is logged alongside an automatic outcome classification: `confirmed` if the delta exceeds 0.001, `negative` if it falls below -0.001, and `neutral` otherwise. The threshold is explicit and documented at 0.1 percentage points for metrics such as ROC-AUC. This provides an automated signal for whether experiments are improving the model, not merely running.


## V. Module 4: Agentic Control

Modules 1 through 3 encode practices that any careful human researcher recognizes as sound methodology: verify your data, validate your configuration, log your reasoning, maintain an audit trail. These practices assume a researcher who reads documentation, exercises judgment about edge cases, and feels professional accountability for the quality of the research record. Autonomous agents satisfy none of these assumptions. They do not read governance documentation unless the system forces them through it. They do not exercise judgment about whether a configuration change might invalidate historical comparisons. They do not feel the professional discomfort that causes a human researcher to pause before overriding a protected field. The relevant question, however, is not how to contain agents but what becomes possible when the governance that careful humans apply through judgment and discipline is instead applied mechanically and universally. The answer is that the same platform that protects against human error, which is intermittent and judgment-dependent, also protects against agent error, which is systematic and judgment-free, without requiring any change to the governance architecture itself.

### The Trust Model

The platform does not model agents as adversarial. It models them as fallible, which is the same assumption a well-designed system makes about any contributor. The difference is in how fallibility is compensated. Humans compensate through judgment, social pressure, and professional norms. Agents require mechanical compensation: validation that cannot be skipped, logging that cannot be forgotten, constraints that cannot be negotiated. The practical consequence is that the platform enables collaboration with less sophisticated and less expensive agents, because the governance boundary is structural rather than dependent on the agent's own reliability. An agent that reliably follows the governed entrypoint produces research records with the same provenance guarantees as a senior data scientist, because the guarantees are properties of the platform, not of the contributor.

### Mechanisms

The `runner_type` field in `experiment.yaml` distinguishes between `human` and `agent` contributors. When `runner_type=agent`, the platform requires `expected_effect` to be set, forcing the agent to declare before execution whether it expects the experiment to `improve` the primary metric, `reduce_variance`, or `explore` a new direction. This is logged, validated against `project_standards.yaml`, and queryable through the experiment analytics described in Module 3. The requirement converts what would otherwise be an undocumented assumption into a structured, auditable record.

Cost tracking is implemented through two CLI arguments on `run_experiment.py`: `--tokens-used` and `--estimated-cost`. These are logged to MLflow as Tier 3 parameters. A dedicated script (`cost_report.py`) aggregates costs by project and week, answering a question that matters for both operational budgeting and regulatory documentation: how much did agent-assisted development cost for this model?

Two mechanisms introduced in earlier modules take on a specific agentic interpretation in this context. The AST-based training guard (Module 2) prevents agents from creating ad hoc training scripts that bypass the governed entrypoint. Without this guard, an agent instructed to "train a model" might write a standalone script calling `.fit()` directly, bypassing configuration validation, provenance logging, and protected field enforcement. The guard does not eliminate the possibility of adversarial bypass through `exec()`-based code construction, but it prevents the accidental bypass that is the relevant failure mode for non-adversarial agents. Similarly, the structured reasoning metadata (Module 3) serves a dual purpose in the agentic context: it captures reasoning for human review, and it forces agents to articulate intent before acting. The experiment analytics then provide a quantitative measure of agent effectiveness, for example the fraction of an agent's `feature_addition` experiments that actually improved the primary metric, rather than relying on qualitative impressions.

### The Collaboration Interface

The collaboration between humans and agents is mediated by the platform's governance architecture. Humans set governance standards in `project_standards.yaml`: which metrics are permitted, which model types are allowed, which hypothesis categories are recognized, what the protected field reference values are. Agents operate within the mechanically enforced boundaries those standards create. When circumstances require a boundary to be moved, the override mechanism (`--override-protected` with audit logging) preserves human authority while creating a permanent record of the decision. The meta-analytics described in Module 3 give humans visibility into agent behavior across experiments and projects, enabling informed delegation rather than blind trust. The platform functions as a coordination layer: it makes human-agent collaboration trustworthy by making the governance boundary explicit, enforceable, and auditable.


## VI. The Composed Platform

The four modules described above are individually useful. Content-addressed storage is valuable even without governance enforcement; tiered provenance is valuable even without agentic control. But the platform's distinctive contribution lies in their composition. Each module addresses a concern the others do not, and the guarantees of each module depend on and strengthen the guarantees of the others. This section describes three compositional properties, that is, properties that arise from the composition of modules rather than from any single module, and then demonstrates them through three case studies.

### Compositional Property 1: Reconstructible Research

The first compositional property combines reproducibility and traceable lineage. A single MLflow run ID serves as an entry point to the complete reconstruction of an experiment: the exact data (CAS hash and schema hash, from Module 1), the exact code (git commit hash and platform hash, from Modules 2 and 3), the exact environment (dependency lock hash and Python version, from Module 3), the exact model configuration (model config hash, with the full declared/effective/dropped hyperparameter audit trail, from Module 3), the reasoning behind the experiment (hypothesis category, change description, and expected effect, from Module 3), governance events that occurred during development (override log and data access log, from Modules 2 and 3), and the cost of agent-assisted development if applicable (from Module 4).

No manual assembly is required. The information is logged automatically by the platform's entrypoint and can be retrieved from a single run identifier. This matters most when the person asking the question is not the person who ran the experiment: a new team member, an auditor, a regulator, or a policy reviewer conducting a retrospective months after the work was completed.

Reproducibility itself rests on four levels, each provided by a different part of the platform. Data-level reproducibility is guaranteed by content-addressed storage, which ensures bit-identical inputs through SHA-256 verification. Code-level reproducibility is guaranteed by the git commit hash, the platform hash, and the feature pipeline hash, which together capture the state of application code, platform infrastructure, and feature engineering logic. Environment-level reproducibility is guaranteed by the dependency lock hash (pinning every transitive dependency through `uv.lock`), the Python version pin, and the Docker image digest for containerized execution. Execution-level reproducibility is guaranteed by the single entrypoint rule, mechanically enforced by the training guard, which ensures that no experiment can bypass the governed code path from configuration to result.

Random seed propagation supports determinism within these levels. The platform injects `random_seed` into all sklearn-compatible models through `_safe_instantiate()` and seeds global numpy and Python random state at run start. Models that reject `random_state` are handled gracefully, with `random_seed_applied=False` logged to MLflow so that non-determinism is visible rather than hidden. Per-fold cross-validation scores are logged individually, enabling independent verification of variance calculations.

The current evaluation scope is cross-validation only. CV scores are relative ranking tools: they indicate whether configuration A outperforms configuration B within a project, but they are not unbiased estimates of production performance. The platform does not currently enforce an independent held-out test set or gate model promotion on a minimum performance threshold. The requirements for such a gate are documented and designed (a CAS-validated `test_set_version` field, a single-use test set policy enforceable through `query_cross_project.py`, and a performance threshold in `promote_model.py`), but they are not yet implemented. Final model validation must currently happen outside this system through a separate process.

[Figure 1: Single Run ID Lineage Reconstruction. Radial diagram with the run ID at center and spokes extending to each provenance element: data hash, code state, environment, model config, reasoning, governance events, cost. This figure is the single image that communicates what the composed platform provides.]

### Compositional Property 2: Institutional Memory

The second compositional property addresses a problem familiar to any team that has experienced turnover: the loss of knowledge about what was tried, what worked, what failed, and why certain decisions were made. In applied ML, this knowledge typically lives in individual researchers' memories, in scattered notebook comments, or in meeting notes that are never consulted again.

The platform converts this tacit knowledge into a structured, queryable record. Reasoning metadata (Module 3) captures the hypothesis behind every experiment in a controlled vocabulary. Protected fields (Module 2) encode the current governance baseline and record every deviation. The meta-analytics described in Module 3, spanning provenance, experiment, and pipeline analytics, make the accumulated record accessible to anyone with query access. A new team member, or an agent starting work on an existing project, can determine what hypotheses have been tested, which hypothesis categories have had the highest confirmation rate, what the current governance baseline is, what overrides have been approved and by whom, and how many experiments it took to reach the current performance level. This is what distinguishes a research platform from a collection of experiment logs: the record is not merely complete but legible, and it accumulates institutional knowledge as a byproduct of ordinary research activity rather than as a separate documentation burden.

### Compositional Property 3: Regulated-Environment Readiness

The third compositional property is that the platform's combined guarantees address core requirements across the regulatory frameworks relevant to its target domains. CAS data integrity and the data access log address BCBS 239 requirements on data accuracy, lineage, and audit trails. The three-layer governance architecture and the protected fields mechanism address SR 11-7 requirements on model development controls and change management. Tiered provenance logging and the append-only audit logs address EBA guidelines on governance and audit trail completeness. For high-risk AI systems under the EU AI Act, the platform's data governance mechanisms (CAS, shadow data boundary, data access log) are relevant to Article 10; its automatic logging of all Tier 1 through 3 parameters is relevant to Article 12; its tiered provenance and reasoning metadata are relevant to Article 13 on transparency; its protected field mechanisms, override audit trail, and promotion prerequisites are relevant to Article 14 on human oversight; and its reproducibility guarantees, random seed propagation, and per-fold CV logging are relevant to Article 15 on accuracy and robustness.

The key point is that regulatory readiness is a property of the research process as the platform defines it, not a post-hoc documentation exercise. A research team that uses the platform's governed workflow produces compliant records as a byproduct of doing research, because the compliance-relevant information (data lineage, code state, reasoning, governance events) is captured automatically by the same entrypoint and validation layers that enforce research quality. A detailed article-by-article regulatory mapping is provided in the Appendix.

### The Three Case Studies

[Case studies are at the sketch stage. This section will be extended with more thorough and procedural demonstrations that trace each scenario through the full module stack. The structure below indicates the intended scope and focus of each case.]

#### Case 1: GDP Nowcasting Pipeline

**Context:** High-stakes policy environment. Multiple economists contributing to quarterly GDP estimates. Ensemble models. Six-month reconstruction requirement for policy review. Temporal data with seasonal structure.

**Scenario:** Economist A builds a Q3 nowcast in June using `temporal_split` validation and an XGBoost ensemble. She logs her reasoning: `hypothesis_category=model_architecture`, `change_description="switch from ridge to XGBoost for non-linear quarterly patterns"`, `expected_effect=improve`. The run records CAS hash, platform hash, dependency lock hash, and per-fold temporal CV scores.

In September, Economist B takes over Q3 refinement. He queries the experiment history and sees the full record: what was tried, what improved the metric, what the current baseline is. He adds a feature (`hypothesis_category=feature_addition`) and the platform logs his run with its own complete provenance.

In December, a policy reviewer asks: "how was the Q3 estimate built?" The reviewer needs no institutional knowledge. A single run ID reconstructs the exact data (CAS hash verifiable against the archive), the exact code (git commit), the exact environment (`uv.lock` hash), and the reasoning chain across both economists' contributions.

**Platform value:** CAS ensures bit-identical data across economists. The temporal split ensures the CV strategy respects the time structure. Reasoning metadata captures the handoff context that would otherwise be lost to team churn. The meta-analytics make the development history legible to anyone, at any time.

#### Case 2: Sectoral Lending Analysis

**Context:** Cross-border data governance at a development bank. Teams in Frankfurt and Luxembourg working on the same lending model. Sensitive banking data subject to BCBS 239. Multiple jurisdictions with different data access regulations.

**Scenario:** The Frankfurt team trains a baseline model using `metric_definition=roc_auc` and `split_strategy=stratified_kfold`, both protected fields. The Luxembourg team, working with a different jurisdiction's data exhibiting severe class imbalance, proposes switching the primary metric to F1.

The platform blocks the change at both runtime and commit time. The Luxembourg team requests an override. A senior data scientist approves, and the run is executed with `--override-protected`. The override log records who approved the change, what changed, when, and why.

Six months later, the bank's internal audit reviews the model development process. The data access log shows which team accessed which datasets and when. The override log shows the metric change was deliberate, approved, and documented. The CAS hashes prove the data used in each run is unchanged.

**Platform value:** Mechanistic governance ensures consistent standards across jurisdictions without relying on cross-office code review discipline. The shadow data boundary prevents accidental production data exposure. Protected field enforcement prevents metric changes under pressure without accountability. The override mechanism preserves flexibility while creating an audit trail.

#### Case 3: Credit Default Prediction

**Context:** Regulated banking. SR 11-7 model risk management requirements. EU AI Act high-risk system classification (credit scoring of natural persons, Annex III, point 5(b)). Agent-assisted development. Model promotion decisions.

**Scenario:** A senior data scientist configures a hyperparameter search: 20 runs across learning rate, max depth, and regularization parameters. An agent executes the search overnight with `runner_type=agent`, `expected_effect=improve`, and cost tracking enabled.

In the morning, the data scientist queries the experiment analytics and sees all 20 runs ranked by metric improvement. The best run shows a positive metric delta with a `confirmed` outcome classification. She reviews the hyperparameter audit trail and confirms that all declared parameters were accepted by the model without drops.

She promotes the run through `promote_model.py`, which runs three prerequisite checks: git cleanliness, CAS integrity verification, and protected field alignment with `project_standards.yaml`. All pass. The run is tagged as promoted with a model card containing intended use, dataset hash, git commit, dependency lock hash, evaluation metrics, and promotion timestamp.

A regulator asks to reconstruct the development process. The promoted run ID provides the exact data, the exact code, the environment state, the full hyperparameter search history with reasoning metadata, the cost of agent-assisted development, and the governance chain confirming no protected field overrides and all promotion prerequisites satisfied.

**Platform value:** Tiered provenance satisfies SR 11-7 documentation requirements. Automatic logging satisfies EU AI Act Article 12. The protected field mechanism and override log are relevant to Article 14 on human oversight. The CV-scope boundary is stated honestly: the platform supports research and model comparison, not final validation. Production deployment requires the held-out test set gate that is documented but not yet implemented.

### Common Thread

All three cases require data integrity without trust in individuals, audit trails without manual documentation, and reproducibility without heroic effort. All three benefit from institutional memory that survives team churn and governance that does not degrade under pressure. The GDP nowcasting and sectoral lending cases demonstrate the platform's value for human-only teams working under regulatory and operational constraints. The credit default case shows the additional value when agents are involved: the same governance architecture that compensates for human fallibility compensates for agent fallibility, with the additional controls of mandatory hypothesis declaration and cost tracking that the agentic context requires.


## VII. Scope and Limitations

The platform evaluates models through cross-validation only. It does not enforce an independent held-out test set, nor does it gate model promotion on a minimum performance threshold. The requirements for a final validation gate are documented and designed but not implemented. Final model validation must currently happen outside this system.

The feature engineering module (`features.py`) contains a working pipeline with feature selection, polynomial features, and binning, but it is not a feature engineering library. The platform governs how features are tracked and hashed; the engineering of features themselves is the researcher's responsibility.

Both audit logs (`data_access.log` and `protected_overrides.log`) are append-only by convention, without cryptographic tamper detection. HMAC signing with keys stored outside the repository is a planned extension. Until that is in place, the logs are useful for audit as records of intent but cannot provide cryptographic proof of non-tampering.

The MLflow backend uses SQLite, which supports only single-writer sequential access. Concurrent use by multiple agents or researchers requires migration to PostgreSQL, a configuration change that is documented and requires no code modifications.

The statsmodels wrappers cover classifiers and regressors from the GLM, GAM, and discrete choice families. Survival models, mixed effects models, and state-space models are not currently supported.

The pre-modeling data checks consist of a five-item checklist covering missing data, data leakage, temporal validation, class imbalance, and protected field alignment. This is a starting point for institutionalizing data scrutiny before modeling begins, and it is due for refinement.


## VIII. Closing

The platform encodes a set of research practices that careful practitioners already value: verifying data integrity, validating configurations against governance rules, logging the reasoning behind experimental decisions, and maintaining audit trails that survive team churn and institutional memory loss. What the platform contributes is not the practices themselves but their mechanical enforcement. Every guarantee described in this essay is implemented as executable code that runs automatically, produces its records without manual intervention, and blocks non-compliant work before it can enter the research record. The modules are complementary in their coverage and mutually reinforcing in their guarantees, and their composition produces the three properties detailed in Section VI: reconstructible research from a single identifier, institutional memory that accumulates as a byproduct of ordinary work, and regulatory readiness as a structural property of the research process.

The architecture is designed for extension. New model types, hypothesis categories, data manifest schemas, and governance rules are added through `project_standards.yaml`, the single file from which all validators read. New modules can be composed with existing ones without modifying the governance architecture. The platform is scaffolding for trustworthy research, built to accommodate the work it has not yet been asked to support.


---

## Appendix: Regulatory Framework Mapping

| Regulatory Framework | Requirement | Platform Mechanism |
|---|---|---|
| SR 11-7 | Model development documentation | Tiered provenance logging (Tiers 1-3) |
| SR 11-7 | Change management and version control | Protected fields, override log, git integration |
| SR 11-7 | Independent validation | CV-scope boundary explicitly documented; promotion prerequisites |
| BCBS 239 | Data accuracy and integrity | SHA-256 CAS with load-time verification |
| BCBS 239 | Data lineage and audit trails | Data access log, dataset hash in every run |
| BCBS 239 | Data aggregation standards | Schema hash detects structural drift |
| EBA GL | Internal governance controls | Three-layer mechanistic enforcement |
| EBA GL | Audit trail completeness | Append-only logs and MLflow provenance |
| EU AI Act Art. 10 | Data governance (high-risk) | CAS, shadow data boundary, data access log |
| EU AI Act Art. 12 | Automatic logging (high-risk) | All Tier 1-3 parameters logged without manual intervention |
| EU AI Act Art. 13 | Transparency (high-risk) | Tiered provenance, reasoning metadata, hyperparameter audit trail |
| EU AI Act Art. 14 | Human oversight (high-risk) | Protected fields, override mechanism, promotion prerequisites |
| EU AI Act Art. 15 | Accuracy and robustness (high-risk) | Reproducibility guarantees, random seed propagation, CV fold logging |

Specific article and section numbers require legal review. The mapping structure reflects the platform's actual mechanisms; regulatory granularity requires validation by qualified counsel.


---

## Suggested Figures

1. **Figure 1: Single Run ID Lineage Reconstruction** (priority). Radial diagram: run ID at center, spokes extending to data hash, code state, environment, model config, reasoning, governance events, cost.

2. **Figure 2: Three-Layer Governance Enforcement.** Flowchart: configuration load through runtime validation, commit through pre-commit hooks, merge through pre-merge verification.

3. **Figure 3: Tiered Provenance Structure.** Layered diagram showing Tier 1, 2, and 3 with representative parameters at each level.

4. **Figure 4: Hyperparameter Audit Trail.** Concrete example showing declared, effective, and dropped parameters for a misconfigured LogisticRegression.

Figure 1 is the most important. It is the single image that communicates the compositional value of the plat