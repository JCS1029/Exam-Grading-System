# Automated Exam Grading & Plagiarism Detection System
## Execution Plan

**Target:** a system a real instructor uses to grade a real course, built by two developers.
**Source requirements:** the course brief (מערכת בדיקת מבחנים) — transcribe handwritten Hebrew and mathematics, grade against the instructor's solution and scoring key, allow creative-but-correct solutions, detect copying in an independent stage, flag low-confidence scans, and produce an instructor report identifying global difficulty points.

---

## 1. Scope

### 1.1 What this system does

| Brief requirement | Implemented by |
| :--- | :--- |
| Handwriting → text, including formulas | Phase 2 — VLM transcription to Hebrew text + LaTeX |
| Operates from the exam and solution alone | Phase 3 — rubric derived from the instructor's own solution; no training corpus |
| Flexibility for creative correct solutions | Phase 3 — rubric milestones + symbolic equivalence, not string matching |
| Independent copy-detection stage | Phase 4 — cohort batch job, runs on anonymised transcripts after grading |
| Flag low-confidence handwriting decoding | Phase 5 — calibrated confidence, routed to human review |
| Instructor report with global difficulty points | Phase 6 — item statistics + misconception clusters |

### 1.2 Non-goals

Explicitly out of scope. Each of these was considered and cut because it costs weeks and does not move any requirement in the brief:

- Kubernetes, Helm, GPU autoscaling, Prometheus/Grafana. One instructor grading one course does not need a cluster.
- Celery / Temporal / Redis. A database-backed job table with a worker process handles this workload.
- PostgreSQL + pgvector, S3/MinIO. SQLite plus local encrypted disk is sufficient and far easier to back up. Postgres stays available as a later swap behind SQLAlchemy.
- Fine-tuning a layout model (YOLO / LayoutLMv3). This requires a labelled corpus of multi-column handwritten Hebrew exams that does not exist and would take a semester to build. Phase 2 achieves segmentation with the VLM plus classical CV priors instead.
- LTI 1.3 / bidirectional LMS integration. Moodle-compatible CSV export covers the real need.
- Role-based access control beyond instructor and teaching assistant.
- Self-hosted embedding models requiring a local GPU.

### 1.3 Scale assumption

Design target: up to 300 students × up to 12 pages, processed within 12 hours of upload. Everything above is sized to that, not to an enterprise tenant.

---

## 2. Deployment posture ladder

The system grades real students, so autonomy is earned with evidence rather than assumed. Each rung requires the previous rung's gate to pass on real exam data.

| Rung | Behaviour | Entry condition |
| :--- | :--- | :--- |
| **R0 — Shadow** | System grades; instructor grades the same exam independently; only the comparison is used. No student sees a system-produced grade. | Phase 3 gate passed on pilot data |
| **R1 — Assistive** | System proposes every grade with evidence; instructor confirms or edits every question before release. | R0 agreement metrics met on a real exam |
| **R2 — Selective autonomy** | High-confidence questions auto-finalise; everything else goes to human review. | R1 shows the high-confidence band holds its error bound over two exams |

**R2 is never the starting configuration, and the ladder never applies to copy detection at all.** Suspected collusion is always instructor-adjudicated — the system produces evidence, never an accusation and never a sanction.

---

## 3. Phases

Each phase has an **exit gate**: numeric criteria measured on held-out data. A phase is not finished when the code runs, it is finished when the gate passes. Gate numbers are targets to be revised once Phase 0 tells us what is actually achievable — a revised target with data behind it is a result; an unmeasured target is decoration.

---

## 3. Phases

Each phase has an **exit gate**: numeric criteria measured on held-out data. A phase is not finished when the code runs, it is finished when the gate passes.

---

### Phase 1 — Intake, Preprocessing, Booklet Reconciliation, & Anonymisation

**Rationale:** Raw mobile scans and multi-page exam PDFs suffer from tilt, low contrast, and varied resolutions. Furthermore, booklet pages must be attributed to the correct student and anonymised before any cloud egress.

**Work**

- Multi-page PDF and image intake (PyMuPDF rasterisation at configurable 300 DPI; JPEG/PNG/TIFF direct).
- **Image Preprocessing**:
  - Deskew via Hough transform plus projection-profile refinement (target residual skew $\le 0.5^\circ$).
  - CLAHE adaptive contrast enhancement for faint pencil handwriting.
  - Page-border homography perspective rectification.
- **Booklet reconciliation & Anonymisation**:
  - Anonymous booklet ID detection from header fields or barcodes.
  - Page count validation per booklet.
  - Identity region detection and local masking before any image leaves the machine.

**Exit gate**

| Metric | Threshold |
| :--- | :--- |
| Pilot pages ingested and preprocessed without crash | 100% |
| Residual skew after correction | $\le 0.5^\circ$ |
| Identity regions masked before cloud egress | 100% (zero exceptions) |
| Booklet mis-attribution on test set | 0 silent errors; every anomaly halted |

---

### Phase 2 — Layout Analysis, Column Splitting, & Question Segmentation

**Rationale:** Empirical evidence from full-page VLM tests showed that sending unsegmented, multi-column handwritten pages directly to a VLM results in scrambled reading order, 35s+ latency, and 503 timeouts. Segmentation into isolated question and column crops is mandatory *before* transcription.

**Work**

- **Gutter & Multi-column Detection**:
  - OpenCV vertical projection profiles and connected-component gutters to divide multi-column layouts into discrete vertical reading bands.
- **Question Region Segmentation**:
  - Detect question delimiters, problem labels (e.g. שאלה 1, סעיף א), and bounded answer areas.
  - Table vs. Free-form classification (e.g. detecting exam summary score grids like `test3sol` page 1).
- **Sub-Region Cropping**:
  - Extract isolated image crops for each question / column / table section and save them with spatial bounding coordinates.
  - Filter out full-page margins and blank areas to drastically reduce VLM token counts and latency.

**Exit gate**

| Metric | Threshold |
| :--- | :--- |
| Column gutter detection on multi-column pages | $\ge 95\%$ |
| Question region recall on pilot exam set | $\ge 99\%$ |
| Latency reduction per crop vs full page | $\ge 70\%$ reduction (< 8s per crop) |

---

### Phase 3 — VLM Feasibility Bake-off & Transcription Pipeline

**Rationale:** Run transcription on the preprocessed, segmented crops produced by Phases 1 and 2. This measures true handwriting accuracy, formula parsing, and cost on bounded, clean inputs without cross-column scrambling.

**Work**

- Transcribe bounded question crops using candidate models (`gemini-3.5-flash`, `gemini-3.6-flash`, with `gemini-3.1-pro` when quota allows).
- **Objective (Calibrated) Quality Metrics**:
  - Do NOT rely on self-reported model confidence (which hallucinates 0.95 on scrambled text).
  - Score Hebrew prose CER via `jiwer` against ground truth.
  - Validate all mathematical expressions through SymPy parsing.
  - Detect strikethrough/crossed-out regions to exclude them from grading.
- Resilient API architecture: `tenacity` exponential backoff retry loop for 503/429 transient errors.
- Side-by-side visual verification tool (`storage/reports/phase0_review.html`) for instructor inspection.

**Exit gate**

| Metric | Threshold |
| :--- | :--- |
| Hebrew prose CER on segmented crops | $\le 8\%$ |
| Formula exact-match / SymPy parse rate | $\ge 90\%$ |
| Reading order correct on segmented columns | $\ge 98\%$ |
| Unhandled 503 / dropped request rate | 0% (handled via retries) |
| Projected full cohort cost | within budget (< $25 total) |

---

### Phase 4 — Grading engine: rubric, diversity, consequential error

**Work**

- **Rubric extraction.** Parse the instructor's solution and scoring key into discrete milestones with point weights. Critically, this is a *proposal* — the instructor reviews and edits the extracted rubric before any grading runs. An LLM misreading the scoring key must not silently become the grading standard.
- **Alternative solution paths.** Two complementary mechanisms, because the brief's diversity requirement cannot be met by rubric enumeration alone:
  1. Instructor-declared alternative paths in the rubric (energy conservation vs. Newtonian, direct proof vs. contradiction).
  2. An open-ended validity check for methods nobody enumerated: the grader must judge whether the student's reasoning is mathematically sound and reaches the required conclusion, independent of the rubric's route. This path is always routed to human review — a novel valid method is exactly the case where the system is least trustworthy and the cost of a wrong automatic zero is highest.
- **Symbolic equivalence** via SymPy: `simplify(student - solution) == 0`, with numeric spot-checking as a fallback for expressions SymPy cannot close. Prevents penalising correct-but-unsimplified answers.
- **Consequential error ("ציון נגרר").** Deduct once for the originating slip, then re-evaluate every downstream milestone by substituting the student's own erroneous intermediate value. Correct reasoning from a wrong number earns full credit for that step.
- **Reproducibility, required for real use.** Temperature 0, exact pinned model version, prompt version recorded, and the raw model response stored with every grade. A grade an instructor cannot reconstruct and defend during an appeal is not usable on real students. Model version must be frozen for the duration of a cohort's grading — a mid-run change means students were graded by different systems.
- Re-grade on rubric change: when the instructor edits a rubric, affected questions are re-run and re-versioned rather than left stale.

**Exit gate**

Compared against the instructor's own independent grading of the same exams (per-question scale normalised to 10 points).

| Metric | Threshold |
| :--- | :--- |
| Mean absolute error per question | ≤ 1.0 point |
| Questions within ±1 point of instructor | ≥ 85% |
| Correlation of total exam score | Pearson r ≥ 0.90 |
| Mean **signed** error | within ±0.3 point — catches systematic over- or under-generosity that MAE hides |
| Curated valid-alternative-method set receiving full credit | ≥ 90% |
| Curated consequential-error set deducting exactly once | ≥ 90% |

The last two rows need purpose-built test sets: write out several genuinely different correct solutions per question, and several solutions containing a deliberate early arithmetic slip with correct subsequent reasoning. These directly test the two hardest requirements in the brief and no generic accuracy metric substitutes for them.

---

### Phase 5 — Independent copy detection

Runs as a separate cohort-wide batch job on anonymised transcripts, after grading, with no access to grades. Independence is a requirement of the brief and also what keeps it from becoming circular.

**Work**

- Layer 1 — semantic similarity: embed per-question answers (`gemini-embedding-2`), pairwise cosine similarity, plus MinHash/LSH n-gram near-duplicate matching via `datasketch`.
- Layer 2 — derivation structure: parse LaTeX step sequences into derivation graphs; compare with graph edit distance and longest common subsequence over canonicalised operations (`networkx`).
- Layer 3 — idiosyncratic fingerprinting, the discriminating signal: two students reaching the same correct answer is expected and carries almost no information. Two students sharing the same bizarre arithmetic error, the same unusual substitution variable, or the same non-standard notation is the actual evidence. Score on *shared anomaly*, explicitly down-weighting agreement with the standard solution path.
- Clustering: threshold plus connected components. Louvain community detection only when the cohort is large enough for it to mean anything — on 40 students it is theatre.
- Evidence dossier: side-by-side crops and transcripts with the specific shared anomalies highlighted, and the reasoning stated in plain language. Output is evidence for a human decision, not a verdict.

**Evaluation.** Real copying is unlabelled, so build a synthetic positive set: take real answers, paraphrase them, perturb a few steps, reorder some derivations, and insert them as additional students. Also build a hard negative set — pairs of independent students both giving the standard textbook solution — because that is the false positive that destroys trust and ruins a student's record.

**Exit gate**

| Metric | Threshold |
| :--- | :--- |
| Recall on synthetic positives | ≥ 90% |
| False positive rate on hard negatives (independent standard solutions) | ≤ 2% |
| Share of all pairs surfaced for review | ≤ 5% — review load must be tractable |

---

### Phase 6 — Confidence calibration and human review

**Work**

- **Do not trust self-reported model confidence.** A VLM will return 0.95 on a garbled transcription; self-reported confidence is poorly calibrated and using it as the primary signal would make the brief's low-confidence flagging requirement decorative. Build the score from signals with actual evidence behind them:
  - **Cross-model disagreement** — transcribe twice (two models, or one model at two temperatures) and measure divergence. This is the strongest available signal for roughly double the transcription cost, and can be applied selectively.
  - LaTeX parse success and symbolic verification outcome (deterministic, therefore trustworthy).
  - Segmentation stability: overlapping regions, ambiguous gutters, unresolved continuation arrows.
  - Grader-internal consistency: whether rubric matching and the symbolic check agree.
  - Self-reported confidence and flagged tokens, as a weak additional input only.
- **Derive the thresholds; do not invent them.** Plot flagged-versus-actually-wrong on the validation set and choose cut points from that curve. Record the curve — it is a reportable result and it is what justifies the bands to an examiner or a department.
- Review workbench: original scan crop beside the transcription and proposed score, with inline LaTeX editing, point override, and a one-click confirm for the fast path.
- Audit trail: every instructor override logged with before/after, timestamp, and the model version and prompt version in force. This is the appeal record.
- Feed overrides back into few-shot examples and rubric refinement.

**Exit gate**

| Metric | Threshold |
| :--- | :--- |
| Error rate among auto-finalised (high-confidence) questions | ≤ 2% — the safety-critical number |
| Precision of low-confidence flagging (flagged items that genuinely contain an error) | ≥ 60% |
| Median instructor review time, medium-confidence question | ≤ 15 seconds, measured not assumed |
| Thresholds traceable to the calibration curve | required |

---

### Phase 7 — Instructor report and global difficulty diagnostics

This is the brief's stated end goal, and the most common way projects like this fail is reaching it with no time left. The thin end-to-end slice in Phase 7's sequencing note exists to prevent exactly that.

**Work**

- **Item statistics sized to the actual cohort.** Item Response Theory two-parameter estimates need roughly 100+ examinees to be stable; on a 40-student course they produce confident-looking noise. Default to Classical Test Theory item analysis — facility index (mean fraction of points earned) and discrimination via point-biserial correlation plus the top/bottom 27% index — and enable IRT only when the cohort is large enough, with the threshold enforced in code rather than left to judgement.
- Misconception clustering: embed the deduction reasons, cluster, and label each cluster, yielding statements of the form "48% of students lost points on Q3 for omitting the integration constant."
- Global difficulty ranking across questions, plus score distribution and per-topic mastery.
- Hebrew PDF output. Render by printing the same Jinja2 HTML the review workbench serves, via headless Chromium (Playwright). A real browser gives native right-to-left layout and already-rendered KaTeX, which avoids the two things that make this painful otherwise — WeasyPrint needs a separate GTK/Pango runtime on Windows and cannot render LaTeX without pre-converting it to MathML or SVG. Still budget real time here: bidirectional prose mixed with left-to-right LaTeX is a known source of rendering defects.
- Per-student feedback report in Hebrew, and Moodle-compatible CSV grade export.

**Exit gate**

| Metric | Threshold |
| :--- | :--- |
| Instructor agreement that the top-3 reported difficulty points match their own judgement | recorded per pilot exam |
| Misconception clusters judged coherent by the instructor | ≥ 70% |
| Hebrew PDF renders correctly, including RTL prose with embedded LaTeX | visually verified |
| Non-discriminating items flagged (point-biserial < 0.2) | reported |

---

### Phase 8 — Privacy, hardening, and supervised pilot

The system now touches real student work, which imposes obligations independent of how well it grades.

**Work**

- Privacy and legal groundwork, needed *before* real student data is processed: institutional approval, a data processing agreement or zero-retention configuration with each API provider, a written retention and deletion policy, and an audit confirming no personally identifying data reaches any external API. The anonymisation gate from Phase 1 is the technical half; this is the other half.
- Encryption at rest for `storage/`, TLS in transit, secrets only via `.env`.
- Instructor and TA roles, scoped to their own courses.
- Backup and restore, exercised at least once — an untested restore is not a backup.
- Failure handling: API rate limits and outages with `tenacity` backoff, resumable jobs, partial-batch recovery. A crash 300 booklets into a 400-booklet run must not require restarting from zero.
- Load check at the design target (300 students × 12 pages).
- **Supervised pilot at rung R0.** Run one real exam in shadow mode: the system grades, the instructor grades independently, and the two are compared. Only if the Phase 3 and Phase 5 gates hold on this real data does the system advance to R1.

**Exit gate**

| Metric | Threshold |
| :--- | :--- |
| PII reaching external APIs | zero, audit-verified |
| Restore from backup | performed successfully |
| Full-cohort run at design target | completes within 12 hours |
| Interrupted run resumed without data loss or duplication | verified |
| Phase 3 and Phase 5 gates on real shadow-mode exam | hold |

---

## 4. Sequencing

Build a **thin vertical slice first**, then deepen. After Phase 0 and a minimal Phase 1, wire one exam and one question end to end — intake, transcription, grading, report — even if every stage is crude. Then improve each stage in phase order.

The reason is scheduling risk. The brief's deliverable is the instructor report; if the work runs long while still inside transcription, there is nothing to show. A crude end-to-end path converts that risk into a quality problem instead of an existence problem, and it surfaces integration mismatches early, when they are cheap.

---

## 5. Cross-cutting requirements

**Reproducibility.** Pinned model versions, temperature 0, versioned prompts, raw responses retained, model version frozen for the duration of a cohort. Without these, grades cannot be defended on appeal and results cannot be reproduced for the report.

**Cost control.** Measure cost per page in Phase 0 and project it before committing to an architecture. Route the bulk of pages through the Flash tier and escalate to Pro only on low confidence. Cache by image hash so re-runs during development are free. Track spend per exam.

**Honest failure.** Every stage must distinguish "confidently determined" from "could not determine", and the second must reach a human. The most damaging failure mode in an automated grading system is not a visible error — it is a confident silent one: a zero for a question the system failed to find, a grade from a mis-attributed booklet, a plagiarism flag on two students who independently wrote the textbook proof.

---

## 6. Technology choices

| Concern | Choice | Reasoning |
| :--- | :--- | :--- |
| API / backend | FastAPI | Already in `requirements.txt`; async suits API-bound work |
| Database | SQLite (WAL) via SQLAlchemy | Sufficient at design scale, trivial to back up; Postgres swappable later |
| Job execution | DB-backed job table + worker process | No broker to operate; resumable by construction |
| Frontend | Jinja2 + HTMX + KaTeX | The review workbench is forms and crops; a SPA is weeks of work for no requirement |
| Storage | Local encrypted `storage/` | Student scans on institution-controlled disk |
| Deployment | Docker Compose on a single machine | Matches the actual operating environment |

### 6.1 Model configuration

The original configuration pinned `gemini-1.5-pro` and `gemini-1.5-flash`, which were **shut down on 29 September 2025** and return 404. `gpt-4o` shuts down 23 October 2026 and `gemini-2.5-pro` retires 16 October 2026, so neither is a viable target either. `.env.example` now carries:

```ini
PRIMARY_VLM_MODEL="gemini-3.1-pro-preview"   # strongest spatial/multimodal reasoning
FAST_VLM_MODEL="gemini-3.5-flash"            # high-volume tier
CROSSCHECK_VLM_MODEL="gpt-5.6-sol"           # independent second opinion
EMBEDDING_MODEL="gemini-embedding-2"         # multimodal; handles text and image crops
```

Confirm availability with a live `models.list` call before Phase 0, since the Gemini 3.x line moves quickly. Pin exact version IDs and **never use the `-latest` aliases** — they re-point without notice, which would mean a cohort graded by two different models.

### 6.2 Missing dependencies

`requirements.txt` currently contains **no Google SDK at all**, so the primary transcription engine cannot be called. Also absent are libraries for components the plan depends on:

```
google-genai          # primary VLM — currently missing entirely
PyMuPDF               # PDF rasterisation for multi-page intake
networkx              # derivation graphs, GED, clustering (Phase 4)
datasketch            # MinHash/LSH near-duplicate detection (Phase 4)
jiwer                 # CER/WER measurement (Phase 0 gate)
jinja2 + playwright   # Hebrew RTL PDF reports via headless Chromium (Phase 6)
sqlalchemy + alembic  # persistence and migrations
tenacity              # API retry/backoff
pytest-asyncio        # async test support
```

---

## 7. Open decisions

1. **Scan quality standard.** Flatbed 300 DPI or phone photographs? This materially changes achievable CER and should be settled with Phase 0 evidence, then imposed as a requirement on the instructor.
2. **Cross-check coverage.** Dual-model transcription on every page, or only where the primary signals low confidence? Roughly doubles transcription cost; Phase 5 calibration data decides.
3. **Autonomy ceiling.** Does the instructor ever want auto-finalised grades (R2), or is assistive review of every question (R1) the permanent operating mode? Affects how hard Phase 5 calibration must be pushed.
4. **Feedback language.** Hebrew only, or bilingual for English-medium programmes?
5. **Retention.** How long are scans, transcripts, and evidence dossiers kept after grades are released, and who may access them during an appeal window?
