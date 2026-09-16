# Master Checklist

> Mirrors the phases in [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md). Mark items `- [x]` as they complete.
>
> **A phase is not done when the code runs — it is done when its exit gate passes on held-out data.** Gates are listed at the end of each phase below. Do not start the next phase before its predecessor's gate passes; the gates are ordered so that a failure surfaces while it is still cheap to respond to.

---

## Phase 0 — Model bake-off and feasibility

Nothing else gets built until this passes. The brief names Hebrew handwriting as the primary difficulty, and its answer determines the architecture of every stage downstream.

- [x] **Fix the broken configuration first**
  - [x] Add `google-genai` to `requirements.txt` — the primary VLM had no SDK installed and could not be called at all
  - [x] Replace the retired model IDs in `.env.example` and `.env` (`gemini-1.5-pro` was shut down 29 Sep 2025 and returns 404)
  - [x] Add missing dependencies: `PyMuPDF`, `networkx`, `datasketch`, `jiwer`, `jinja2`, `playwright`, `sqlalchemy`, `alembic`, `tenacity`, `pytest-asyncio`
  - [x] Install Chromium for PDF rendering (`playwright install chromium`) and confirm Hebrew RTL output renders
  - [ ] Verify model availability with a live `models.list` call once `GEMINI_API_KEY` is set; pin exact version IDs; do not use `-latest` aliases
- [ ] **Assemble the pilot set** (12–15 real handwritten pages, obtained with consent)
  - [ ] At least 4 multi-column pages
  - [ ] At least 2 pages with heavy strikethrough
  - [ ] A deliberate spread of handwriting legibility
- [ ] **Hand-produce ground truth** — Hebrew prose transcript plus LaTeX per formula (tedious, unavoidable, and the measuring stick for the whole project)
- [ ] **Run the comparison**, recording cost and latency per page
  - [ ] `gemini-3.1-pro-preview`
  - [ ] `gemini-3.5-flash`
  - [ ] `gpt-5.6-sol`
  - [ ] A classical baseline (Tesseract-Hebrew or DocTR) to establish what the VLM actually buys
- [ ] **Score prose and mathematics separately** — they fail differently and a blended number hides which half is broken
  - [ ] Hebrew CER via `jiwer`, after NFC normalisation and niqqud stripping
  - [ ] Formula exact-match after SymPy canonicalisation
  - [ ] Rate of LaTeX that fails to parse at all, reported separately
- [ ] **Test cheap multi-column handling** before assuming a layout model is needed: full page to the VLM with an OpenCV vertical projection profile as a gutter prior
- [ ] **Project full-cohort cost** from measured per-page cost, not from published token prices (image inputs dominate)
- [ ] **Write the decision memo** — numbers, chosen model pinned, and an explicit statement of what fails and why

**Gate:** Hebrew CER ≤ 8% · formula exact-match ≥ 85% · LaTeX parses ≥ 95% · question regions located 100% · projected cost within budget.

**If the gate fails, stop and escalate** — require better scans, move to templated answer boxes, or narrow the question types. Building past a failed transcription gate produces a system that grades noise.

---

## Phase 1 — Intake, booklet reconciliation, preprocessing, anonymisation

- [ ] Multi-page PDF intake via PyMuPDF rasterisation at configurable DPI; direct JPEG/PNG/TIFF
- [ ] **Booklet reconciliation** — the highest-consequence correctness problem in the system
  - [ ] Detect anonymous booklet ID from the exam ID field or barcode
  - [ ] Validate page count per booklet against the expected count
  - [ ] Reconcile the booklet set against the course roster
  - [ ] Halt and escalate on any unmatched page, duplicate ID, or short booklet — silent mis-attribution gives a student someone else's grade and no downstream metric detects it
- [ ] Deskew (Hough transform plus projection-profile refinement)
- [ ] Perspective rectification by page-border homography
- [ ] CLAHE adaptive contrast for faint pencil
- [ ] **Anonymisation before any external API call**
  - [ ] Detect and mask the identity header region
  - [ ] Assign pseudonym; store the mapping in a separate local table that never leaves the machine
- [ ] Unit tests under simulated rotation, shadow, and low contrast
- [ ] Test set with deliberately shuffled and short booklets

**Gate:** 100% of pilot pages ingest without crash · residual skew ≤ 0.5° · identity masked before egress 100% (hard gate) · zero silent booklet mis-attributions.

---

## Phase 2 — Layout, multi-column reading order, transcription

- [ ] Gutter detection via vertical ink-density projection profile
- [ ] VLM question and sub-question segmentation, with the gutter prior supplied
- [ ] **Coverage validation against the question list parsed from the instructor's exam**
  - [ ] Keep *answered*, *deliberately blank*, and *not found* as three distinct states
  - [ ] Never convert *not found* into a zero — halt and escalate instead
- [ ] Reading-order resolution within a question (RTL columns for prose, LTR preserved inside math blocks)
- [ ] Follow visual continuation arrows
- [ ] Exclude strikethrough and scratch-work regions from grading
- [ ] Transcription to the canonical schema: `hebrew_text`, `math_latex[]`, `interleaved_markdown`, `flagged_tokens[]`, plus retained raw response
- [ ] LaTeX validation via SymPy/KaTeX, with re-transcription then cross-model check on parse failure
- [ ] Prompt versioning, recorded on every transcript
- [ ] Response caching by image hash, so development re-runs are free
- [ ] Held-out evaluation set, separate from the Phase 0 pilot

**Gate:** segmentation recall ≥ 99% / precision ≥ 95% · reading order correct ≥ 95% on multi-column · strikethrough excluded ≥ 90% · CER regression ≤ 2 pp vs. Phase 0 · unanswered vs. not-found distinguished 100%.

---

## Phase 3 — Grading engine

- [ ] **Rubric extraction** from the instructor's solution and scoring key into weighted milestones
  - [ ] Instructor review-and-edit UI, mandatory before any grading runs — an LLM misreading the scoring key must not silently become the grading standard
  - [ ] Rubric versioning
  - [ ] Re-grade affected questions when a rubric is edited, rather than leaving stale grades
- [ ] **Solution diversity** (the brief's core requirement)
  - [ ] Instructor-declared alternative paths as parallel milestone trajectories
  - [ ] Open validity check for unanticipated methods — judge soundness and conclusion independently of the rubric's route
  - [ ] Route every open-validity outcome to human review; a novel valid method is where the system is least reliable and a wrong automatic zero costs the most
- [ ] **Symbolic equivalence** — `simplify(student - solution) == 0` via SymPy, with numeric spot-checking where simplification cannot close
- [ ] **Consequential error (ציון נגרר)**
  - [ ] Deduct once at the originating milestone
  - [ ] Extract the student's own erroneous intermediate value
  - [ ] Re-evaluate downstream milestones by substituting that value
  - [ ] Award full credit for downstream reasoning that is correct given the student's own number
  - [ ] Implement as explicit substitution, not left to model judgement, so behaviour is inspectable and consistent across the cohort
- [ ] Hebrew feedback generation, constructive in tone
- [ ] **Reproducibility** — temperature 0, pinned model version frozen for the cohort, prompt and rubric versions and raw response stored on every grade
- [ ] **Build the two purpose-made test sets** (no generic accuracy metric substitutes for these)
  - [ ] Several genuinely different valid solutions per question
  - [ ] Several solutions with a deliberate early arithmetic slip and correct subsequent reasoning
- [ ] Instructor independently grades a set of exams, to compare against

**Gate:** MAE ≤ 1.0 pt/question · within ±1 pt ≥ 85% · total-score Pearson r ≥ 0.90 · mean *signed* error within ±0.3 pt (catches systematic bias that MAE hides) · alternative-method set at full credit ≥ 90% · consequential-error set deducting exactly once ≥ 90%.

---

## Phase 4 — Independent copy detection

Separate cohort batch job over anonymised transcripts, with no access to grades.

- [ ] **Layer 1 — semantic similarity**
  - [ ] Per-question embeddings (`gemini-embedding-2`)
  - [ ] Pairwise cosine similarity matrix
  - [ ] MinHash/LSH n-gram near-duplicate matching (`datasketch`)
- [ ] **Layer 2 — derivation structure**
  - [ ] Parse LaTeX step sequences into derivation graphs
  - [ ] Graph edit distance and longest common subsequence over canonicalised operations (`networkx`)
- [ ] **Layer 3 — idiosyncratic fingerprinting** (the layer that actually discriminates)
  - [ ] Detect shared unusual arithmetic errors
  - [ ] Detect shared non-standard variables and notation
  - [ ] Detect matching diagram-layout peculiarities
  - [ ] Weight on shared *deviation* from the standard solution, down-weighting agreement with it — two students reaching the correct answer the standard way carries almost no information
- [ ] Clustering by threshold plus connected components; gate Louvain behind a cohort-size check
- [ ] **Evidence dossier** — side-by-side crops and transcripts, shared anomalies highlighted, reasoning in plain language
- [ ] Never emit an accusation, a verdict, or a sanction; the instructor adjudicates
- [ ] **Evaluation sets**
  - [ ] Synthetic positives: real answers paraphrased, perturbed, reordered, reinserted as extra students
  - [ ] Hard negatives: independent students both writing the standard textbook solution

**Gate:** recall on synthetic positives ≥ 90% · false positive rate on hard negatives ≤ 2% · share of pairs surfaced for review ≤ 5%.

---

## Phase 5 — Confidence calibration and human review

- [ ] **Build the confidence score from trustworthy signals** — not from self-reported model confidence, which is poorly calibrated and will return 0.95 on a garbled transcription
  - [ ] LaTeX parse success and symbolic verification outcome (deterministic — highest trust)
  - [ ] Cross-model transcription disagreement (two models, or one at two temperatures)
  - [ ] Segmentation stability: overlaps, ambiguous gutters, unresolved arrows
  - [ ] Rubric-matcher vs. symbolic-check agreement
  - [ ] Self-reported confidence and flagged tokens, as a weak input only
  - [ ] Decide whether cross-model checking runs on every page or selectively (roughly doubles transcription cost)
- [ ] **Derive the band thresholds from data**
  - [ ] Plot flagged-versus-actually-wrong on the validation set
  - [ ] Read the cut points off the curve; keep the curve as a reportable result and the justification for the bands
- [ ] **Review workbench**
  - [ ] Side-by-side original crop and transcription with proposed score
  - [ ] Inline LaTeX editor
  - [ ] Point override controls
  - [ ] One-click confirm for the fast path
  - [ ] Highlight the specific uncertain tokens
- [ ] Audit log of every override: before/after, user, timestamp, model and prompt versions in force
- [ ] Feed overrides back into few-shot examples and rubric refinement
- [ ] Measure actual median review time

**Gate:** error rate among auto-finalised questions ≤ 2% (the safety-critical number) · low-confidence flagging precision ≥ 60% · median review time ≤ 15 s · thresholds traceable to the calibration curve.

---

## Phase 6 — Instructor report and global difficulty diagnostics

The brief's stated end goal. Reach a crude version of it early (see Sequencing below) so that running out of time becomes a quality problem rather than an existence problem.

- [ ] **Item statistics sized to the actual cohort**
  - [ ] Facility index per question (mean fraction of points earned)
  - [ ] Discrimination via point-biserial correlation, plus the top/bottom 27% index
  - [ ] Flag non-discriminating items (point-biserial < 0.2)
  - [ ] Gate IRT behind a cohort-size check in code — 2PL needs roughly 100+ examinees, and on a 40-student section it produces confident-looking noise
- [ ] **Misconception clustering**
  - [ ] Embed deduction reasons
  - [ ] Cluster and label
  - [ ] Emit global difficulty statements ("48% of students lost points on Q3 for omitting the integration constant")
- [ ] Global difficulty ranking across questions
- [ ] Score distribution and per-topic mastery view
- [ ] **Hebrew PDF rendering** — print the workbench's own Jinja2 HTML via headless Chromium (Playwright), which gives native RTL and already-rendered KaTeX; budget real time, as RTL prose interleaved with LTR LaTeX is a known source of rendering defects
- [ ] Per-student Hebrew feedback report
- [ ] Moodle-compatible CSV grade export

**Gate:** instructor agrees the top-3 reported difficulty points match their own judgement · ≥ 70% of misconception clusters judged coherent · Hebrew RTL PDF with embedded LaTeX renders correctly · non-discriminating items reported.

---

## Phase 7 — Privacy, hardening, supervised pilot

- [ ] **Privacy and legal groundwork, completed before any real student data is processed**
  - [ ] Institutional approval
  - [ ] Data processing agreement or zero-retention configuration with each API provider
  - [ ] Written retention and deletion policy, including the appeal-window access rule
  - [ ] Audit confirming no personally identifying data reaches any external API
- [ ] Encryption at rest for `storage/`; TLS in transit; secrets only via `.env`
- [ ] Instructor and TA roles, scoped to their own courses
- [ ] Backup and restore, **exercised at least once** — an untested restore is not a backup
- [ ] Failure handling
  - [ ] `tenacity` retry/backoff for rate limits and outages
  - [ ] Resumable jobs via `jobs.resume_token`
  - [ ] Partial-batch recovery — a crash at booklet 300 of 400 must not restart from zero
- [ ] Load check at the design target (300 students × 12 pages within 12 hours)
- [ ] **Supervised pilot at rung R0 (shadow mode)** — system grades, instructor grades independently, only the comparison is used; no student sees a system-produced grade
- [ ] Verify the Phase 3 and Phase 5 gates hold on this real data before advancing to R1
- [ ] API documentation and instructor guide
- [ ] Verify clean setup from a fresh clone following `README.md`
- [ ] Docker Compose deployment

**Gate:** zero PII reaching external APIs (audit-verified) · restore performed successfully · full-cohort run completes within 12 hours · interrupted run resumed without loss or duplication · Phase 3 and Phase 5 gates hold on the real shadow-mode exam.

---

## Sequencing note

After Phase 0 and a minimal Phase 1, **wire one exam and one question end to end** — intake, transcription, grading, report — even if every stage is crude. Then deepen each stage in phase order.

- [ ] Thin vertical slice running end to end

The brief's deliverable is the instructor report. If the work runs long while still inside transcription, there is nothing to show. A crude end-to-end path also surfaces integration mismatches early, while they are still cheap to fix.

---

## Deployment posture

Autonomy is earned with evidence, never assumed. Each rung requires the previous rung's gate to pass on real data.

- [ ] **R0 — Shadow.** System and instructor grade independently; only the comparison is used.
- [ ] **R1 — Assistive.** System proposes every grade with evidence; instructor confirms or edits each one before release.
- [ ] **R2 — Selective autonomy.** High-confidence questions auto-finalise; everything else routes to review. Requires the high-confidence error bound to hold across two exams.

R2 is never the starting configuration. **The ladder does not apply to copy detection at all** — suspected collusion is always instructor-adjudicated.
