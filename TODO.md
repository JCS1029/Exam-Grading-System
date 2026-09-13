# Project Master To-Do Checklist

> **Automated Exam Grading & Plagiarism Detection System**  
> Track project execution progress across all phases. Mark items as completed (`- [x]`) as you and your partner develop the system.

---

## Phase 1: Architecture, Data Pipeline & Security
- [ ] **Repository Setup & Safety**
  - [x] Create `.gitignore` to prevent leaking `.env`, student scans, and cache to public repo
  - [x] Create `.env.example` template with dummy credentials
  - [ ] Initialize Python virtual environment (`python -m venv venv`) and `requirements.txt`
  - [ ] Set up project folder structure (`backend/`, `frontend/`, `tests/`, `storage/`)
  - [ ] Add `.gitkeep` to storage subdirectories (`uploads/`, `processed/`, `crops/`, `reports/`)

- [ ] **Document Intake & Preprocessing Microservice**
  - [ ] Support multi-page PDF and image upload endpoints (JPEG, PNG, TIFF)
  - [ ] Implement image deskewing and perspective rectification
  - [ ] Implement CLAHE adaptive contrast enhancement for faint pencil handwriting
  - [ ] Build automated student ID header anonymization/masking (PII protection)
  - [ ] Write unit tests for image preprocessing under simulated rotation and shadows

---

## Phase 2: Layout Parsing & Bi-Directional Transcription (Hebrew + Math)
- [ ] **Multi-Column Layout Analysis**
  - [ ] Implement vertical projection / gutter detection for multi-column exams
  - [ ] Build bounding box segmentation for handwritten work blocks
  - [ ] Develop Topological Reading Order DAG:
    - [ ] Prioritize Right-to-Left (RTL) column progression for Hebrew
    - [ ] Preserve Left-to-Right (LTR) math derivation blocks within each column
  - [ ] Filter out crossed-out / scribbled-out scratch work from active reading order

- [ ] **VLM Transcription Engine**
  - [ ] Build Gemini 1.5 Pro / Flash client wrapper with JSON schema enforcement
  - [ ] Formulate prompt for interleaved Hebrew cursive text + LaTeX equations
  - [ ] Implement LaTeX syntax validator (checking balanced braces, brackets, and delimiters)
  - [ ] Build fallback cross-validation logic (GPT-4o) for borderline transcription confidence
  - [ ] Implement OCR confidence score calculation per question

---

## Phase 3: Semantic Grading Engine & Flexible Reasoning
- [ ] **Rubric Decomposition Engine**
  - [ ] Build parser for instructor master solution and point distribution key
  - [ ] Deconstruct solutions into an **Atomic Rubric Graph (ARG)**:
    - [ ] Discrete milestones with specific point weights
    - [ ] Pre-defined valid alternative trajectories (e.g. proof methods, substitutions)
    - [ ] Common misconception penalty rules
  - [ ] Implement instructor UI preview for verifying/adjusting the generated rubric

- [ ] **Multi-Agent Evaluation & Symbolic Verification**
  - [ ] Build Method Classifier to identify which valid solution method the student chose
  - [ ] Integrate SymPy CAS engine for exact algebraic and derivative equivalence checking
  - [ ] Implement Consequential Error ("נגרר") Propagation Engine:
    - [ ] Deduct points solely for an initial arithmetic slip
    - [ ] Re-evaluate subsequent steps based on student's intermediate value to award full partial credit
  - [ ] Implement Hebrew Pedagogical Feedback Generator with constructive explanations
  - [ ] Compute composite Confidence Score ($S_{conf}$) for every graded question

---

## Phase 4: Independent Plagiarism & Collusion Detection
- [ ] **Decoupled Similarity Analysis**
  - [ ] Generate dense vector embeddings for all student question answers
  - [ ] Compute pairwise cosine similarity matrix across the cohort
  - [ ] Implement MinHash / n-gram matching for near-duplicate phrasing

- [ ] **Mathematical Derivation Matching & Anomaly Detection**
  - [ ] Parse LaTeX equation sequences into derivation step DAGs
  - [ ] Calculate Graph Edit Distance and longest common derivation subsequences
  - [ ] Implement Idiosyncratic Error Fingerprinting:
    - [ ] Detect shared, non-standard incorrect calculations (the "smoking gun")
    - [ ] Identify shared unusual variable names or identical layout quirks
  - [ ] Implement Louvain community clustering to detect collusion groups
  - [ ] Generate side-by-side visual diff reports highlighting matching derivation steps

---

## Phase 5: Instructor Dashboard & HITL Review Interface
- [ ] **Human-in-the-Loop (HITL) Verification Workbench**
  - [ ] Build split-screen review UI (original scan crop vs. transcribed LaTeX + suggested score)
  - [ ] Implement 10-second rapid review queue for medium-confidence questions
  - [ ] Add one-click approval and manual point override controls
  - [ ] Record instructor adjustments to continuously improve prompt few-shot examples

- [ ] **Global Difficulty & Cohort Analytics**
  - [ ] Calculate Item Response Theory (IRT) difficulty ($b$) and discrimination ($a$) parameters
  - [ ] Unsupervised error clustering to identify "Where the class struggled"
  - [ ] Build interactive score distribution charts and topic mastery heatmaps
  - [ ] Export final grades to CSV and LMS-compatible formats (Moodle, Canvas)

---

## Phase 6: System Integration, Testing & Deployment
- [ ] **Testing & Quality Assurance**
  - [ ] Assemble benchmark dataset of sample exams (including multi-column, alternative proofs, and collusion pairs)
  - [ ] End-to-end integration test (upload $\to$ preprocess $\to$ transcribe $\to$ grade $\to$ plagiarism report)
  - [ ] Performance and load testing for bulk exam intake
- [ ] **Partner Collaboration & Deployment**
  - [ ] Verify clean execution from fresh clone using `README.md` instructions
  - [ ] Containerize application with Docker & Docker Compose
  - [ ] Write API documentation and instructor user guide
