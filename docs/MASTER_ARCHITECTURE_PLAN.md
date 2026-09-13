# Automated Exam Grading & Plagiarism Detection System
# Technical Blueprint & Master Architecture Specification

## 1. System Overview
The **Automated Exam Grading & Plagiarism Detection System** addresses the critical bottleneck in academic grading by digitizing, deciphering, grading, and auditing handwritten student exam scans. The system accommodates:
- **Bi-directional Multimodal Input**: Handwritten Hebrew explanations (Right-to-Left / RTL) tightly interleaved with mathematical equations, graphs, and matrices (Left-to-Right / LTR).
- **Multi-Column & Non-Linear Layouts**: Student work arranged across split columns, margins, and marked continuation pointers.
- **Semantic & Flexible Grading**: Full credit allocation for valid alternative/creative proofs and derivations, complete with consequential error ("נגרר") propagation.
- **Decoupled Anti-Collusion & Plagiarism Detection**: Multi-layer clustering detecting anomalous matching derivations, identical idiosyncratic errors, and statistical outliers across the cohort.
- **Human-in-the-Loop (HITL) Fallback**: Multi-tier confidence scoring isolating ambiguous handwriting for rapid instructor verification.
- **Cohort Analytics**: Deep Item Response Theory (IRT) analytics, error clustering, and global difficulty diagnostics for the teaching staff.

---

## 2. Architectural Strategy & Model Selection

### 2.1 VLM & OCR Pipeline: Hebrew + Complex Mathematical Formulas
The recognition of handwritten Hebrew mixed with mathematical expressions presents distinct challenges:
1. **Bi-directional Context Switching**: Standard OCR engines flip token order or misidentify the boundary between RTL Hebrew and LTR mathematical operators.
2. **Visual Ambiguity**: Cursive Hebrew letters (e.g., ז vs. ו, ם vs. ס, ח vs. ת) closely resemble mathematical symbols (e.g., $z, 1, \sigma, \cap$).
3. **Complex Two-Dimensional Notation**: Fractions, matrices, exponents, and summation limits require spatial semantic understanding rather than single-line text reading.

#### Recommended Model Strategy: Tiered Vision-Language Pipeline
- **Primary VLM: Google Gemini 1.5 Pro / Flash**:
  - **Rationale**: Demonstrates superior multimodal understanding of mixed Semitic scripts and handwritten mathematical notation. Its native 2D visual grounding and 2M token context window enable end-to-end question reasoning and direct translation into clean, standard LaTeX and markdown with inline confidence tags.
- **Secondary / Fallback VLM: OpenAI GPT-4o**:
  - Used in dual-verification mode when mathematical AST validation fails or transcription confidence is marginal ($0.70 \le C < 0.85$).
- **Local Preprocessing & Document Layout Analysis (DLA)**:
  - **OpenCV & DocTR**: Perspective rectification, dewarping of warped mobile photo scans, and CLAHE adaptive thresholding.
  - **YOLOv11-Doc / LayoutLMv3**: Fine-tuned on exam layouts to detect question headers ("שאלה 1", "סעיף ב'"), multi-column split gutters, margin notes, and crossed-out/strikethrough scribbles.
  - **Topological Reading Order DAG**: Reconstructs proper column reading sequence (RTL columns: Right $\to$ Left, with LTR mathematical sub-blocks).

---

### 2.2 Semantic & Flexible Grading Engine (Diversity & Equivalence)
A common flaw in automated grading is penalizing correct solutions that deviate from the instructor's specific derivation. This engine enforces pedagogical flexibility:
1. **Atomic Rubric Graph (ARG)**:
   - The instructor's master solution and scoring key are deconstructed into atomic conceptual milestones (e.g., Milestone 1: Boundary condition setup [2 pts]; Milestone 2: Differential substitution [3 pts]; Milestone 3: Algebraic resolution [2 pts]).
   - Alternative valid paths (e.g., proof by contradiction vs. direct derivation; energy conservation vs. kinematics) are encoded as parallel viable milestone trajectories.
2. **Symbolic CAS Verification (SymPy Engine)**:
   - For mathematical solutions, the engine uses Computer Algebra Systems (CAS) to verify exact algebraic equivalence:
     $$\text{Simplify}(f_{\text{student}} - f_{\text{solution}}) \equiv 0$$
   - This ensures students are not penalized for unsimplified yet correct algebraic expressions or non-standard notation.
3. **Consequential Error ("נגרר") Propagation Engine**:
   - If a student commits an early calculation slip (e.g., $3+4=8$), the engine docks points solely for that arithmetic milestone. It then re-evaluates all subsequent conceptual derivations substituting the student's erroneous intermediate value. If the subsequent logical reasoning is 100% sound, full partial credit is awarded.
4. **Multi-Agent Evaluation Loop**:
   - **Method Classifier Agent**: Categorizes the approach.
   - **Rubric Matcher Agent**: Aligns steps to milestones.
   - **Adversarial Auditor Agent**: Checks for grade consistency, rubric alignment, and produces constructive, polite Hebrew feedback.

---

### 2.3 Independent Plagiarism & Collusion Detection Module
To prevent cognitive bias, the plagiarism detection engine operates as an independent batch microservice on anonymized submissions:
1. **Layer 1: Structural & Textual Semantic Embedding**:
   - Submissions are embedded using multilingual dense representation models (`BAAI/bge-m3` or `text-embedding-3-large`).
   - Pairwise cosine distance matrices identify unusually high structural similarity.
2. **Layer 2: Derivation Graph Isomorphism**:
   - Sequences of mathematical transformations are converted to Directed Acyclic Graphs (DAGs).
   - Computes Graph Edit Distance (GED) and Longest Common Subsequence (LCS) across student pairs.
3. **Layer 3: Idiosyncratic Error Fingerprinting ("The Smoking Gun")**:
   - Two students obtaining the correct answer is expected. Two students making the exact same obscure arithmetic error (e.g., $17 \times 3 = 49$), adopting identical non-standard variable names, or reproducing the same aberrant sketch layout represents statistical proof of collusion.
   - Uses Louvain community detection to isolate and cluster collusion rings.
4. **Evidence Dossier**:
   - Generates side-by-side visual diffs highlighting shared idiosyncratic derivations with confidence metrics for instructor review.

---

### 2.4 Confidence & Fallback Logic (Human-in-the-Loop)
An aggregate confidence score is calculated for every graded question:
$$S_{conf} = 0.35 C_{ocr} + 0.20 C_{layout} + 0.30 C_{eval} + 0.15 C_{symbolic}$$

- **High Confidence ($S_{conf} \ge 0.88$)**: Graded automatically; grades and feedback immediately finalized.
- **Medium Confidence ($0.72 \le S_{conf} < 0.88$)**: Added to the **10-Second Quick Verification Queue**. The instructor sees a highlighted scan crop with pre-filled recommendations and confirms with one click.
- **Low Confidence ($S_{conf} < 0.72$)**: Full manual fallback. The original scan crop is shown side-by-side with transcribed text and manual scoring sliders.
- **Active Learning**: Instructor edits and confirmations are logged to fine-tune layout heuristics and improve prompt few-shot examples.

---

### 2.5 Instructor Analytics & Global Difficulty Diagnostics
The analytics suite aggregates cohort results to give actionable pedagogical insights:
1. **Item Response Theory (IRT) Metrics**:
   - Computes difficulty parameter $b$ and discrimination index $a$ for each exam question.
2. **Unsupervised Misconception Clustering**:
   - Groups student deduction rationales using semantic clustering to identify collective failure points (e.g., "54% of students forgot to check the boundary condition at $x=0$ in Question 3b").
3. **Class Mastery Heatmap**:
   - Maps student performance to syllabus topics to guide instructor review sessions before subsequent exams.
