# Automated Exam Grading & Plagiarism Detection System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Security: FERPA/GDPR Ready](https://img.shields.io/badge/Security-PII%20Anonymized-green.svg)](#security--privacy)

An enterprise-grade AI system that streamlines university-level STEM exam grading, recognizes mixed **handwritten Hebrew and mathematical formulas**, respects **creative/alternative problem-solving paths**, calculates **consequential error ("ציון נגרר")**, detects **student collusion/copying independently**, and generates **global difficulty diagnostics** for instructors.

---

## Key Capabilities

1. **Bilingual Vision-Language Transcription (Hebrew + Complex Math)**
   - Transcribes handwritten Hebrew prose (Right-to-Left / RTL) seamlessly interleaved with mathematical formulas and matrices (Left-to-Right / LTR).
   - Powered by Google Gemini 3.x with independent cross-checking via OpenAI GPT-5.6, selected empirically in the Phase 0 model bake-off (see [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md)).
   - Converts formulas directly into standard LaTeX.

2. **Multi-Column Layout Analysis & Reading Order DAG**
   - Detects vertical gutters and multi-column exam partitions.
   - Enforces correct reading order: Right column $\to$ Left column for Hebrew, with LTR formula sub-blocks.
   - Automatically detects and ignores crossed-out / strikethrough scratch work.

3. **Semantic & Flexible Grading Engine ("Diversity of Solutions")**
   - **Atomic Rubric Graph (ARG)**: Deconstructs the instructor's solution into atomic milestones with parallel valid solving trajectories (e.g., energy conservation vs. Newton's laws).
   - **Symbolic Equivalence (SymPy CAS)**: Verifies $\text{Simplify}(f_{\text{student}} - f_{\text{solution}}) \equiv 0$ so valid unsimplified forms receive full credit.
   - **Consequential Error ("נגרר") Engine**: An arithmetic slip docks points only once. Subsequent steps that logically follow from the student's intermediate value receive full partial credit.
   - Generates constructive, encouraging pedagogical feedback in Hebrew.

4. **Decoupled Plagiarism & Collusion Detection**
   - Runs independently of individual grading to prevent bias.
   - **3-Layer Detection**:
     1. Dense vector semantic similarity.
     2. Mathematical derivation graph isomorphism.
     3. **Idiosyncratic Error Fingerprinting**: Flags shared anomalous errors (e.g. two students making the exact same bizarre arithmetic slip or using identical rare variable notations).
   - Generates side-by-side visual diff reports for instructor review.

5. **Human-in-the-Loop (HITL) Verification Workbench**
   - Computes an aggregate confidence score ($S_{conf}$) per question.
   - High-confidence answers ($\ge 88\%$) are auto-finalized.
   - Borderline answers ($72\% - 87\%$) enter a **10-second rapid verification queue** with pre-filled recommendations for one-click approval.

6. **Global Difficulty & Cohort Analytics**
   - Calculates Item Response Theory (IRT) difficulty ($b$) and discrimination ($a$) parameters per question.
   - Automatically clusters common misconceptions to show instructors where the class struggled globally.

---

## Quickstart Guide

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 installed
- Git

### 2. Setup Repository
```bash
# Clone the repository
git clone <your-repo-url>
cd "Exam grading system"

# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Secrets Safely
```bash
# Copy the example environment file
cp .env.example .env
```
Open `.env` and add your personal API keys (e.g., `GEMINI_API_KEY`, `OPENAI_API_KEY`).  
> **Warning**: Never commit your `.env` file to Git! It is already listed in `.gitignore`.

---

## Project Structure

```
├── backend/                  # Python backend services
│   ├── config.py             # Settings and environment variables
│   ├── schemas.py            # Pydantic data schemas
│   ├── preprocessing/        # Image deskewing, CLAHE, PII anonymization
│   ├── layout/               # Multi-column analyzer & reading order DAG
│   ├── transcription/        # VLM transcription engine (Hebrew + LaTeX)
│   ├── grading/              # Atomic rubrics, SymPy CAS, Consequential error
│   ├── plagiarism/           # Decoupled collusion & error fingerprinting
│   ├── analytics/            # IRT metrics & misconception clustering
│   └── api/                  # FastAPI endpoints & static file server
├── frontend/                 # Instructor dashboard & HITL workbench
├── storage/                  # Local storage for uploads, crops, and reports
│   ├── uploads/              # Raw exam scans (Git ignored)
│   ├── processed/            # Anonymized & deskewed images (Git ignored)
│   └── crops/                # Segmented question crops (Git ignored)
├── docs/                     # Architectural documentation
├── .env.example              # Template for environment variables
├── .gitignore                # Protects secrets and student PII
├── requirements.txt          # Python package dependencies
├── TODO.md                   # Interactive master checklist for tracking tasks
├── IMPLEMENTATION_PLAN.md    # Detailed system implementation specification
└── README.md                 # Project overview and instructions
```

---

## Security & Public Repository Safety

Because this repository is intended to be public, strict security practices must be followed:

1. **Zero Secret Leakage**:
   - Never hardcode API keys, passwords, or tokens in source code.
   - All credentials must be loaded via `.env` using `backend/config.py`.
   - Before pushing to Git, run `git status` to ensure no `.env` or sensitive files are staged.
2. **Student Privacy (FERPA / GDPR)**:
   - The preprocessing module automatically applies a digital anonymization bar over student identity headers and assigns a pseudonym (`ANON_...`).
   - The `storage/` folder is included in `.gitignore` to ensure no student exam scans are committed to the public repository.

---

## Partner Collaboration & Progress Tracking

- Use **[`TODO.md`](./TODO.md)** to track and divide tasks between you and your partner.
- Each task includes an interactive checkbox (`- [ ]`) grouped by phase.
- Consult **[`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md)** for detailed technical requirements and architectural decisions for each module.
