# Local exam dataset (private)

Place exam PDFs here locally. This folder is gitignored — do not commit student scans, solutions, or grade CSVs.

Layout used by the pipeline:

```
docs/Dataset/
  exams/        one PDF per student booklet, named by booklet ID (02.pdf, 05.pdf, …)  → Phase 1 input
  sol.pdf       instructor solution; correct options in red                         → Phase 4 answer key
  grades.csv    instructor grades, rows "id,grade" (no header)                       → Phase 4 evaluation
```

Keep `sol.pdf` out of `exams/`, otherwise Phase 1 treats it as a student booklet.
