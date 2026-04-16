Here's your ready-to-use prompt:

---

**📋 Prompt — Final Deliverables Checklist & Packaging**

```
I have completed a sentiment classification project using BoW and TF-IDF on 
product reviews. Help me finalize and package all deliverables to a professional 
standard. Here is what I need:

─────────────────────────────────────────
1. JUPYTER NOTEBOOK (.ipynb)
─────────────────────────────────────────
Audit and clean my existing notebook so that:
- Each task has its own cell block with a markdown heading 
  (e.g., ## Task 1 — Data Loading, ## Task 7 — Sentiment Classification)
- All cells run top-to-bottom without errors (restart & run all)
- Outputs are clean: no stack traces, no redundant print statements
- Add a markdown cell at the very top with:
    Title, author name, date, and a 3-line project summary
- Add a markdown cell at the very bottom with:
    Key takeaways (3–5 bullet points)

─────────────────────────────────────────
2. reviews.csv
─────────────────────────────────────────
- Confirm or generate a CSV with at minimum 100 rows
- Required columns: review_text (raw text) and rating (integer 1–5)
- Distribution should be realistic: not all 5-star, mix of sentiments
- If generating synthetic data, make reviews feel natural and varied in length
- Save with: df.to_csv('reviews.csv', index=False)

─────────────────────────────────────────
3. SCREENSHOTS GUIDE
─────────────────────────────────────────
Add print statements or display calls in the notebook that clearly output 
the following, so they are easy to screenshot:
- Sparse matrix shapes for both BoW and TF-IDF 
  (e.g., "BoW matrix shape: (800, 5000), Sparsity: 98.3%")
- Full classification_report for every model × feature combination
- The final results comparison table (model × feature → accuracy, precision, 
  recall, F1)
Label each output block with a comment like: # SCREENSHOT: Matrix Shape

─────────────────────────────────────────
4. WRITTEN REPORT (1–2 pages)
─────────────────────────────────────────
Write a structured 1–2 page report in markdown with these sections:

## Introduction
  - What the project does and why it matters (3–4 sentences)

## Observations by Task
  - Task 1–2: Data loading and exploration — what the data looked like
  - Task 3–4: Preprocessing — what cleaning steps were applied and why
  - Task 5–6: Feature engineering — BoW vs TF-IDF mechanics briefly explained
  - Task 7: Model results — reference actual numbers from the results table

## Conclusions
  - Which method performed best (BoW or TF-IDF) and why
  - Which model performed best (Logistic Regression or Naive Bayes) and why
  - One limitation of this approach and one suggestion for improvement

## Summary Table
  - Reprint the model × feature results table here

Format the report so it can be copied into Google Docs or Word as-is.
Use plain markdown — no LaTeX, no HTML.

─────────────────────────────────────────
ASSUMPTIONS
─────────────────────────────────────────
- Dataframe is loaded as df with columns review_text and rating
- sklearn, pandas, numpy, and matplotlib are available
- Notebook is in Jupyter or JupyterLab environment
```

---

**💡 Power-up options — append any of these to the prompt:**

| Add this line | What it does |
|---|---|
| *"Also export the report as a .md file"* | Saves report directly to disk |
| *"Add a bar chart comparing F1 scores across all models"* | Visual for the report |
| *"Flag any cells that take >10s to run with a ⚠️ warning comment"* | Performance awareness |
| *"Generate 200 synthetic reviews instead of 100"* | Bigger, more robust dataset |

---

Want me to generate the actual report text or the synthetic `reviews.csv` data right now?