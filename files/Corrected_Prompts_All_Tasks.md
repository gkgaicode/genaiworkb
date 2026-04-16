# Text Processing Pipeline — Corrected Prompts (All Tasks)
> All fixes applied: unified column name (`clean_text`), state reload contract, error handling, max_features, role assignment, NLTK auto-downloads, and scipy memory calculation.

---

## GLOBAL CONVENTIONS (apply to every task)

| Convention | Value |
|---|---|
| Raw CSV file | `reviews.csv` |
| Cleaned CSV file | `cleaned_reviews.csv` |
| Raw text column | `review_text` |
| Cleaned text column | `clean_text` ← **used everywhere** |
| Rating column | `rating` |
| DataFrame variable | `df` |
| sklearn version | any ≥ 1.3 |
| max_features cap | `5000` on all vectorizers |

---

## Phase 0 — Data Collection

```
You are an expert Python web scraper. Help me scrape product reviews from
Flipkart (or Amazon) for a text processing pipeline project.

Requirements:
- Scrape minimum 100 reviews from a high-review product (e.g. earphones)
- Use BeautifulSoup + requests (or Selenium if JS-rendered)
- Handle pagination to collect enough reviews
- Extract: review_text (raw review body), rating (integer 1–5), review_title
- Add polite delays: time.sleep(random.uniform(1.5, 3.0)) between requests
- Spoof User-Agent headers to avoid blocks
- On HTTP 429 or 503: wait 5s and retry up to 3 times before failing with a
  clear message: "Scraping blocked after 3 retries on page N"
- Save output to reviews.csv with columns: review_text, rating, review_title
- Ensure no HTML tags, no encoding artifacts in review_text (strip with BeautifulSoup .get_text())
- Handle missing fields: if review_text or rating is missing, skip that review
- Print progress every 10 reviews scraped: "Scraped N reviews so far..."

After writing the code, show a sample of the first 5 rows of the CSV output.
Confirm final row count is printed: "Total reviews saved: N"
Do not scrape restricted or sensitive content.
```

**Fixes applied vs original:**
- Added retry logic for HTTP 429/503 (was missing)
- Added HTML artifact stripping via `.get_text()`
- Added explicit progress + final count confirmation
- Added random delay range instead of fixed sleep

---

## Task 1 — Text Preprocessing

```
You are an expert NLP engineer. I have a CSV file called reviews.csv with
columns: review_text (raw text), rating (integer 1–5), review_title.

Write a clean, modular Python function called preprocess(text) that performs
the following steps IN ORDER:

1. If input is NaN or empty string, return empty string immediately
2. Convert text to lowercase
3. Remove URLs using regex: re.sub(r'http\S+', '', text)
4. Remove HTML tags using regex: re.sub(r'<.*?>', '', text)
5. Remove special characters: re.sub(r'[^a-z0-9\s]', '', text)
6. Tokenize using nltk.word_tokenize(text)
7. Remove punctuation using string.punctuation check
8. Remove English stopwords using nltk.corpus.stopwords.words('english')
9. Lemmatize using nltk.stem.WordNetLemmatizer

At the top of the script, silently download all required NLTK resources:
  import nltk
  for resource in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
                   'punkt_tab', 'omw-1.4']:
      nltk.download(resource, quiet=True)

Apply preprocess() to all rows in reviews.csv:
- Store cleaned text in a new column named: clean_text
  (use this exact name — all downstream tasks depend on it)
- Print how many rows had NaN/empty review_text (were skipped)
- Print before/after example for 3 sample reviews
- Save updated dataframe to cleaned_reviews.csv (index=False)
- Confirm: "Saved cleaned_reviews.csv with N rows and columns: review_text, rating, clean_text"

List all required library imports at the top.
Add inline comments explaining each preprocessing step.
```

**Fixes applied vs original:**
- Added NaN guard as step 1 (prevents Counter crash in Task 2)
- Added explicit NLTK silent downloads including `punkt_tab` and `omw-1.4`
- Standardized output column name to `clean_text` (fixes Task 7 KeyError)
- Added confirmation print with column names for debugging

---

## Task 2 — Vocabulary Creation

```
You are an expert NLP engineer. I have a CSV file called cleaned_reviews.csv
with columns: review_text, rating, clean_text.

Write clean, modular Python code to build a vocabulary from the corpus:

1. Load cleaned_reviews.csv
   - If file not found, raise FileNotFoundError with message:
     "Run Task 1 first to generate cleaned_reviews.csv"
   - Drop rows where clean_text is NaN or empty; print count dropped

2. Flatten all tokens from clean_text into a single corpus list
   (split by whitespace: clean_text is already preprocessed text, not raw)

3. Build vocabulary manually using collections.Counter
   - Print total vocabulary size (unique tokens)
   - Print top 30 most frequent words with counts

4. Build vocabulary using sklearn CountVectorizer(max_features=5000):
   - Fit on clean_text column
   - Extract .vocabulary_ attribute
   - Print vocab size from sklearn

5. Compare: print difference in size between manual and sklearn counts
   and explain why they differ (max_features cap)

6. Plot bar chart of top 20 most frequent words using matplotlib
   - Title: "Top 20 Words by Frequency"
   - Save figure as vocab_frequency.png

7. Save full vocabulary to vocab.csv with columns: word, frequency
   (sorted by frequency descending)

8. Print 10 words from the middle of the vocabulary to show diversity

List all imports at the top. Add inline comments.
```

**Fixes applied vs original:**
- Added FileNotFoundError guard with clear message
- Added NaN drop with count printed
- Changed tokenization note (whitespace split on clean_text, not word_tokenize again)
- Added max_features=5000 to match all downstream tasks
- Added vocab_frequency.png save

---

## Task 3 — Feature Engineering

```
You are an expert NLP and Machine Learning engineer. I have a CSV file called
cleaned_reviews.csv with columns: review_text, rating, clean_text.

Write clean, modular Python code to build three feature matrices:

SETUP:
- Load cleaned_reviews.csv; drop rows where clean_text is NaN or empty
- Print dataset size: "Using N reviews for feature engineering"
- Use max_features=5000 on ALL vectorizers to keep matrices manageable

1. ONE-HOT ENCODING (OHE):
   - vectorizer_ohe = CountVectorizer(binary=True, max_features=5000)
   - ohe_matrix = vectorizer_ohe.fit_transform(df['clean_text'])
   - Print shape; show first 3 rows as DataFrame with feature names as columns

2. BAG OF WORDS (BoW):
   - vectorizer_bow = CountVectorizer(max_features=5000)
   - bow_matrix = vectorizer_bow.fit_transform(df['clean_text'])
   - Print shape; show first 3 rows as DataFrame

3. TF-IDF:
   - vectorizer_tfidf = TfidfVectorizer(max_features=5000)
   - tfidf_matrix = vectorizer_tfidf.fit_transform(df['clean_text'])
   - Print shape; show first 3 rows as DataFrame

SAVE STATE for reuse in Tasks 4, 5, 7:
   import joblib, scipy.sparse as sp
   sp.save_npz('ohe_matrix.npz', ohe_matrix)
   sp.save_npz('bow_matrix.npz', bow_matrix)
   sp.save_npz('tfidf_matrix.npz', tfidf_matrix)
   joblib.dump(vectorizer_ohe,   'vectorizer_ohe.pkl')
   joblib.dump(vectorizer_bow,   'vectorizer_bow.pkl')
   joblib.dump(vectorizer_tfidf, 'vectorizer_tfidf.pkl')
   print("Saved matrices and vectorizers to disk.")

SIDE-BY-SIDE COMPARISON:
   Show OHE, BoW, and TF-IDF values for the same first review,
   for the same 10 most common feature names.

List all imports. Add section headers and inline comments.
```

**Fixes applied vs original:**
- Added max_features=5000 (prevents memory/speed issues on real data)
- Added joblib/scipy save block so Tasks 4, 5, 7 can reload without re-fitting
- Exact variable names documented for downstream use

---

## Task 4 — Comparison Analysis

```
You are an expert NLP and Data Science engineer.

SETUP — reload saved matrices (do NOT re-fit):
   import joblib, scipy.sparse as sp
   ohe_matrix      = sp.load_npz('ohe_matrix.npz')
   bow_matrix      = sp.load_npz('bow_matrix.npz')
   tfidf_matrix    = sp.load_npz('tfidf_matrix.npz')
   vectorizer_ohe   = joblib.load('vectorizer_ohe.pkl')
   vectorizer_bow   = joblib.load('vectorizer_bow.pkl')
   vectorizer_tfidf = joblib.load('vectorizer_tfidf.pkl')
   If any file is missing, print: "Run Task 3 first to generate matrices."
   and exit.

1. COMPARISON TABLE:
   - Select first 5 reviews and 10 representative feature words
   - Build pandas DataFrame with 3 rows per review: OHE | BoW | TF-IDF
   - Print with clear column and index labels

2. TOP TF-IDF WORDS PER DOCUMENT:
   - For each of the first 5 reviews:
     * Use argsort() on the tfidf_matrix row (convert to dense first)
     * Print: "Review N → top 5 words: word(score), ..."

3. COMMON WORDS WEIGHT ANALYSIS:
   - Identify 5 most frequent words from vocab.csv (from Task 2)
   - Show their BoW counts vs TF-IDF scores side by side
   - Print numeric explanation of why common words have low TF-IDF weight

4. VISUAL COMPARISON:
   - Grouped bar chart: BoW vs TF-IDF scores for top 15 words,
     averaged across first 3 reviews
   - Use matplotlib; include legend and axis labels
   - Save as comparison_chart.png

5. WRITTEN ANALYSIS (print statements in notebook):
   - Why OHE loses frequency information
   - Why BoW is blind to word importance across documents
   - Why TF-IDF penalises high-frequency low-information words

Use .toarray() when converting sparse to dense.
List all imports. Add section headers.
```

**Fixes applied vs original:**
- Added state reload block at top (solves fresh-kernel problem)
- Added missing-file guard with clear error message
- References vocab.csv from Task 2 for common words

---

## Task 5 — Sparse Matrix Analysis

```
You are an expert NLP and Data Science engineer.

SETUP — reload saved matrices:
   import joblib, scipy.sparse as sp, sys, numpy as np, matplotlib.pyplot as plt
   ohe_matrix   = sp.load_npz('ohe_matrix.npz')
   bow_matrix   = sp.load_npz('bow_matrix.npz')
   tfidf_matrix = sp.load_npz('tfidf_matrix.npz')
   If any file is missing, print: "Run Task 3 first." and exit.

1. MATRIX SHAPE ANALYSIS:
   - Print .shape of each matrix (documents × vocab size)
   - Formatted table: Matrix | Rows | Columns | Total Elements

2. SPARSITY CALCULATION:
   - Formula: sparsity = 1 - (matrix.nnz / (rows * cols))
   - Express as percentage, 2 decimal places
   - Table: Matrix | Non-Zero Elements | Total Elements | Sparsity %

3. MEMORY USAGE — use accurate byte calculation (NOT sys.getsizeof):
   For each sparse matrix m:
     sparse_bytes = m.data.nbytes + m.indices.nbytes + m.indptr.nbytes
     dense_bytes  = m.shape[0] * m.shape[1] * 8  # float64
     saving_pct   = (1 - sparse_bytes / dense_bytes) * 100
   Print table: Matrix | Sparse KB | Dense KB | Saving %

4. SCALE SIMULATION (no actual allocation — just math):
   For doc_counts = [100, 10_000, 100_000, 1_000_000]:
     Estimate sparse and dense memory using the sparsity ratio from step 2
     Print scaling table; flag rows where dense > 1 GB with "⚠ Infeasible"

5. VISUAL ANALYSIS:
   - plt.spy() sparsity pattern for first 50 rows of all 3 matrices (subplots)
   - Bar chart: sparsity % across OHE, BoW, TF-IDF
   - Bar chart: sparse vs dense memory (KB) for all three
   - Save combined figure as sparsity_analysis.png

6. WRITTEN EXPLANATION (print statements):
   - Why NLP matrices are sparse
   - Why dense storage fails at scale
   - How CSR format stores only non-zero values
   - When to use TruncatedSVD / PCA instead

Add try/except around scale simulation. List all imports.
```

**Fixes applied vs original:**
- Replaced `sys.getsizeof()` with correct `m.data.nbytes + m.indices.nbytes + m.indptr.nbytes`
- Added state reload block
- Scale simulation uses math not allocation (no MemoryError risk)

---

## Task 6 — Real-World Questions

```
You are an expert NLP Engineer and Data Scientist with deep industry experience.
I have implemented OHE, Bag of Words, and TF-IDF on product reviews.

Answer these three questions with detailed explanations and runnable Python demos:

1. WHY BAG OF WORDS FAILS SEMANTICALLY:
   - Show BoW vector for: "The movie was fantastic" vs "The film was amazing"
     → prove zero word overlap despite identical meaning
   - Show BoW vector for: "not good" vs "good"
     → prove they look similar despite opposite meaning
   - Print the actual vectorizer vocabulary and vectors for these examples
   - List what is missing: word order, synonyms, negation, context
   - Mention solutions: Word2Vec, GloVe, BERT (do NOT benchmark them —
     just name them and explain conceptually what they add)

2. WHEN TO USE BOW VS TF-IDF:
   Use case comparison table:
   | Use Case | Recommended | Reason |
   Cover: spam detection, search ranking, document classification,
   keyword extraction, sentiment analysis, recommendation systems
   
   Then show a small runnable classification demo on 10 synthetic sentences
   comparing accuracy of CountVectorizer vs TfidfVectorizer with
   LogisticRegression. Print accuracy for both.

3. TF-IDF LIMITATIONS:
   For each limitation below, show a concrete code example AND suggest a
   modern alternative:
   - Ignores word order (n-grams partially help — show TfidfVectorizer(ngram_range=(1,2)))
   - No semantic understanding (synonyms → different features)
   - Fixed vocabulary — OOV problem (show transform on unseen word)
   - Poor on short texts (show IDF scores on a 3-word document)
   - IDF noisy on tiny corpora (show score on 5-doc corpus)

IMPORTANT: For the final summary table comparing BoW / TF-IDF / Word2Vec / BERT,
mark Word2Vec and BERT values as "(approx)" — do not present them as benchmarked
numbers since they depend heavily on model size and training data.

List all imports. Use print() for all explanatory text so it renders in notebook.
```

**Fixes applied vs original:**
- Added "(approx)" requirement for Word2Vec/BERT values (prevents hallucinated metrics)
- Added concrete runnable demo for use-case comparison
- Added n-gram example for OOV/word-order limitation

---

## Task 7 — Sentiment Classification

```
You are an expert Machine Learning engineer. I have a CSV file called
cleaned_reviews.csv with columns: review_text, rating (1–5), clean_text.

Write a complete sentiment classification pipeline:

SETUP:
   import pandas as pd
   df = pd.read_csv('cleaned_reviews.csv')
   # Validate column names
   assert 'clean_text' in df.columns, "Column 'clean_text' not found. Run Task 1 first."
   assert 'rating' in df.columns, "Column 'rating' not found."

1. LABEL CREATION:
   - positive (1): ratings 4 and 5
   - negative (0): ratings 1 and 2
   - Drop rating == 3 (ambiguous)
   - Print class distribution after filtering
   - If either class has fewer than 20 samples, raise ValueError:
     "Insufficient data: only N samples in class X. Scrape more reviews."

2. TRAIN/TEST SPLIT:
   - train_test_split(test_size=0.2, random_state=42, stratify=y)
   - Print train/test sizes

3. BOW FEATURES + MODELS:
   - CountVectorizer(max_features=5000) — fit on X_train, transform X_test
   - Train LogisticRegression(max_iter=1000, random_state=42)
   - Train MultinomialNB()
   - Evaluate each: accuracy_score + classification_report
   - Print under header: "# SCREENSHOT: BoW Classification Report"

4. TF-IDF FEATURES + MODELS:
   - TfidfVectorizer(max_features=5000) — fit on X_train, transform X_test
   - Same two models, same evaluation
   - Print under header: "# SCREENSHOT: TF-IDF Classification Report"

5. RESULTS TABLE:
   - pandas DataFrame with columns: Model, Feature, Accuracy, Precision, Recall, F1
   - Extract macro avg values from classification_report(output_dict=True)
   - Round all metrics to 4 decimal places
   - Print under header: "# SCREENSHOT: Results Comparison Table"
   - Save to results_table.csv

6. INTERPRETATION:
   - Print which model + feature combination achieved the highest F1
   - Print 3-sentence explanation of why TF-IDF down-weights filler words

List all imports. Add inline comments. Use random_state=42 everywhere.
```

**Fixes applied vs original:**
- Added role assignment ("You are an expert ML engineer")
- Fixed column name: `clean_text` (was `cleaned_review` — would cause KeyError)
- Added column assertion at top with clear error messages
- Added class-size guard (raises ValueError if < 20 samples per class)
- Added max_features=5000 to match Tasks 3–5
- Added `output_dict=True` for clean metric extraction
- Added results_table.csv save

---

## Final Deliverables Checklist & Packaging

```
I have completed a sentiment classification project using BoW and TF-IDF on
product reviews. Help me finalize and package all deliverables.

ASSUMPTIONS:
- DataFrame loaded as df with columns: review_text, rating, clean_text
- Saved artifacts exist: bow_matrix.npz, tfidf_matrix.npz, results_table.csv
- sklearn, pandas, numpy, matplotlib, joblib available
- Jupyter or JupyterLab environment

─────────────────────────────────────────
1. JUPYTER NOTEBOOK (.ipynb)
─────────────────────────────────────────
Audit my notebook so that:
- Each task has its own markdown heading: ## Task N — Name
- All cells run top-to-bottom without errors (test with Restart & Run All)
- No stack traces, no redundant prints in outputs
- Top markdown cell contains: Title, author, date, 3-line project summary
- Bottom markdown cell contains: 3–5 key takeaways
- Every cell that produces a screenshot-worthy output is labelled:
    # SCREENSHOT: <description>

─────────────────────────────────────────
2. reviews.csv
─────────────────────────────────────────
- Confirm or generate: minimum 100 rows
- Required columns: review_text, rating (integer 1–5)
- Realistic distribution (not all 5-star)
- If synthetic: vary length (short 5-word reviews to long 80-word reviews)
- Save: df.to_csv('reviews.csv', index=False)
- Print: "reviews.csv saved with N rows."

─────────────────────────────────────────
3. SCREENSHOT GUIDE
─────────────────────────────────────────
Ensure these outputs are clearly labeled in the notebook:
- BoW matrix shape + sparsity %     → # SCREENSHOT: BoW Matrix Shape
- TF-IDF matrix shape + sparsity %  → # SCREENSHOT: TF-IDF Matrix Shape
- BoW classification reports        → # SCREENSHOT: BoW Classification Report
- TF-IDF classification reports     → # SCREENSHOT: TF-IDF Classification Report
- Final results comparison table    → # SCREENSHOT: Results Comparison Table

─────────────────────────────────────────
4. WRITTEN REPORT (1–2 pages, markdown)
─────────────────────────────────────────
Write a structured report with these sections:

## Introduction
  - What the project does and why it matters (3–4 sentences)

## Observations by Task
  - Tasks 1–2: raw data characteristics, vocabulary stats
  - Tasks 3–4: preprocessing decisions and feature comparisons
  - Tasks 5–6: sparsity findings and BoW/TF-IDF trade-offs
  - Task 7: actual numbers from results_table.csv

## Conclusions
  - Best method (BoW vs TF-IDF) with F1 evidence
  - Best model (LR vs NB) with accuracy evidence
  - One limitation + one improvement suggestion

## Summary Table
  - Copy contents of results_table.csv here

Format for Google Docs / Word copy-paste.
Use plain markdown — no LaTeX, no HTML.
Save report as report.md

─────────────────────────────────────────
POWER-UP OPTIONS (append as needed):
─────────────────────────────────────────
"Also export the report as a PDF using nbconvert."
"Add a bar chart comparing F1 scores across all 4 model-feature combos."
"Generate 200 synthetic reviews instead of 100."
"Add confusion matrix heatmap for the best-performing model."
```

---

*End of corrected prompts. Use the GLOBAL CONVENTIONS table at the top to ensure consistent column names and variable names across all tasks.*
