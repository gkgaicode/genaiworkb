Here's a ready-to-use prompt you can paste directly into Claude or any AI coding assistant:

---

**📋 Prompt — Task 7: Sentiment Classification (BoW vs TF-IDF)**

```
I have a product reviews dataset with a 'rating' column (1–5 stars) and a 
'cleaned_review' column (preprocessed text). Help me complete a full sentiment 
classification pipeline:

1. LABEL CREATION
   - Map ratings 4–5 → positive (1), 1–2 → negative (0)
   - Drop 3-star reviews (ambiguous)
   - Show class distribution after filtering

2. TRAIN/TEST SPLIT
   - Use train_test_split(test_size=0.2, random_state=42, stratify=y)

3. BOW FEATURES + MODELS
   - Vectorize with CountVectorizer on train set, transform test set
   - Train LogisticRegression(max_iter=1000) and MultinomialNB
   - Evaluate both with accuracy_score and classification_report

4. TF-IDF FEATURES + MODELS
   - Repeat step 3 using TfidfVectorizer instead of CountVectorizer
   - Same two models, same evaluation

5. RESULTS TABLE
   - Build a pandas DataFrame comparing all 4 combinations:
     model × feature_type → Accuracy, Precision, Recall, F1 (macro avg)
   - Print the table cleanly

6. INTERPRETATION
   - Add a brief printed summary explaining why TF-IDF typically outperforms BoW
     (down-weighting high-frequency filler words, surfacing discriminative terms)

Use sklearn. Assume the dataframe is already loaded as `df` with columns 
'cleaned_review' and 'rating'. Write clean, commented, production-ready code.
```

---

**💡 Tips for best results:**

- If your dataframe or column names differ, mention them upfront (e.g. `df['review_text']`, `df['stars']`)
- Add *"also plot a confusion matrix for the best model"* if you want a visual
- Add *"use cross-validation instead of a single split"* for more robust evaluation
- If you're in a Jupyter notebook, add *"display results in a styled DataFrame"*

Want me to also generate the actual Python code for this task right now?