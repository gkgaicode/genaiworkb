# Get me good prompt to code : Task 2 — Vocabulary Creation

## 🗒️ Answer

Here's an effective prompt you can use directly with Claude or any AI coding assistant for Task 2 (Vocabulary Creation):

---

**Prompt:**

```
You are an expert NLP engineer. I have a CSV file called cleaned_reviews.csv 
with a column named clean_text containing preprocessed product reviews.

Write clean, modular Python code to build a vocabulary from the corpus and 
perform the following steps IN ORDER:

1. Load cleaned_reviews.csv and read the clean_text column
2. Flatten all tokenized words into a single corpus list
3. Build vocabulary manually using Python's collections.Counter
4. Print total vocabulary size (number of unique tokens)
5. Print top 30 most frequent words with their counts
6. Build vocabulary alternatively using sklearn's CountVectorizer:
   - Fit on clean_text column
   - Extract .vocabulary_ attribute
   - Print vocab size from sklearn
7. Compare manual vs sklearn vocabulary sizes
8. Plot a bar chart of top 20 most frequent words using matplotlib
9. Save full vocabulary with word frequencies to vocab.csv 
   with columns: word, frequency

Additional requirements:
- Add inline comments explaining each step
- Handle empty or NaN rows gracefully
- List all required imports at the top
- Print a sample of 10 words from the middle of the vocabulary
  to show diversity beyond top words

Show all printed outputs and display the bar chart.
```

---

This prompt is structured, comprehensive, and requests both manual and 
sklearn-based approaches with visual output — essential for a complete 
vocabulary analysis. [[5](https://www.altersquare.io/5-ai-prompts-every-developer-should-master-copy-paste-ready/)] [[6](https://towardsdatascience.com/advanced-prompt-engineering-for-data-science-projects/)]

## 🌐 Sources

1. [reddit.com - I compiled 200 advanced Claude prompts for coding](https://www.reddit.com/r/PromptEngineering/comments/1sfcosw/i_compiled_200_advanced_claude_prompts_for_coding/)
2. [medium.com - Ultimate Prompts for Every Developer | by Onix React](https://medium.com/@onix_react/ultimate-prompts-for-every-developer-031a6d26a569)
3. [community.openai.com - Sourcing useful ChatGPT coding prompts to feature - Codex](https://community.openai.com/t/sourcing-useful-chatgpt-coding-prompts-to-feature/1357452)
4. [dataplatform.cloud.ibm.com - Sample foundation model prompts for common tasks](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-prompt-samples.html?context=wx)
5. [altersquare.io - 5 AI Prompts Every Developer Should Master](https://www.altersquare.io/5-ai-prompts-every-developer-should-master-copy-paste-ready/)
6. [towardsdatascience.com - Advanced Prompt Engineering for Data Science Projects](https://towardsdatascience.com/advanced-prompt-engineering-for-data-science-projects/)