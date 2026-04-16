# Get me effective prompt you can use directly with Claude or any AI coding assistant for : Task 3 — Feature Engineering

## 🗒️ Answer

Here's an effective prompt you can use directly with Claude or any AI coding assistant for Task 3 (Feature Engineering):

---

**Prompt:**

```
You are an expert NLP and Machine Learning engineer. I have a CSV file called 
cleaned_reviews.csv with a column named clean_text containing preprocessed 
product reviews.

Write clean, modular Python code to implement three feature engineering 
techniques IN ORDER:

1. ONE-HOT ENCODING (OHE):
   - Use CountVectorizer(binary=True) from sklearn
   - Fit and transform clean_text column
   - Print shape of OHE matrix
   - Show first 3 rows as a readable DataFrame with feature names

2. BAG OF WORDS (BoW):
   - Use CountVectorizer() from sklearn
   - Fit and transform clean_text column
   - Print shape of BoW matrix
   - Show first 3 rows as a readable DataFrame with feature names

3. TF-IDF:
   - Use TfidfVectorizer() from sklearn
   - Fit and transform clean_text column
   - Print shape of TF-IDF matrix
   - Show first 3 rows as a readable DataFrame with feature names

Additional requirements:
- Store all three vectorizers and matrices as named variables
  for reuse in later tasks
- Add inline comments explaining each technique
- List all required imports at the top
- Print a clear section header before each technique
- Save all three matrices as: ohe_matrix, bow_matrix, tfidf_matrix
- Display a side-by-side sample comparison of all three 
  representations for the same review
```

---

This prompt is precise, ordered, and structured for modular reuse in downstream tasks — following best practices for AI-assisted coding. [[1](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)] [[5](https://pub.towardsai.net/the-prompts-i-actually-use-to-code-with-ai-assistants-a442597bd6c0)]

## 🌐 Sources

1. [platform.claude.com - Prompting best practices - Claude API Docs](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)
2. [sderosiaux.medium.com - How I Learned to Prompt Claude Code Better — Four Modes](https://sderosiaux.medium.com/how-i-learned-to-prompt-ai-better-my-four-modes-177bddcfa6bd)
3. [reddit.com - The Only Prompt You Need : r/ClaudeAI](https://www.reddit.com/r/ClaudeAI/comments/1gds696/the_only_prompt_you_need/)
4. [github.com - langgptai/awesome-claude-prompts](https://github.com/langgptai/awesome-claude-prompts)
5. [pub.towardsai.net - The Prompts I Actually Use to Code with AI Assistants](https://pub.towardsai.net/the-prompts-i-actually-use-to-code-with-ai-assistants-a442597bd6c0)
6. [aws.amazon.com - Prompt engineering techniques and best practices](https://aws.amazon.com/blogs/machine-learning/prompt-engineering-techniques-and-best-practices-learn-by-doing-with-anthropics-claude-3-on-amazon-bedrock/)